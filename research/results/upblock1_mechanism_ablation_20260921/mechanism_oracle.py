"""Causal mechanism ablation at logical DeepCache up_block_1."""

from __future__ import annotations

import time
from typing import Any

import torch

from DeepCache.extension.deepcache import DeepCacheSDHelper
from research.results.block_feature_oracle_20260921.block_replacement_oracle import (
    guided_noise,
    relative_l1,
)

TARGET_KEY = ("up", "block", 1, 0)
PROBE_KINDS = ("activation_only", "suffix_only", "activation_plus_suffix")


class UpBlock1MechanismOracle(DeepCacheSDHelper):
    """Disentangle teacher activation replacement from suffix recomputation."""

    def __init__(self, pipe, refresh_steps, probe_steps, guidance_scale=7.5):
        super().__init__(pipe)
        self.set_params(cache_interval=1, cache_branch_id=0)
        self.refresh_steps = frozenset(refresh_steps)
        self.probe_steps = frozenset(probe_steps)
        self.guidance_scale = guidance_scale
        self.mode = "idle"
        self.teacher_target: Any = None
        self.probe_kind: str | None = None
        self.after_target = False
        self.target_hit = False
        self.trace = []
        self.last_refresh_step = 0

    def enable(self):
        self.timesteps = [int(t) for t in self.pipe.scheduler.timesteps]
        if len(self.timesteps) != 20 or len(set(self.timesteps)) != 20:
            raise ValueError("This experiment requires 20 unique timesteps")
        super().enable()

    def is_refresh_step(self):
        return self.cur_timestep in self.refresh_steps

    def is_skip_step(self, block_i, layer_i, blocktype="down"):
        if self.is_refresh_step():
            return False
        cache_block_id = self.params["cache_block_id"]
        cache_layer_id = self.params["cache_layer_id"]
        if block_i > cache_block_id or blocktype == "mid":
            return True
        if block_i < cache_block_id:
            return False
        return layer_i >= cache_layer_id if blocktype == "down" else layer_i > cache_layer_id

    @staticmethod
    def probe_action(kind):
        if kind == "activation_only":
            return True, False
        if kind == "suffix_only":
            return False, True
        if kind == "activation_plus_suffix":
            return True, True
        raise ValueError(f"Unknown probe kind: {kind}")

    @staticmethod
    def output_tensor(value):
        if torch.is_tensor(value):
            return value
        sample = getattr(value, "sample", None)
        if torch.is_tensor(sample):
            return sample
        if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
            return value[0]
        raise TypeError(f"Cannot locate output tensor in {type(value)!r}")

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key = (blocktype, block_name, block_i, layer_i)
        original = block.forward
        self.function_dict[key] = original

        def forward(*args, **kwargs):
            if self.mode == "teacher":
                result = original(*args, **kwargs)
                if key == TARGET_KEY:
                    self.teacher_target = result
                if self.is_refresh_step():
                    self.cached_output[key] = result
                return result

            if self.mode == "baseline":
                if self.is_skip_step(block_i, layer_i, blocktype):
                    return self.cached_output[key]
                result = original(*args, **kwargs)
                self.cached_output[key] = result
                return result

            if self.mode == "probe":
                if key == TARGET_KEY:
                    use_teacher, recompute_suffix = self.probe_action(self.probe_kind)
                    self.target_hit = True
                    self.after_target = recompute_suffix
                    return self.teacher_target if use_teacher else self.cached_output[key]
                if self.after_target:
                    return original(*args, **kwargs)
                if self.is_skip_step(block_i, layer_i, blocktype):
                    return self.cached_output[key]
                return original(*args, **kwargs)

            return original(*args, **kwargs)

        block.forward = forward

    def wrap_unet_forward(self):
        original_unet_forward = self.pipe.unet.forward
        self.function_dict["unet_forward"] = original_unet_forward

        def forward(*args, **kwargs):
            timestep_value = args[1] if len(args) > 1 else kwargs["timestep"]
            timestep = int(timestep_value.flatten()[0].item()) if torch.is_tensor(timestep_value) else int(timestep_value)
            step = len(self.trace)
            if step >= len(self.timesteps) or timestep != self.timesteps[step]:
                raise RuntimeError(f"Unexpected scheduler timestep {timestep} at index {step}")
            self.cur_timestep = step
            refresh = self.is_refresh_step()

            self.teacher_target = None
            self.mode = "teacher"
            teacher_result = original_unet_forward(*args, **kwargs)
            if self.teacher_target is None:
                raise RuntimeError("Teacher up_block_1 activation was not captured")
            teacher_noise = self.output_tensor(teacher_result)

            if refresh:
                baseline_result = teacher_result
                self.last_refresh_step = step
                cache_after = dict(self.cached_output)
            else:
                cache_before = dict(self.cached_output)
                self.mode = "baseline"
                baseline_result = original_unet_forward(*args, **kwargs)
                cache_after = dict(self.cached_output)
            baseline_noise = self.output_tensor(baseline_result)
            teacher_guided = guided_noise(teacher_noise, self.guidance_scale)
            baseline_guided = guided_noise(baseline_noise, self.guidance_scale)
            baseline_error = relative_l1(teacher_guided, baseline_guided)

            row = {
                "step": step,
                "timestep": timestep,
                "refresh": refresh,
                "age": 0 if refresh else step - self.last_refresh_step,
                "baseline_guided_noise_error": baseline_error,
                "probes": {},
            }

            if not refresh and step in self.probe_steps:
                for kind in PROBE_KINDS:
                    self.cached_output = dict(cache_before)
                    self.mode = "probe"
                    self.probe_kind = kind
                    self.after_target = False
                    self.target_hit = False
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    probe_result = original_unet_forward(*args, **kwargs)
                    end.record()
                    if not self.target_hit:
                        raise RuntimeError(f"Probe target not reached for {kind}")
                    torch.cuda.synchronize()
                    probe_noise = self.output_tensor(probe_result)
                    probe_error = relative_l1(
                        teacher_guided, guided_noise(probe_noise, self.guidance_scale)
                    )
                    recovery = (
                        (baseline_error - probe_error) / baseline_error
                        if baseline_error > 1e-12
                        else 0.0
                    )
                    row["probes"][kind] = {
                        "guided_noise_error": probe_error,
                        "error_reduction": baseline_error - probe_error,
                        "recovery_fraction": recovery,
                        "milliseconds": start.elapsed_time(end),
                    }

            self.cached_output = cache_after
            self.mode = "idle"
            self.probe_kind = None
            self.after_target = False
            self.trace.append(row)
            return teacher_result

        self.pipe.unet.forward = forward
