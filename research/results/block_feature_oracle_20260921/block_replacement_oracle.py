"""Block feature profiling and activation-replacement oracle for DeepCache.

The oracle follows the full (uncached) DPM-Solver++ trajectory.  On reuse steps it
also evaluates a scheduled DeepCache pass.  For selected steps, one cached block
activation is replaced by the activation from the full pass and the remaining
suffix is recomputed.  These probe passes never advance the scheduler or mutate
persistent cache state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from DeepCache.extension.deepcache import DeepCacheSDHelper


BlockKey = tuple[str, str, int, int]


TARGETS: dict[str, BlockKey] = {
    "down_block_0": ("down", "block", 0, 0),
    "down_block_1": ("down", "block", 1, 0),
    "down_block_2": ("down", "block", 2, 0),
    "down_block_3": ("down", "block", 3, 0),
    "mid_block": ("mid", "mid_block", 0, 0),
    # Up-block indices below use DeepCache's deep-to-shallow logical indexing.
    "up_block_3": ("up", "block", 3, 0),
    "up_block_2": ("up", "block", 2, 0),
    "up_block_1": ("up", "block", 1, 0),
    "up_boundary": ("up", "attentions", 0, 1),
}
KEY_TO_NAME = {value: key for key, value in TARGETS.items()}


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, Mapping):
        result: list[torch.Tensor] = []
        for child in value.values():
            result.extend(_tensor_leaves(child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for child in value:
            result.extend(_tensor_leaves(child))
        return result
    sample = getattr(value, "sample", None)
    if torch.is_tensor(sample):
        return [sample]
    return []


def tree_error(reference: Any, estimate: Any) -> dict[str, float] | None:
    """Return aggregate relative-L1 and cosine distance for matching tensor trees."""
    ref_leaves = _tensor_leaves(reference)
    est_leaves = _tensor_leaves(estimate)
    if len(ref_leaves) != len(est_leaves) or not ref_leaves:
        return None

    abs_error = 0.0
    abs_reference = 0.0
    dot = 0.0
    ref_square = 0.0
    est_square = 0.0
    elements = 0
    for ref, est in zip(ref_leaves, est_leaves):
        if ref.shape != est.shape:
            return None
        ref_f = ref.detach().float()
        est_f = est.detach().float()
        abs_error += torch.sum(torch.abs(est_f - ref_f)).item()
        abs_reference += torch.sum(torch.abs(ref_f)).item()
        dot += torch.sum(ref_f * est_f).item()
        ref_square += torch.sum(ref_f * ref_f).item()
        est_square += torch.sum(est_f * est_f).item()
        elements += ref.numel()

    cosine = dot / max(math.sqrt(ref_square * est_square), 1e-12)
    return {
        "relative_l1": abs_error / max(abs_reference, 1e-12),
        "cosine_distance": 1.0 - cosine,
        "numel": elements,
    }


def relative_l1(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.mean(torch.abs(estimate.float() - reference.float()))
    denominator = torch.mean(torch.abs(reference.float())).clamp_min(1e-12)
    return (numerator / denominator).item()


def guided_noise(noise: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    unconditional, conditional = noise.chunk(2)
    return unconditional + guidance_scale * (conditional - unconditional)


class BlockReplacementOracleHelper(DeepCacheSDHelper):
    """Scheduled DeepCache with feature capture and causal replacement probes."""

    def __init__(
        self,
        pipe,
        refresh_steps: list[int],
        replacement_steps: list[int],
        guidance_scale: float = 7.5,
    ):
        super().__init__(pipe=pipe)
        self.refresh_steps = frozenset(refresh_steps)
        self.replacement_steps = frozenset(replacement_steps)
        self.guidance_scale = guidance_scale
        self.mode = "idle"
        self.after_target = False
        self.replacement_target: BlockKey | None = None
        self.replacement_hit = False
        self.teacher_outputs: dict[BlockKey, Any] = {}
        self.baseline_outputs: dict[BlockKey, Any] = {}
        self.trace: list[dict[str, Any]] = []
        self.last_refresh_step = 0

    def enable(self):
        self.timesteps = [int(t) for t in self.pipe.scheduler.timesteps]
        if len(self.timesteps) != 20 or len(set(self.timesteps)) != 20:
            raise ValueError("This experiment requires 20 unique scheduler timesteps")
        super().enable()

    def is_refresh_step(self) -> bool:
        return self.cur_timestep in self.refresh_steps

    def is_skip_step(self, block_i: int, layer_i: int, blocktype: str = "down") -> bool:
        if self.is_refresh_step():
            return False
        cache_block_id = self.params["cache_block_id"]
        cache_layer_id = self.params["cache_layer_id"]
        if block_i > cache_block_id or blocktype == "mid":
            return True
        if block_i < cache_block_id:
            return False
        if blocktype == "down":
            return layer_i >= cache_layer_id
        return layer_i > cache_layer_id

    @staticmethod
    def _output_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        sample = getattr(value, "sample", None)
        if torch.is_tensor(sample):
            return sample
        if isinstance(value, Sequence) and value and torch.is_tensor(value[0]):
            return value[0]
        raise TypeError(f"Cannot locate output tensor in {type(value)!r}")

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key: BlockKey = (blocktype, block_name, block_i, layer_i)
        original = block.forward
        self.function_dict[key] = original

        def forward(*args, **kwargs):
            if self.mode == "teacher":
                result = original(*args, **kwargs)
                if key in KEY_TO_NAME:
                    self.teacher_outputs[key] = result
                if self.is_refresh_step():
                    self.cached_output[key] = result
                return result

            if self.mode == "baseline":
                if self.is_skip_step(block_i, layer_i, blocktype):
                    result = self.cached_output[key]
                else:
                    result = original(*args, **kwargs)
                    self.cached_output[key] = result
                if key in KEY_TO_NAME:
                    self.baseline_outputs[key] = result
                return result

            if self.mode == "replacement":
                if key == self.replacement_target:
                    if key not in self.teacher_outputs:
                        raise RuntimeError(f"Missing teacher activation for {key}")
                    self.replacement_hit = True
                    self.after_target = True
                    return self.teacher_outputs[key]
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
            timestep = args[1] if len(args) > 1 else kwargs["timestep"]
            if torch.is_tensor(timestep):
                timestep = int(timestep.flatten()[0].item())
            elif not isinstance(timestep, int):
                timestep = int(timestep)
            step_index = len(self.trace)
            if step_index >= len(self.timesteps) or timestep != self.timesteps[step_index]:
                raise RuntimeError(
                    f"Unexpected scheduler timestep {timestep} at index {step_index}"
                )
            self.cur_timestep = step_index
            refresh = self.is_refresh_step()

            self.teacher_outputs = {}
            self.baseline_outputs = {}
            self.mode = "teacher"
            teacher_result = original_unet_forward(*args, **kwargs)
            teacher_noise = self._output_tensor(teacher_result)

            if refresh:
                baseline_result = teacher_result
                baseline_noise = teacher_noise
                self.baseline_outputs = dict(self.teacher_outputs)
                self.last_refresh_step = step_index
                cache_after = dict(self.cached_output)
            else:
                cache_before = dict(self.cached_output)
                self.mode = "baseline"
                baseline_result = original_unet_forward(*args, **kwargs)
                baseline_noise = self._output_tensor(baseline_result)
                cache_after = dict(self.cached_output)

            teacher_guided = guided_noise(teacher_noise, self.guidance_scale)
            baseline_guided = guided_noise(baseline_noise, self.guidance_scale)
            baseline_error = relative_l1(teacher_guided, baseline_guided)

            row: dict[str, Any] = {
                "step": step_index,
                "timestep": timestep,
                "refresh": refresh,
                "age": 0 if refresh else step_index - self.last_refresh_step,
                "baseline_noise_error": relative_l1(teacher_noise, baseline_noise),
                "baseline_guided_noise_error": baseline_error,
                "blocks": {},
                "replacement": {},
            }
            for name, key in TARGETS.items():
                if key in self.teacher_outputs and key in self.baseline_outputs:
                    block_metrics = tree_error(self.teacher_outputs[key], self.baseline_outputs[key])
                    if block_metrics is not None:
                        row["blocks"][name] = block_metrics

            if not refresh and step_index in self.replacement_steps:
                timing_events = []
                for name, key in TARGETS.items():
                    if key not in self.teacher_outputs:
                        continue
                    self.cached_output = dict(cache_before)
                    self.mode = "replacement"
                    self.replacement_target = key
                    self.replacement_hit = False
                    self.after_target = False
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    replacement_result = original_unet_forward(*args, **kwargs)
                    end_event.record()
                    if not self.replacement_hit:
                        raise RuntimeError(f"Replacement target was not reached: {name} {key}")
                    replacement_noise = self._output_tensor(replacement_result)
                    replacement_guided = guided_noise(replacement_noise, self.guidance_scale)
                    replacement_error = relative_l1(teacher_guided, replacement_guided)
                    recovery = (
                        (baseline_error - replacement_error) / baseline_error
                        if baseline_error > 1e-12
                        else 0.0
                    )
                    row["replacement"][name] = {
                        "guided_noise_error": replacement_error,
                        "error_reduction": baseline_error - replacement_error,
                        "recovery_fraction": recovery,
                    }
                    timing_events.append((name, start_event, end_event))
                torch.cuda.synchronize()
                for name, start_event, end_event in timing_events:
                    row["replacement"][name]["suffix_milliseconds"] = start_event.elapsed_time(end_event)

            self.cached_output = cache_after
            self.mode = "idle"
            self.replacement_target = None
            self.after_target = False
            self.trace.append(row)
            return teacher_result

        self.pipe.unet.forward = forward
