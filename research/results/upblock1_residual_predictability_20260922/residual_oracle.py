"""Teacher-forced residual predictability helpers for logical up_block_1."""
from __future__ import annotations

from typing import Any
import torch
from DeepCache.extension.deepcache import DeepCacheSDHelper

TARGET_KEY = ("up", "block", 1, 0)
MODELS = ("channel_mean", "channel_affine", "oracle")


def tensor_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    sample = getattr(value, "sample", None)
    if torch.is_tensor(sample):
        return sample
    if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"Cannot extract tensor from {type(value)!r}")


def relative_l1(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return (torch.mean(torch.abs(estimate.float() - reference.float())) /
            torch.mean(torch.abs(reference.float())).clamp_min(1e-12)).item()


def guided_noise(noise: torch.Tensor, scale: float = 7.5) -> torch.Tensor:
    uncond, cond = noise.chunk(2)
    return uncond + scale * (cond - uncond)


def channel_stats(cached: torch.Tensor, teacher: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """Sufficient statistics for y=teacher-cached versus cached, per channel."""
    if cached.shape != teacher.shape or cached.ndim != 4:
        raise ValueError("Expected matching NCHW activation tensors")
    x = cached.detach().float()
    y = teacher.detach().float() - x
    dims = (0, 2, 3)
    return {
        "n": x.shape[0] * x.shape[2] * x.shape[3],
        "sum_x": x.sum(dims).cpu().double(),
        "sum_y": y.sum(dims).cpu().double(),
        "sum_x2": (x * x).sum(dims).cpu().double(),
        "sum_xy": (x * y).sum(dims).cpu().double(),
    }


def combine_stats(items: list[dict]) -> dict:
    if not items:
        raise ValueError("No statistics to combine")
    return {
        "n": sum(int(item["n"]) for item in items),
        **{key: sum((item[key] for item in items[1:]), items[0][key].clone())
           for key in ("sum_x", "sum_y", "sum_x2", "sum_xy")},
    }


def fit_channel_models(stats: dict) -> dict[str, torch.Tensor]:
    n = float(stats["n"])
    mean_x = stats["sum_x"] / n
    mean_y = stats["sum_y"] / n
    covariance = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / n
    variance = stats["sum_x2"] - stats["sum_x"] * stats["sum_x"] / n
    slope = torch.where(variance.abs() > 1e-12, covariance / variance, torch.zeros_like(variance))
    intercept = mean_y - slope * mean_x
    return {"mean": mean_y.float(), "slope": slope.float(), "intercept": intercept.float()}


class ScheduledHelper(DeepCacheSDHelper):
    def __init__(self, pipe, refresh_steps, probe_steps, guidance_scale=7.5):
        super().__init__(pipe)
        self.set_params(cache_interval=1, cache_branch_id=0)
        self.refresh_steps = frozenset(refresh_steps)
        self.probe_steps = frozenset(probe_steps)
        self.guidance_scale = guidance_scale
        self.trace = []
        self.mode = "idle"
        self.teacher_target = None
        self.baseline_target = None
        self.last_refresh_step = 0

    def enable(self):
        self.timesteps = [int(t) for t in self.pipe.scheduler.timesteps]
        if len(self.timesteps) != 20 or len(set(self.timesteps)) != 20:
            raise ValueError("Expected 20 unique scheduler timesteps")
        super().enable()

    def is_refresh_step(self):
        return self.cur_timestep in self.refresh_steps

    def is_skip_step(self, block_i, layer_i, blocktype="down"):
        if self.is_refresh_step():
            return False
        cb = self.params["cache_block_id"]
        cl = self.params["cache_layer_id"]
        if block_i > cb or blocktype == "mid":
            return True
        if block_i < cb:
            return False
        return layer_i >= cl if blocktype == "down" else layer_i > cl


class StatsCollectorHelper(ScheduledHelper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sufficient_stats = {}

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key = (blocktype, block_name, block_i, layer_i)
        original = block.forward
        self.function_dict[key] = original

        def forward(*args, **kwargs):
            if self.mode == "teacher":
                result = original(*args, **kwargs)
                if key == TARGET_KEY:
                    self.teacher_target = tensor_output(result)
                if self.is_refresh_step():
                    self.cached_output[key] = result
                return result
            if self.mode == "baseline":
                if self.is_skip_step(block_i, layer_i, blocktype):
                    result = self.cached_output[key]
                else:
                    result = original(*args, **kwargs)
                    self.cached_output[key] = result
                if key == TARGET_KEY:
                    self.baseline_target = tensor_output(result)
                return result
            return original(*args, **kwargs)
        block.forward = forward

    def wrap_unet_forward(self):
        original = self.pipe.unet.forward
        self.function_dict["unet_forward"] = original

        def forward(*args, **kwargs):
            step = len(self.trace)
            timestep_arg = args[1] if len(args) > 1 else kwargs["timestep"]
            timestep = int(timestep_arg.flatten()[0].item()) if torch.is_tensor(timestep_arg) else int(timestep_arg)
            if step >= len(self.timesteps) or timestep != self.timesteps[step]:
                raise RuntimeError(f"Unexpected timestep {timestep} at step {step}")
            self.cur_timestep = step
            refresh = self.is_refresh_step()
            self.teacher_target = self.baseline_target = None
            self.mode = "teacher"
            teacher_result = original(*args, **kwargs)
            teacher_noise = tensor_output(teacher_result)
            if self.teacher_target is None:
                raise RuntimeError("Teacher target missing")
            if refresh:
                baseline_result = teacher_result
                baseline_noise = teacher_noise
                self.last_refresh_step = step
            else:
                self.mode = "baseline"
                baseline_result = original(*args, **kwargs)
                baseline_noise = tensor_output(baseline_result)
                if self.baseline_target is None:
                    raise RuntimeError("Baseline target missing")
                if step in self.probe_steps:
                    self.sufficient_stats[str(step)] = channel_stats(self.baseline_target, self.teacher_target)
            self.trace.append({
                "step": step, "timestep": timestep, "refresh": refresh,
                "age": 0 if refresh else step - self.last_refresh_step,
                "baseline_guided_noise_error": relative_l1(
                    guided_noise(teacher_noise, self.guidance_scale),
                    guided_noise(baseline_noise, self.guidance_scale)),
            })
            self.mode = "idle"
            return teacher_result
        self.pipe.unet.forward = forward


class PredictorEvaluationHelper(ScheduledHelper):
    def __init__(self, *args, parameters, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
        self.after_target = False
        self.probe_model = None
        self.target_hit = False

    def prediction(self, model: str, cached: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if model == "oracle":
            return teacher
        params = self.parameters[str(self.cur_timestep)]
        if model == "channel_mean":
            residual = params["mean"].to(device=cached.device, dtype=cached.dtype)[None, :, None, None]
        elif model == "channel_affine":
            slope = params["slope"].to(device=cached.device, dtype=cached.dtype)[None, :, None, None]
            intercept = params["intercept"].to(device=cached.device, dtype=cached.dtype)[None, :, None, None]
            residual = slope * cached + intercept
        else:
            raise ValueError(model)
        return cached + residual

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key = (blocktype, block_name, block_i, layer_i)
        original = block.forward
        self.function_dict[key] = original

        def forward(*args, **kwargs):
            if self.mode == "teacher":
                result = original(*args, **kwargs)
                if key == TARGET_KEY:
                    self.teacher_target = tensor_output(result)
                if self.is_refresh_step():
                    self.cached_output[key] = result
                return result
            if self.mode == "baseline":
                if self.is_skip_step(block_i, layer_i, blocktype):
                    result = self.cached_output[key]
                else:
                    result = original(*args, **kwargs)
                    self.cached_output[key] = result
                if key == TARGET_KEY:
                    self.baseline_target = tensor_output(result)
                return result
            if self.mode == "probe":
                if key == TARGET_KEY:
                    cached = tensor_output(self.cached_output[key])
                    corrected = self.prediction(self.probe_model, cached, self.teacher_target)
                    self.corrected_target = corrected
                    self.target_hit = True
                    self.after_target = True
                    return corrected
                if self.after_target:
                    return original(*args, **kwargs)
                if self.is_skip_step(block_i, layer_i, blocktype):
                    return self.cached_output[key]
                return original(*args, **kwargs)
            return original(*args, **kwargs)
        block.forward = forward

    def wrap_unet_forward(self):
        original = self.pipe.unet.forward
        self.function_dict["unet_forward"] = original

        def forward(*args, **kwargs):
            step = len(self.trace)
            timestep_arg = args[1] if len(args) > 1 else kwargs["timestep"]
            timestep = int(timestep_arg.flatten()[0].item()) if torch.is_tensor(timestep_arg) else int(timestep_arg)
            if step >= len(self.timesteps) or timestep != self.timesteps[step]:
                raise RuntimeError(f"Unexpected timestep {timestep} at step {step}")
            self.cur_timestep = step
            refresh = self.is_refresh_step()
            self.teacher_target = self.baseline_target = None
            self.mode = "teacher"
            teacher_result = original(*args, **kwargs)
            teacher_noise = tensor_output(teacher_result)
            if refresh:
                baseline_result = teacher_result
                baseline_noise = teacher_noise
                self.last_refresh_step = step
                cache_after = dict(self.cached_output)
            else:
                cache_before = dict(self.cached_output)
                self.mode = "baseline"
                baseline_result = original(*args, **kwargs)
                baseline_noise = tensor_output(baseline_result)
                cache_after = dict(self.cached_output)
            teacher_guided = guided_noise(teacher_noise, self.guidance_scale)
            baseline_error = relative_l1(teacher_guided, guided_noise(baseline_noise, self.guidance_scale))
            row = {"step": step, "timestep": timestep, "refresh": refresh,
                   "age": 0 if refresh else step - self.last_refresh_step,
                   "baseline_guided_noise_error": baseline_error, "models": {}}
            if not refresh and step in self.probe_steps:
                true_residual = self.teacher_target.float() - self.baseline_target.float()
                for model in MODELS:
                    self.cached_output = dict(cache_before)
                    self.mode = "probe"
                    self.probe_model = model
                    self.after_target = False
                    self.target_hit = False
                    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                    start.record(); probe_result = original(*args, **kwargs); end.record()
                    if not self.target_hit:
                        raise RuntimeError(f"Target not hit for {model}")
                    torch.cuda.synchronize()
                    probe_error = relative_l1(teacher_guided, guided_noise(tensor_output(probe_result), self.guidance_scale))
                    predicted_residual = self.corrected_target.float() - self.baseline_target.float()
                    row["models"][model] = {
                        "residual_relative_l1": relative_l1(true_residual, predicted_residual),
                        "corrected_feature_relative_l1": relative_l1(self.teacher_target, self.corrected_target),
                        "guided_noise_error": probe_error,
                        "recovery_fraction": ((baseline_error - probe_error) / baseline_error
                                              if baseline_error > 1e-12 else 0.0),
                        "milliseconds": start.elapsed_time(end),
                    }
            self.cached_output = cache_after
            self.mode = "idle"; self.probe_model = None; self.after_target = False
            self.trace.append(row)
            return teacher_result
        self.pipe.unet.forward = forward
