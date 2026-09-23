"""Current-step spatial feature conditioned residual prediction helpers."""
from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F
from DeepCache.extension.deepcache import DeepCacheSDHelper

TARGET_KEY = ("up", "block", 1, 0)
SHALLOW_KEY = ("up", "attentions", 0, 0)
MODELS = ("cached_lowrank", "conv_delta", "shallow_delta", "combined", "combined_spatial_shuffle", "oracle")
BASE_MODELS = MODELS[:4]


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


def match_spatial(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] == reference.shape[-2:]:
        return value
    return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def feature_tensor(model: str, cached: torch.Tensor, conv_delta: torch.Tensor,
                   shallow_delta: torch.Tensor, spatial_shuffle: bool = False) -> torch.Tensor:
    conv_delta = match_spatial(conv_delta, cached)
    shallow_delta = match_spatial(shallow_delta, cached)
    if spatial_shuffle:
        shifts = (max(1, cached.shape[-2] // 2), max(1, cached.shape[-1] // 2))
        conv_delta = torch.roll(conv_delta, shifts=shifts, dims=(-2, -1))
        shallow_delta = torch.roll(shallow_delta, shifts=shifts, dims=(-2, -1))
    if model == "cached_lowrank":
        return cached
    if model == "conv_delta":
        return torch.cat([cached, conv_delta], dim=1)
    if model == "shallow_delta":
        return torch.cat([cached, shallow_delta], dim=1)
    if model in ("combined", "combined_spatial_shuffle"):
        return torch.cat([cached, conv_delta, shallow_delta], dim=1)
    raise ValueError(model)


def sample_spatial(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Return rows (batch * sampled spatial positions, channels)."""
    flat = value.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
    spatial = value.shape[-2] * value.shape[-1]
    rows = []
    for batch_index in range(value.shape[0]):
        rows.append(flat[batch_index * spatial + indices])
    return torch.cat(rows, dim=0)


def fit_lowrank(train_by_step: dict[str, dict[str, torch.Tensor]], rank: int = 16,
                ridge: float = 1e-3) -> dict:
    """Fit a shared supervised input projection and step-specific output heads."""
    prepared = {}
    cross = None
    for step, tensors in train_by_step.items():
        x = tensors["x"].double()
        y = tensors["y"].double()
        x_mean = x.mean(0)
        x_std = x.std(0, unbiased=False).clamp_min(1e-5)
        y_mean = y.mean(0)
        y_std = y.std(0, unbiased=False).clamp_min(1e-5)
        xs = (x - x_mean) / x_std
        ys = (y - y_mean) / y_std
        contribution = xs.T @ ys / xs.shape[0]
        cross = contribution if cross is None else cross + contribution
        prepared[step] = (xs, ys, x_mean, x_std, y_mean, y_std)
    cross = (cross / len(prepared)).float()
    q = min(rank, min(cross.shape))
    torch.manual_seed(0)
    projection, _, _ = torch.pca_lowrank(cross, q=q, center=False, niter=4)
    projection = projection[:, :q].double()
    result = {"projection": projection.float(), "steps": {}, "rank": q, "ridge": ridge}
    eye = torch.eye(q, dtype=torch.double)
    for step, (xs, ys, x_mean, x_std, y_mean, y_std) in prepared.items():
        z = xs @ projection
        head = torch.linalg.solve(z.T @ z + ridge * eye, z.T @ ys)
        result["steps"][step] = {
            "x_mean": x_mean.float(), "x_std": x_std.float(),
            "y_mean": y_mean.float(), "y_std": y_std.float(),
            "head": head.float(),
        }
    return result


def predict_lowrank(parameters: dict, step: int, features: torch.Tensor) -> torch.Tensor:
    params = parameters["steps"][str(step)]
    dtype, device = features.dtype, features.device
    x_mean = params["x_mean"].to(device=device, dtype=dtype)[None, :, None, None]
    x_std = params["x_std"].to(device=device, dtype=dtype)[None, :, None, None]
    projection = parameters["projection"].to(device=device, dtype=dtype)
    head = params["head"].to(device=device, dtype=dtype)
    y_mean = params["y_mean"].to(device=device, dtype=dtype)[None, :, None, None]
    y_std = params["y_std"].to(device=device, dtype=dtype)[None, :, None, None]
    standardized = (features - x_mean) / x_std
    z = torch.einsum("nchw,cr->nrhw", standardized, projection)
    y_standardized = torch.einsum("nrhw,ro->nohw", z, head)
    return y_standardized * y_std + y_mean


class ScheduledHelper(DeepCacheSDHelper):
    def __init__(self, pipe, refresh_steps, probe_steps, guidance_scale=7.5):
        super().__init__(pipe)
        self.set_params(cache_interval=1, cache_branch_id=0)
        self.refresh_steps = frozenset(refresh_steps)
        self.probe_steps = frozenset(probe_steps)
        self.guidance_scale = guidance_scale
        self.trace = []
        self.mode = "idle"
        self.last_refresh_step = 0
        self.teacher_target = self.baseline_target = None
        self.current_conv = self.anchor_conv = None
        self.teacher_shallow = self.baseline_shallow = self.anchor_shallow = None
        self.hook = None

    def enable(self):
        self.timesteps = [int(t) for t in self.pipe.scheduler.timesteps]
        if len(self.timesteps) != 20 or len(set(self.timesteps)) != 20:
            raise ValueError("Expected 20 unique scheduler timesteps")
        super().enable()
        def conv_hook(module, args, output):
            if self.mode == "teacher":
                self.current_conv = tensor_output(output).detach()
        self.hook = self.pipe.unet.conv_in.register_forward_hook(conv_hook)

    def disable(self):
        if self.hook is not None:
            self.hook.remove(); self.hook = None
        super().disable()

    def is_refresh_step(self):
        return self.cur_timestep in self.refresh_steps

    def is_skip_step(self, block_i, layer_i, blocktype="down"):
        if self.is_refresh_step():
            return False
        cb, cl = self.params["cache_block_id"], self.params["cache_layer_id"]
        if block_i > cb or blocktype == "mid": return True
        if block_i < cb: return False
        return layer_i >= cl if blocktype == "down" else layer_i > cl


class SpatialCollectorHelper(ScheduledHelper):
    def __init__(self, *args, samples_per_branch=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_per_branch = samples_per_branch
        self.samples = {}

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key = (blocktype, block_name, block_i, layer_i)
        original = block.forward
        self.function_dict[key] = original
        def forward(*args, **kwargs):
            if self.mode == "teacher":
                result = original(*args, **kwargs)
                if key == TARGET_KEY: self.teacher_target = tensor_output(result).detach()
                if key == SHALLOW_KEY: self.teacher_shallow = tensor_output(result).detach()
                if self.is_refresh_step(): self.cached_output[key] = result
                return result
            if self.mode == "baseline":
                if self.is_skip_step(block_i, layer_i, blocktype): result = self.cached_output[key]
                else:
                    result = original(*args, **kwargs); self.cached_output[key] = result
                if key == TARGET_KEY: self.baseline_target = tensor_output(result).detach()
                if key == SHALLOW_KEY: self.baseline_shallow = tensor_output(result).detach()
                return result
            return original(*args, **kwargs)
        block.forward = forward

    def wrap_unet_forward(self):
        original = self.pipe.unet.forward
        self.function_dict["unet_forward"] = original
        def forward(*args, **kwargs):
            step = len(self.trace)
            timestep_arg = args[1] if len(args) > 1 else kwargs["timestep"]
            timestep = int(timestep_arg.flatten()[0]) if torch.is_tensor(timestep_arg) else int(timestep_arg)
            if step >= len(self.timesteps) or timestep != self.timesteps[step]:
                raise RuntimeError(f"Unexpected timestep {timestep} at step {step}")
            self.cur_timestep = step; refresh = self.is_refresh_step()
            self.teacher_target = self.baseline_target = None
            self.teacher_shallow = self.baseline_shallow = None
            self.current_conv = None; self.mode = "teacher"
            teacher_result = original(*args, **kwargs)
            teacher_noise = tensor_output(teacher_result)
            if any(v is None for v in (self.teacher_target, self.teacher_shallow, self.current_conv)):
                raise RuntimeError("Teacher feature capture incomplete")
            if refresh:
                baseline_result = teacher_result; baseline_noise = teacher_noise
                self.anchor_conv = self.current_conv.detach().clone()
                self.anchor_shallow = self.teacher_shallow.detach().clone()
                self.last_refresh_step = step
            else:
                self.mode = "baseline"; baseline_result = original(*args, **kwargs)
                baseline_noise = tensor_output(baseline_result)
                if any(v is None for v in (self.baseline_target, self.baseline_shallow, self.anchor_conv, self.anchor_shallow)):
                    raise RuntimeError("Baseline or anchor feature capture incomplete")
                if step in self.probe_steps:
                    cached = self.baseline_target
                    conv_delta = match_spatial(self.current_conv - self.anchor_conv, cached)
                    shallow_delta = match_spatial(self.baseline_shallow - self.anchor_shallow, cached)
                    residual = self.teacher_target - cached
                    spatial = cached.shape[-2] * cached.shape[-1]
                    count = min(self.samples_per_branch, spatial)
                    indices = torch.linspace(0, spatial - 1, count, device=cached.device).round().long().unique()
                    self.samples[str(step)] = {
                        "cached": sample_spatial(cached, indices).half().cpu(),
                        "conv_delta": sample_spatial(conv_delta, indices).half().cpu(),
                        "shallow_delta": sample_spatial(shallow_delta, indices).half().cpu(),
                        "residual": sample_spatial(residual, indices).half().cpu(),
                        "shapes": {"cached": list(cached.shape), "conv": list(conv_delta.shape),
                                   "shallow": list(shallow_delta.shape)},
                    }
            self.trace.append({"step": step, "timestep": timestep, "refresh": refresh,
                "age": 0 if refresh else step - self.last_refresh_step,
                "baseline_guided_noise_error": relative_l1(guided_noise(teacher_noise), guided_noise(baseline_noise))})
            self.mode = "idle"; return teacher_result
        self.pipe.unet.forward = forward


class SpatialEvaluationHelper(ScheduledHelper):
    def __init__(self, *args, parameters, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
        self.after_target = False; self.probe_model = None; self.target_hit = False
        self.corrected_target = None

    def corrected(self, model: str) -> torch.Tensor:
        if model == "oracle": return self.teacher_target
        base_model = "combined" if model == "combined_spatial_shuffle" else model
        features = feature_tensor(base_model, self.baseline_target,
            self.current_conv - self.anchor_conv,
            self.baseline_shallow - self.anchor_shallow,
            spatial_shuffle=(model == "combined_spatial_shuffle"))
        residual = predict_lowrank(self.parameters[base_model], self.cur_timestep, features)
        return self.baseline_target + residual

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key = (blocktype, block_name, block_i, layer_i)
        original = block.forward; self.function_dict[key] = original
        def forward(*args, **kwargs):
            if self.mode == "teacher":
                result = original(*args, **kwargs)
                if key == TARGET_KEY: self.teacher_target = tensor_output(result).detach()
                if key == SHALLOW_KEY: self.teacher_shallow = tensor_output(result).detach()
                if self.is_refresh_step(): self.cached_output[key] = result
                return result
            if self.mode == "baseline":
                if self.is_skip_step(block_i, layer_i, blocktype): result = self.cached_output[key]
                else:
                    result = original(*args, **kwargs); self.cached_output[key] = result
                if key == TARGET_KEY: self.baseline_target = tensor_output(result).detach()
                if key == SHALLOW_KEY: self.baseline_shallow = tensor_output(result).detach()
                return result
            if self.mode == "probe":
                if key == TARGET_KEY:
                    self.corrected_target = self.corrected(self.probe_model)
                    self.target_hit = True; self.after_target = True
                    return self.corrected_target
                if self.after_target: return original(*args, **kwargs)
                if self.is_skip_step(block_i, layer_i, blocktype): return self.cached_output[key]
                return original(*args, **kwargs)
            return original(*args, **kwargs)
        block.forward = forward

    def wrap_unet_forward(self):
        original = self.pipe.unet.forward; self.function_dict["unet_forward"] = original
        def forward(*args, **kwargs):
            step = len(self.trace)
            timestep_arg = args[1] if len(args) > 1 else kwargs["timestep"]
            timestep = int(timestep_arg.flatten()[0]) if torch.is_tensor(timestep_arg) else int(timestep_arg)
            if step >= len(self.timesteps) or timestep != self.timesteps[step]:
                raise RuntimeError(f"Unexpected timestep {timestep} at step {step}")
            self.cur_timestep = step; refresh = self.is_refresh_step()
            self.teacher_target = self.baseline_target = None
            self.teacher_shallow = self.baseline_shallow = None
            self.current_conv = None; self.mode = "teacher"
            teacher_result = original(*args, **kwargs); teacher_noise = tensor_output(teacher_result)
            if refresh:
                baseline_result = teacher_result; baseline_noise = teacher_noise
                self.anchor_conv = self.current_conv.detach().clone()
                self.anchor_shallow = self.teacher_shallow.detach().clone()
                self.last_refresh_step = step; cache_after = dict(self.cached_output)
            else:
                cache_before = dict(self.cached_output); self.mode = "baseline"
                baseline_result = original(*args, **kwargs); baseline_noise = tensor_output(baseline_result)
                cache_after = dict(self.cached_output)
            teacher_guided = guided_noise(teacher_noise, self.guidance_scale)
            baseline_error = relative_l1(teacher_guided, guided_noise(baseline_noise, self.guidance_scale))
            row = {"step": step, "timestep": timestep, "refresh": refresh,
                   "age": 0 if refresh else step - self.last_refresh_step,
                   "baseline_guided_noise_error": baseline_error, "models": {}}
            if not refresh and step in self.probe_steps:
                true_residual = self.teacher_target.float() - self.baseline_target.float()
                for model in MODELS:
                    self.cached_output = dict(cache_before); self.mode = "probe"; self.probe_model = model
                    self.after_target = False; self.target_hit = False
                    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                    start.record(); probe_result = original(*args, **kwargs); end.record()
                    if not self.target_hit: raise RuntimeError(f"Target not hit for {model}")
                    torch.cuda.synchronize()
                    predicted_residual = self.corrected_target.float() - self.baseline_target.float()
                    probe_error = relative_l1(teacher_guided, guided_noise(tensor_output(probe_result), self.guidance_scale))
                    row["models"][model] = {
                        "residual_relative_l1": relative_l1(true_residual, predicted_residual),
                        "corrected_feature_relative_l1": relative_l1(self.teacher_target, self.corrected_target),
                        "guided_noise_error": probe_error,
                        "recovery_fraction": ((baseline_error - probe_error) / baseline_error if baseline_error > 1e-12 else 0.0),
                        "milliseconds": start.elapsed_time(end),
                    }
            self.cached_output = cache_after; self.mode = "idle"; self.probe_model = None; self.after_target = False
            self.trace.append(row); return teacher_result
        self.pipe.unet.forward = forward
