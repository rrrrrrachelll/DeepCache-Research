"""Collection and model components for solver-aware cache correction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from DeepCache.extension.deepcache import DeepCacheSDHelper


REFRESH_BALANCED8 = (0, 2, 5, 7, 10, 13, 16, 19)


def tensor_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    sample = getattr(value, "sample", None)
    if torch.is_tensor(sample):
        return sample
    if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"Cannot extract tensor from {type(value)!r}")


def guided_noise(noise: torch.Tensor, scale: float = 7.5) -> torch.Tensor:
    if noise.shape[0] != 2:
        raise ValueError("Week-one protocol requires one CFG image (batch size two)")
    unconditional, conditional = noise.chunk(2)
    return unconditional + scale * (conditional - unconditional)


def relative_l1(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.mean(torch.abs(estimate.float() - reference.float()))
    denominator = torch.mean(torch.abs(reference.float())).clamp_min(1e-12)
    return (numerator / denominator).item()


def scheduler_history(scheduler, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Return newest and second-newest converted DPM outputs without mutation."""
    values = [value for value in scheduler.model_outputs if value is not None]
    newest = values[-1].detach() if values else torch.zeros_like(reference)
    second = values[-2].detach() if len(values) > 1 else torch.zeros_like(reference)
    return newest, second, (int(bool(values)), int(len(values) > 1))


def x0_from_epsilon(scheduler, step: int, sample: torch.Tensor,
                    epsilon: torch.Tensor) -> torch.Tensor:
    sigma = scheduler.sigmas[step].to(device=sample.device, dtype=sample.dtype)
    alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma)
    return (sample - sigma_t * epsilon) / alpha_t


class Round0CollectorHelper(DeepCacheSDHelper):
    """Collect full/cache pairs while advancing the real cached trajectory."""

    def __init__(self, pipe, refresh_steps=REFRESH_BALANCED8, guidance_scale=7.5):
        super().__init__(pipe)
        self.set_params(cache_interval=1, cache_branch_id=0)
        self.refresh_steps = tuple(refresh_steps)
        self.refresh_set = frozenset(refresh_steps)
        self.guidance_scale = guidance_scale
        self.mode = "idle"
        self.samples = []
        self.trace = []
        self.last_refresh_step = 0

    def enable(self):
        self.timesteps = tuple(int(t) for t in self.pipe.scheduler.timesteps)
        if len(self.timesteps) != 20 or len(set(self.timesteps)) != 20:
            raise ValueError("Expected a fixed schedule of 20 unique timesteps")
        if not self.refresh_steps or self.refresh_steps[0] != 0:
            raise ValueError("The first call must refresh an empty cache")
        super().enable()

    def is_refresh(self):
        return self.cur_timestep in self.refresh_set

    def is_skip_step(self, block_i, layer_i, blocktype="down"):
        if self.is_refresh():
            return False
        cache_block = self.params["cache_block_id"]
        cache_layer = self.params["cache_layer_id"]
        if block_i > cache_block or blocktype == "mid":
            return True
        if block_i < cache_block:
            return False
        return layer_i >= cache_layer if blocktype == "down" else layer_i > cache_layer

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key = (blocktype, block_name, block_i, layer_i)
        original = block.forward
        self.function_dict[key] = original

        def forward(*args, **kwargs):
            if self.mode == "full":
                result = original(*args, **kwargs)
                if self.is_refresh():
                    self.cached_output[key] = result
                return result
            if self.mode == "cache":
                if self.is_skip_step(block_i, layer_i, blocktype):
                    return self.cached_output[key]
                result = original(*args, **kwargs)
                self.cached_output[key] = result
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
            if step >= 20 or timestep != self.timesteps[step]:
                raise RuntimeError(f"Unexpected timestep {timestep} at call {step}")
            self.cur_timestep = step
            refresh = self.is_refresh()
            model_input = args[0] if args else kwargs["sample"]
            latent = model_input[:1].detach()
            history1, history2, history_valid = scheduler_history(self.pipe.scheduler, latent)
            cache_identity_before = {key: id(value) for key, value in self.cached_output.items()}
            scheduler_state_before = (self.pipe.scheduler._step_index,
                tuple(id(value) if value is not None else None for value in self.pipe.scheduler.model_outputs))

            self.mode = "full"
            full_result = original(*args, **kwargs)
            full = guided_noise(tensor_output(full_result), self.guidance_scale)
            if refresh:
                cached_result = full_result
                cached = full
                self.last_refresh_step = step
            else:
                if cache_identity_before != {key: id(value) for key, value in self.cached_output.items()}:
                    raise RuntimeError("Shadow full call mutated the DeepCache state")
                if scheduler_state_before != (self.pipe.scheduler._step_index,
                    tuple(id(value) if value is not None else None for value in self.pipe.scheduler.model_outputs)):
                    raise RuntimeError("Shadow full call mutated the scheduler state")
                self.mode = "cache"
                cached_result = original(*args, **kwargs)
                cached = guided_noise(tensor_output(cached_result), self.guidance_scale)

            sigma = self.pipe.scheduler.sigmas[step].float().item()
            lambda_value = float(-torch.log(self.pipe.scheduler.sigmas[step].float().clamp_min(1e-12)))
            anchor_lambda = float(-torch.log(self.pipe.scheduler.sigmas[self.last_refresh_step].float().clamp_min(1e-12)))
            row = {
                "step": step, "timestep": timestep, "refresh": refresh,
                "age": step - self.last_refresh_step,
                "sigma": sigma, "lambda": lambda_value,
                "lambda_distance": abs(lambda_value - anchor_lambda),
                "history1_valid": history_valid[0], "history2_valid": history_valid[1],
                "guided_error": relative_l1(full, cached),
            }
            if refresh and row["guided_error"] != 0:
                raise RuntimeError("Refresh step must have zero cache error")
            if not refresh:
                self.samples.append({
                    "step": step,
                    "latent": latent.half().cpu(),
                    "cached": cached.detach().half().cpu(),
                    "full": full.detach().half().cpu(),
                    "target": (full - cached).detach().half().cpu(),
                    "x0_cached": x0_from_epsilon(self.pipe.scheduler, step, latent, cached).detach().half().cpu(),
                    "history1": history1.detach().half().cpu(),
                    "history2": history2.detach().half().cpu(),
                    "scalars": torch.tensor([
                        timestep / 1000.0, sigma, lambda_value,
                        float(row["age"]), row["lambda_distance"],
                        float(history_valid[0]), float(history_valid[1]),
                    ], dtype=torch.float32),
                })
            self.trace.append(row)
            self.mode = "idle"
            return cached_result

        self.pipe.unet.forward = forward


VARIANT_CHANNELS = {"c0": 12, "c1": 12, "c2": 16, "c3": 20}


class FiLMResidualBlock(nn.Module):
    def __init__(self, width: int, scalar_width: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.film = nn.Linear(scalar_width, width * 2)
        self.activation = nn.SiLU()

    def forward(self, value, condition):
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm1(value)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(self.activation(hidden))
        hidden = self.conv2(self.activation(self.norm2(hidden)))
        return value + hidden


class SolverAwareCorrector(nn.Module):
    def __init__(self, variant="c3", width=32, blocks=4, scalar_features=7):
        super().__init__()
        if variant not in VARIANT_CHANNELS:
            raise ValueError(f"Unknown variant {variant}")
        self.variant = variant
        self.scalar_features = scalar_features
        scalar_width = 64
        self.scalar_mlp = nn.Sequential(nn.Linear(scalar_features, scalar_width), nn.SiLU(),
                                        nn.Linear(scalar_width, scalar_width))
        self.input = nn.Conv2d(VARIANT_CHANNELS[variant], width, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResidualBlock(width, scalar_width) for _ in range(blocks)])
        self.output = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(self, batch):
        values = [batch["latent"], batch["cached"], batch["x0_cached"]]
        if self.variant in ("c2", "c3"):
            values.append(batch["history1"])
        if self.variant == "c3":
            values.append(batch["history2"])
        return torch.cat(values, dim=1)

    def conditioned_scalars(self, scalars):
        result = scalars.clone()
        if self.variant == "c0":
            result[:, 3:] = 0
        elif self.variant == "c1":
            result[:, 5:] = 0
        elif self.variant == "c2":
            result[:, 6:] = 0
        return result

    def forward(self, batch):
        condition = self.scalar_mlp(self.conditioned_scalars(batch["scalars"]))
        hidden = self.input(self.features(batch))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
