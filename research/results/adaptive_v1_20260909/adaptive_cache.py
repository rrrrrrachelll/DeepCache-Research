"""Experimental SD1.5/DPM-Solver++ cache controller; original helper is unchanged."""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler

from DeepCache import DeepCacheSDHelper


@dataclass(frozen=True)
class CachePolicy:
    risk_threshold: float = 0.65
    feature_weight: float = 2.0
    feature_spike: float = 0.5
    max_age: int = 3
    initial_full_steps: int = 2
    final_full_steps: int = 1

    def __post_init__(self):
        for name in ("risk_threshold", "feature_spike"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.feature_weight) or self.feature_weight < 0:
            raise ValueError("feature_weight must be finite and nonnegative")
        for name in ("max_age", "initial_full_steps", "final_full_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class RefreshController:
    """One decision per UNet call, before any cached module runs."""
    def __init__(self, policy, total_steps):
        self.policy = policy
        self.total_steps = total_steps
        self.risk = 0.0
        self.last_refresh = None
        self.previous_lambda = None
        self.next_index = 0

    def decide(self, index, log_snr_half, feature_change):
        if index != self.next_index or index >= self.total_steps:
            raise RuntimeError("Unexpected step order; create a new helper for each generation")
        self.next_index += 1
        if not math.isfinite(log_snr_half):
            raise ValueError("Nonfinite log-SNR")
        delta = 0.0 if self.previous_lambda is None else abs(log_snr_half - self.previous_lambda)
        self.previous_lambda = log_snr_half
        finite_feature = math.isfinite(feature_change)
        score = self.risk + delta * (1 + self.policy.feature_weight * feature_change) if finite_feature else None
        age = 0 if self.last_refresh is None else index - self.last_refresh
        if index < self.policy.initial_full_steps:
            reason = "initial"
        elif index >= self.total_steps - self.policy.final_full_steps:
            reason = "final"
        elif not finite_feature:
            reason = "nonfinite_feature"
        elif self.policy.feature_weight > 0 and feature_change >= self.policy.feature_spike:
            reason = "feature_spike"
        elif age >= self.policy.max_age:
            reason = "max_age"
        elif score >= self.policy.risk_threshold:
            reason = "risk"
        else:
            reason = "reuse"
        refresh = reason != "reuse"
        self.risk = 0.0 if refresh else score
        if refresh:
            self.last_refresh = index
        return dict(index=index, log_snr_half=log_snr_half, delta_lambda=delta,
                    feature_change=feature_change if finite_feature else None,
                    age_before=age, risk_before_decision=score, risk_after=self.risk,
                    refresh=refresh, reason=reason)


class AdaptiveDeepCacheHelper(DeepCacheSDHelper):
    """SD1.5 branch 0 only. Enable after scheduler.set_timesteps, once per image."""
    def __init__(self, pipe, policy=None):
        super().__init__(pipe)
        self.policy = policy or CachePolicy()
        self.active = False
        self._probe_handle = None
        self.trace = []

    def enable(self, pipe=None):
        if self.active:
            raise RuntimeError("Helper is already enabled")
        if pipe is not None:
            raise ValueError("Pass the pipeline in the constructor")
        scheduler = self.pipe.scheduler
        if not isinstance(scheduler, DPMSolverMultistepScheduler):
            raise ValueError("V1 supports DPMSolverMultistepScheduler only")
        if scheduler.config.algorithm_type != "dpmsolver++" or scheduler.config.solver_order != 2:
            raise ValueError("V1 requires second-order dpmsolver++")
        if scheduler.num_inference_steps is None:
            raise ValueError("Call scheduler.set_timesteps before enabling")
        if not hasattr(self.pipe.unet, "conv_in"):
            raise ValueError("V1 requires a UNet with conv_in")
        self._timesteps = scheduler.timesteps.detach().cpu().tolist()
        if len(set(self._timesteps)) != len(self._timesteps):
            raise ValueError("Repeated timesteps are not supported by this experimental controller")
        ids = torch.tensor(self._timesteps, dtype=torch.long)
        lambdas = (scheduler.alpha_t.log() - scheduler.sigma_t.log()).detach().cpu()
        self._lambdas = lambdas[ids].tolist()
        self.controller = RefreshController(self.policy, len(self._timesteps))
        self.trace = []
        self._previous_feature = None
        self._call_index = -1
        self._refresh = None
        self.set_params(cache_interval=1, cache_branch_id=0)
        super().enable()
        self._probe_handle = self.pipe.unet.conv_in.register_forward_hook(self._observe_feature)
        self.active = True

    def disable(self):
        if not self.active:
            return
        self._probe_handle.remove()
        self._probe_handle = None
        super().disable()
        self._previous_feature = None
        self.active = False

    def wrap_unet_forward(self):
        self.function_dict["unet_forward"] = self.pipe.unet.forward

        def wrapped(*args, **kwargs):
            self._call_index += 1
            self._refresh = None
            t = args[1] if len(args) > 1 else kwargs["timestep"]
            t = t.item() if torch.is_tensor(t) else t
            if self._call_index >= len(self._timesteps) or t != self._timesteps[self._call_index]:
                raise RuntimeError("Scheduler schedule changed or helper reused across generations")
            result = self.function_dict["unet_forward"](*args, **kwargs)
            if self._refresh is None:
                raise RuntimeError("conv_in probe was not executed")
            return result

        self.pipe.unet.forward = wrapped

    def _observe_feature(self, module, inputs, output):
        if self._refresh is not None:
            raise RuntimeError("conv_in executed more than once per UNet call")
        # Observe fresh, pre-cache features. Pooling bounds proxy storage/reduction cost.
        feature = F.avg_pool2d(output.detach(), kernel_size=4).float()
        change = 0.0
        if self._previous_feature is not None:
            change = ((feature - self._previous_feature).abs().mean() /
                      self._previous_feature.abs().mean().clamp_min(1e-6)).item()
        self._previous_feature = feature
        row = self.controller.decide(self._call_index, self._lambdas[self._call_index], change)
        row["timestep"] = self._timesteps[self._call_index]
        self._refresh = row["refresh"]
        self.trace.append(row)

    def is_skip_step(self, block_i, layer_i, blocktype="down"):
        if self._refresh is None:
            raise RuntimeError("Cached module executed before fresh feature probe")
        if self._refresh:
            return False
        # Original branch-0 spatial skip boundary, with a different temporal decision.
        if block_i > self.params["cache_block_id"] or blocktype == "mid":
            return True
        if block_i < self.params["cache_block_id"]:
            return False
        boundary = self.params["cache_layer_id"]
        return layer_i >= boundary if blocktype == "down" else layer_i > boundary

    def configuration(self):
        return asdict(self.policy)
