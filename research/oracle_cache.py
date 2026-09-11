"""Offline paired probes on the uncached trajectory; never an online policy."""
import torch
import torch.nn.functional as F

from DeepCache import DeepCacheSDHelper


def tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    return output.sample


def relative_l1(current, reference):
    return ((current.float() - reference.float()).abs().mean()
            / reference.float().abs().mean().clamp_min(1e-6)).item()


class OracleCacheHelper(DeepCacheSDHelper):
    boundary_key = ("up", "attentions", 0, 1)
    mid_key = ("mid", "mid_block", 0, 0)

    def __init__(self, pipe, interval=3, guidance_scale=7.5):
        super().__init__(pipe)
        if interval < 1:
            raise ValueError("interval must be positive")
        self.set_params(cache_interval=interval, cache_branch_id=0)
        self.interval = interval
        self.guidance_scale = guidance_scale
        self.trace = []
        self.bypass = False
        self.capture = False
        self.features = {}
        self.previous = None
        self.anchor = None
        self.hooks = []

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        key = (blocktype, block_name, block_i, layer_i)
        self.function_dict[key] = block.forward

        def forward(*args, **kwargs):
            # Oracle recomputation must neither read nor overwrite the real cache.
            if self.bypass:
                return self.function_dict[key](*args, **kwargs)
            skip = self.is_skip_step(block_i, layer_i, blocktype)
            if skip:
                return self.cached_output[key]
            result = self.function_dict[key](*args, **kwargs)
            self.cached_output[key] = result
            return result

        block.forward = forward

    def enable(self):
        if type(self.pipe.scheduler).__name__ != "DPMSolverMultistepScheduler":
            raise ValueError("This experiment requires DPMSolverMultistepScheduler")
        self.timesteps = [int(t) for t in self.pipe.scheduler.timesteps]
        if len(set(self.timesteps)) != len(self.timesteps):
            raise ValueError("Unique timesteps required")
        self.lambdas = (self.pipe.scheduler.alpha_t.log()
                        - self.pipe.scheduler.sigma_t.log())[self.timesteps].tolist()
        self.trace, self.features = [], {}
        self.previous = self.anchor = None
        assert len(self.pipe.unet.up_blocks[-1].attentions) == 3
        super().enable()
        for name, module in [("conv", self.pipe.unet.conv_in),
                             ("boundary", self.pipe.unet.up_blocks[-1].attentions[-2]),
                             ("mid", self.pipe.unet.mid_block)]:
            def hook(module, args, output, name=name):
                if self.capture:
                    self.features[name] = tensor(output).detach().clone()
            self.hooks.append(module.register_forward_hook(hook))

    def wrap_unet_forward(self):
        self.function_dict["unet_forward"] = self.pipe.unet.forward

        def forward(*args, **kwargs):
            index = len(self.trace)
            assert int(args[1]) == self.timesteps[index]
            self.cur_timestep = index
            refresh = index % self.interval == 0
            self.features = {}
            self.capture, self.bypass = True, not refresh
            try:
                full = self.function_dict["unet_forward"](*args, **kwargs)
            finally:
                self.capture = self.bypass = False
            assert set(self.features) == {"conv", "boundary", "mid"}
            pooled = F.avg_pool2d(self.features["conv"], 4).float()
            change = 0.0 if self.previous is None else relative_l1(pooled, self.previous)
            if refresh:
                self.anchor = pooled.clone()
                self.anchor_index = index
            cached_change = relative_l1(pooled, self.anchor)
            cached = full if refresh else self.function_dict["unet_forward"](*args, **kwargs)
            true_noise, cached_noise = tensor(full), tensor(cached)
            assert true_noise.shape[0] == 2, "Single image with CFG required"
            def guided(noise):
                uncond, cond = noise.chunk(2)
                return uncond + self.guidance_scale * (cond - uncond)
            row = dict(index=index, timestep=self.timesteps[index], refresh=refresh,
                       age=index-self.anchor_index, lambda_value=self.lambdas[index],
                       delta_lambda=0.0 if index == 0 else abs(self.lambdas[index]-self.lambdas[index-1]),
                       lambda_distance=abs(self.lambdas[index]-self.lambdas[self.anchor_index]),
                       conv_previous=change, conv_cache=cached_change,
                       noise_error=relative_l1(cached_noise, true_noise),
                       guided_noise_error=relative_l1(guided(cached_noise), guided(true_noise)))
            for name, key in [("boundary", self.boundary_key), ("mid", self.mid_key)]:
                current, old = self.features[name], tensor(self.cached_output[key])
                row[name + "_error"] = relative_l1(old, current)
                for branch, position in [("uncond", 0), ("cond", 1)]:
                    row[name + "_" + branch + "_error"] = relative_l1(old[position], current[position])
            if refresh:
                assert row["boundary_error"] == row["mid_error"] == row["noise_error"] == 0
            self.trace.append(row)
            self.previous = pooled
            self.features = {}
            # Only the uncached prediction advances the scheduler.
            return full

        self.pipe.unet.forward = forward

    def disable(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        super().disable()
        self.features = {}
        self.previous = self.anchor = None
