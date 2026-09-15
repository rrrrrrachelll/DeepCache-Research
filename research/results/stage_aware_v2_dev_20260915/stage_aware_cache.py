"""DeepCache helper with a fixed, auditable refresh schedule."""
from DeepCache import DeepCacheSDHelper


FIXED3_REFRESH = (0, 3, 6, 9, 12, 15, 18)
UNIFORM9_REFRESH = (0, 2, 5, 7, 10, 12, 14, 17, 19)
STAGE_AWARE9_REFRESH = (0, 4, 7, 10, 13, 14, 16, 18, 19)


def validate_refresh_indices(indices, total_steps=20):
    values = tuple(indices)
    if not values or values[0] != 0:
        raise ValueError("The first UNet call must refresh the empty cache")
    if values != tuple(sorted(set(values))):
        raise ValueError("Refresh indices must be unique and increasing")
    if values[-1] >= total_steps or values[0] < 0:
        raise ValueError("Refresh index outside the sampling schedule")
    return values


class ScheduledDeepCacheHelper(DeepCacheSDHelper):
    """Use the original branch-0 cache boundary with predetermined refresh calls."""

    def __init__(self, pipe, refresh_indices, total_steps=20):
        super().__init__(pipe)
        self.total_steps = total_steps
        self.refresh_indices = validate_refresh_indices(refresh_indices, total_steps)
        self.refresh_set = set(self.refresh_indices)
        self.active = False
        self.trace = []

    def enable(self, pipe=None):
        if self.active:
            raise RuntimeError("Helper is already enabled")
        if pipe is not None:
            raise ValueError("Pass the pipeline in the constructor")
        scheduler = self.pipe.scheduler
        if scheduler.num_inference_steps != self.total_steps:
            raise ValueError("Call scheduler.set_timesteps with the configured step count")
        self._timesteps = tuple(int(t) for t in scheduler.timesteps)
        if len(self._timesteps) != self.total_steps or len(set(self._timesteps)) != self.total_steps:
            raise ValueError("A unique, fixed-length timestep schedule is required")
        self.set_params(cache_interval=1, cache_branch_id=0)
        self.trace = []
        self._call_index = -1
        self._refresh = None
        super().enable()
        self.active = True

    def disable(self):
        if not self.active:
            return
        super().disable()
        self.active = False

    def wrap_unet_forward(self):
        self.function_dict["unet_forward"] = self.pipe.unet.forward

        def wrapped(*args, **kwargs):
            self._call_index += 1
            if self._call_index >= self.total_steps:
                raise RuntimeError("Helper reused beyond its sampling schedule")
            timestep = args[1] if len(args) > 1 else kwargs["timestep"]
            timestep = int(timestep.item() if hasattr(timestep, "item") else timestep)
            if timestep != self._timesteps[self._call_index]:
                raise RuntimeError("Scheduler schedule changed while helper was enabled")
            self.cur_timestep = self._call_index
            self._refresh = self._call_index in self.refresh_set
            self.trace.append(dict(index=self._call_index, timestep=timestep,
                                   refresh=self._refresh,
                                   reason="scheduled_refresh" if self._refresh else "scheduled_reuse"))
            return self.function_dict["unet_forward"](*args, **kwargs)

        self.pipe.unet.forward = wrapped

    def is_skip_step(self, block_i, layer_i, blocktype="down"):
        if self._refresh is None:
            raise RuntimeError("Cached module executed outside a scheduled UNet call")
        if self._refresh:
            return False
        if block_i > self.params["cache_block_id"] or blocktype == "mid":
            return True
        if block_i < self.params["cache_block_id"]:
            return False
        boundary = self.params["cache_layer_id"]
        return layer_i >= boundary if blocktype == "down" else layer_i > boundary
