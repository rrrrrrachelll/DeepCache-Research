"""Teacher-forced matched encoder-decoder recomputation sweep for branch-0 DeepCache."""
from __future__ import annotations
from typing import Any
import torch
from DeepCache.extension.deepcache import DeepCacheSDHelper

FRONTIERS = {
    "decoder_only": ("mid", "mid_block", 0, 0),
    "down3_pair": ("down", "block", 3, 0),
    "down2_pair": ("down", "block", 2, 0),
    "down1_pair": ("down", "block", 1, 0),
    "full_unet": ("down", "block", 0, 0),
}


def tensor_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value): return value
    sample=getattr(value,"sample",None)
    if torch.is_tensor(sample): return sample
    if isinstance(value,tuple) and value and torch.is_tensor(value[0]): return value[0]
    raise TypeError(f"Cannot extract tensor from {type(value)!r}")


def relative_l1(reference,estimate):
    return (torch.mean(torch.abs(estimate.float()-reference.float())) /
            torch.mean(torch.abs(reference.float())).clamp_min(1e-12)).item()


def guided_noise(noise,scale=7.5):
    uncond,cond=noise.chunk(2); return uncond+scale*(cond-uncond)


class MatchedPathOracle(DeepCacheSDHelper):
    def __init__(self,pipe,refresh_steps,probe_steps,guidance_scale=7.5):
        super().__init__(pipe); self.set_params(cache_interval=1,cache_branch_id=0)
        self.refresh_steps=frozenset(refresh_steps); self.probe_steps=frozenset(probe_steps)
        self.guidance_scale=guidance_scale; self.trace=[]; self.mode='idle'
        self.frontier_key=None; self.frontier_started=False; self.frontier_hit=False
        self.recomputed_keys=set(); self.last_refresh_step=0

    def enable(self):
        self.timesteps=[int(t) for t in self.pipe.scheduler.timesteps]
        if len(self.timesteps)!=20 or len(set(self.timesteps))!=20: raise ValueError('Expected 20 unique timesteps')
        super().enable()

    def is_refresh_step(self): return self.cur_timestep in self.refresh_steps

    def is_skip_step(self,block_i,layer_i,blocktype='down'):
        if self.is_refresh_step(): return False
        cb,cl=self.params['cache_block_id'],self.params['cache_layer_id']
        if block_i>cb or blocktype=='mid': return True
        if block_i<cb: return False
        return layer_i>=cl if blocktype=='down' else layer_i>cl

    @staticmethod
    def starts_frontier(key,frontier_key,already_started=False):
        return already_started or key==frontier_key

    def wrap_block_forward(self,block,block_name,block_i,layer_i,blocktype='down'):
        key=(blocktype,block_name,block_i,layer_i); original=block.forward; self.function_dict[key]=original
        def forward(*args,**kwargs):
            if self.mode=='teacher':
                result=original(*args,**kwargs)
                if self.is_refresh_step(): self.cached_output[key]=result
                return result
            if self.mode=='baseline':
                if self.is_skip_step(block_i,layer_i,blocktype): return self.cached_output[key]
                result=original(*args,**kwargs); self.cached_output[key]=result; return result
            if self.mode=='probe':
                if key==self.frontier_key and not self.frontier_started:
                    self.frontier_started=True; self.frontier_hit=True
                if self.frontier_started:
                    self.recomputed_keys.add(key); return original(*args,**kwargs)
                if self.is_skip_step(block_i,layer_i,blocktype): return self.cached_output[key]
                return original(*args,**kwargs)
            return original(*args,**kwargs)
        block.forward=forward

    def wrap_unet_forward(self):
        original=self.pipe.unet.forward; self.function_dict['unet_forward']=original
        def forward(*args,**kwargs):
            step=len(self.trace); timestep_arg=args[1] if len(args)>1 else kwargs['timestep']
            timestep=int(timestep_arg.flatten()[0]) if torch.is_tensor(timestep_arg) else int(timestep_arg)
            if step>=len(self.timesteps) or timestep!=self.timesteps[step]:
                raise RuntimeError(f'Unexpected timestep {timestep} at step {step}')
            self.cur_timestep=step; refresh=self.is_refresh_step(); self.mode='teacher'
            teacher_result=original(*args,**kwargs); teacher_noise=tensor_output(teacher_result)
            baseline_ms=0.0
            if refresh:
                baseline_result=teacher_result; baseline_noise=teacher_noise
                self.last_refresh_step=step; cache_after=dict(self.cached_output)
            else:
                cache_before=dict(self.cached_output); self.mode='baseline'
                start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                start.record(); baseline_result=original(*args,**kwargs); end.record(); torch.cuda.synchronize()
                baseline_ms=start.elapsed_time(end); baseline_noise=tensor_output(baseline_result)
                cache_after=dict(self.cached_output)
            teacher_guided=guided_noise(teacher_noise,self.guidance_scale)
            baseline_error=relative_l1(teacher_guided,guided_noise(baseline_noise,self.guidance_scale))
            row={'step':step,'timestep':timestep,'refresh':refresh,
                 'age':0 if refresh else step-self.last_refresh_step,
                 'baseline_guided_noise_error':baseline_error,'baseline_milliseconds':baseline_ms,'frontiers':{}}
            if not refresh and step in self.probe_steps:
                for name,key in FRONTIERS.items():
                    self.cached_output=dict(cache_before); self.mode='probe'; self.frontier_key=key
                    self.frontier_started=False; self.frontier_hit=False; self.recomputed_keys=set()
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    start.record(); probe_result=original(*args,**kwargs); end.record(); torch.cuda.synchronize()
                    if not self.frontier_hit: raise RuntimeError(f'Frontier not reached: {name} {key}')
                    probe_error=relative_l1(teacher_guided,guided_noise(tensor_output(probe_result),self.guidance_scale))
                    recovery=(baseline_error-probe_error)/baseline_error if baseline_error>1e-12 else 0.0
                    probe_ms=start.elapsed_time(end)
                    row['frontiers'][name]={'guided_noise_error':probe_error,
                        'error_reduction':baseline_error-probe_error,'recovery_fraction':recovery,
                        'milliseconds':probe_ms,'incremental_milliseconds':probe_ms-baseline_ms,
                        'recomputed_wrapped_modules':len(self.recomputed_keys)}
                    if name=='full_unet' and probe_error>1e-6:
                        raise RuntimeError(f'Full-UNet control mismatch at step {step}: {probe_error}')
            self.cached_output=cache_after; self.mode='idle'; self.frontier_key=None; self.frontier_started=False
            self.trace.append(row); return teacher_result
        self.pipe.unet.forward=forward
