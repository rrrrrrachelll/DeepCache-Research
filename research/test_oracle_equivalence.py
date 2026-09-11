import unittest
from types import SimpleNamespace

import torch
from diffusers import DPMSolverMultistepScheduler
from DeepCache import DeepCacheSDHelper
from research.oracle_cache import OracleCacheHelper, relative_l1
from research.test_oracle_cache import UNet


class EquivalenceTests(unittest.TestCase):
    def test_shadow_matches_original_deepcache(self):
        torch.manual_seed(4)
        scheduler = DPMSolverMultistepScheduler()
        scheduler.set_timesteps(5)
        pipe = SimpleNamespace(unet=UNet().eval(), scheduler=scheduler)
        inputs = [torch.randn(2, 4, 8, 8) for _ in range(5)]
        with torch.no_grad():
            full = [pipe.unet(x, t) for x, t in zip(inputs, scheduler.timesteps)]
            original = DeepCacheSDHelper(pipe)
            original.set_params(cache_interval=3, cache_branch_id=0)
            original.enable()
            shadow = [pipe.unet(x, t) for x, t in zip(inputs, scheduler.timesteps)]
            original.disable()
            helper = OracleCacheHelper(pipe)
            helper.enable()
            for index, (x, t) in enumerate(zip(inputs, scheduler.timesteps)):
                pipe.unet(x, t)
                self.assertEqual(helper.trace[-1]["noise_error"], relative_l1(shadow[index], full[index]))
            helper.disable()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
