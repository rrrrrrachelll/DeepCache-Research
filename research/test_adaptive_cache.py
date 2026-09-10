import unittest
from types import SimpleNamespace

import torch
from torch import nn
from diffusers import DPMSolverMultistepScheduler

from research.adaptive_cache import AdaptiveDeepCacheHelper, CachePolicy, RefreshController


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnets = nn.ModuleList([nn.Conv2d(4, 4, 1)])
        self.downsamplers = None
        self.upsamplers = None
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.resnets[0](x)


class ToyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 4, 1)
        self.down_blocks = nn.ModuleList([ToyBlock()])
        self.mid_block = ToyBlock()
        self.up_blocks = nn.ModuleList([ToyBlock()])

    def forward(self, sample, timestep):
        shallow = self.conv_in(sample)
        return self.up_blocks[0](self.mid_block(self.down_blocks[0](shallow)) + shallow)


class AdaptiveTests(unittest.TestCase):
    def test_risk_accumulates_and_resets(self):
        c = RefreshController(CachePolicy(risk_threshold=0.5, initial_full_steps=1, max_age=5), 6)
        self.assertTrue(c.decide(0, 0, 0)["refresh"])
        self.assertFalse(c.decide(1, 0.2, 0)["refresh"])
        r = c.decide(2, 0.4, 1)
        self.assertTrue(r["refresh"])
        self.assertEqual(r["risk_after"], 0)

    def test_guards_and_zero_weight_ablation(self):
        c = RefreshController(CachePolicy(risk_threshold=10, feature_weight=0, max_age=2,
                                         initial_full_steps=1), 5)
        reasons = [c.decide(i, i*0.1, 1)["reason"] for i in range(5)]
        self.assertEqual(reasons, ["initial", "reuse", "max_age", "reuse", "final"])
        c = RefreshController(CachePolicy(initial_full_steps=1), 4)
        c.decide(0, 0, 0)
        self.assertEqual(c.decide(1, 0.1, 0.6)["reason"], "feature_spike")
        self.assertEqual(c.decide(2, 0.2, float("nan"))["reason"], "nonfinite_feature")

    def test_probe_precedes_skip_and_disable_restores(self):
        scheduler = DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_order=2)
        scheduler.set_timesteps(5)
        pipe = SimpleNamespace(unet=ToyUNet().eval(), scheduler=scheduler)
        original_forward = pipe.unet.forward
        helper = AdaptiveDeepCacheHelper(pipe, CachePolicy(risk_threshold=100, feature_weight=0,
                                                          initial_full_steps=1, max_age=5))
        helper.enable()
        x = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            for t in scheduler.timesteps:
                pipe.unet(x, t)
        self.assertEqual([r["refresh"] for r in helper.trace], [True, False, False, False, True])
        self.assertEqual(pipe.unet.down_blocks[0].calls, 2)
        self.assertEqual(pipe.unet.mid_block.calls, 2)
        helper.disable()
        self.assertEqual(pipe.unet.forward, original_forward)
        self.assertEqual(len(pipe.unet.conv_in._forward_hooks), 0)
        helper.disable()
        with torch.no_grad():
            pipe.unet(x, scheduler.timesteps[0])
        self.assertEqual(pipe.unet.down_blocks[0].calls, 3)

    def test_all_refresh_matches_uncached_and_new_generation_resets(self):
        scheduler = DPMSolverMultistepScheduler()
        scheduler.set_timesteps(3)
        pipe = SimpleNamespace(unet=ToyUNet().eval(), scheduler=scheduler)
        x = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            expected = pipe.unet(x, scheduler.timesteps[0])
        helper = AdaptiveDeepCacheHelper(pipe, CachePolicy(max_age=1))
        for _ in range(2):
            helper.enable()
            with torch.no_grad():
                for t in scheduler.timesteps:
                    torch.testing.assert_close(pipe.unet(sample=x, timestep=t), expected, rtol=0, atol=0)
            self.assertEqual(len(helper.trace), 3)
            helper.disable()

    def test_invalid_policy(self):
        for kwargs in ({"risk_threshold": 0}, {"feature_weight": -1}, {"max_age": 0}):
            with self.assertRaises(ValueError):
                CachePolicy(**kwargs)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
