import unittest
from types import SimpleNamespace

import torch
from torch import nn
from diffusers import DPMSolverMultistepScheduler

from research.results.stage_aware_v2_dev_20260915.stage_aware_cache import (
    ScheduledDeepCacheHelper, STAGE_AWARE9_REFRESH, UNIFORM9_REFRESH,
    validate_refresh_indices,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnets = nn.ModuleList([nn.Conv2d(4, 4, 1)])
        self.downsamplers = self.upsamplers = None

    def forward(self, x):
        return self.resnets[0](x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([Block()])
        self.mid_block = Block()
        self.up_blocks = nn.ModuleList([Block()])

    def forward(self, sample, timestep):
        return self.up_blocks[0](self.mid_block(self.down_blocks[0](sample)))


class ScheduledCacheTests(unittest.TestCase):
    def test_equal_budget_and_stage_allocation(self):
        self.assertEqual(len(UNIFORM9_REFRESH), 9)
        self.assertEqual(len(STAGE_AWARE9_REFRESH), 9)
        self.assertEqual([sum(low <= i <= high for i in STAGE_AWARE9_REFRESH)
                          for low, high in ((0, 6), (7, 13), (14, 19))], [2, 3, 4])
        gaps = [b-a-1 for a, b in zip(STAGE_AWARE9_REFRESH, STAGE_AWARE9_REFRESH[1:])]
        self.assertLessEqual(max(gaps[:2]), 3)
        self.assertLessEqual(max(gaps[2:5]), 2)
        self.assertLessEqual(max(gaps[5:]), 1)

    def test_invalid_schedules(self):
        for values in ((), (1, 2), (0, 2, 2), (0, 20)):
            with self.assertRaises(ValueError):
                validate_refresh_indices(values)

    def test_helper_follows_exact_schedule_and_restores_unet(self):
        scheduler = DPMSolverMultistepScheduler()
        scheduler.set_timesteps(20)
        pipe = SimpleNamespace(unet=UNet().eval(), scheduler=scheduler)
        original = pipe.unet.forward
        helper = ScheduledDeepCacheHelper(pipe, STAGE_AWARE9_REFRESH)
        helper.enable()
        with torch.no_grad():
            for timestep in scheduler.timesteps:
                pipe.unet(torch.randn(1, 4, 8, 8), timestep)
        self.assertEqual(tuple(r["index"] for r in helper.trace if r["refresh"]),
                         STAGE_AWARE9_REFRESH)
        helper.disable()
        self.assertEqual(pipe.unet.forward, original)


if __name__ == "__main__":
    unittest.main()
