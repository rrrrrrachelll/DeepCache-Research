import unittest
from types import SimpleNamespace

import torch
from torch import nn
from diffusers import DPMSolverMultistepScheduler

from research.oracle_cache import OracleCacheHelper, relative_l1


class Block(nn.Module):
    def __init__(self, up=False):
        super().__init__()
        self.resnets = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(3 if up else 1)])
        self.attentions = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(3 if up else 0)])
        self.downsamplers = self.upsamplers = None

    def forward(self, x):
        for i, resnet in enumerate(self.resnets):
            x = resnet(x)
            if self.attentions:
                x = self.attentions[i](x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 4, 1)
        self.down_blocks = nn.ModuleList([Block()])
        self.mid_block = Block()
        self.up_blocks = nn.ModuleList([Block(up=True)])

    def forward(self, x, timestep):
        shallow = self.conv_in(x)
        return self.up_blocks[0](self.mid_block(self.down_blocks[0](shallow))) + shallow


class OracleTests(unittest.TestCase):
    def test_oracle_preserves_output_and_cache_anchor(self):
        torch.manual_seed(3)
        scheduler = DPMSolverMultistepScheduler()
        scheduler.set_timesteps(5)
        unet = UNet().eval()
        pipe = SimpleNamespace(unet=unet, scheduler=scheduler)
        inputs = [torch.randn(2, 4, 8, 8) for _ in range(5)]
        original = unet.forward
        with torch.no_grad():
            expected = [unet(x, t) for x, t in zip(inputs, scheduler.timesteps)]
            helper = OracleCacheHelper(pipe)
            helper.enable()
            for index, (x, t) in enumerate(zip(inputs, scheduler.timesteps)):
                actual = unet(x, t)
                torch.testing.assert_close(actual, expected[index], rtol=0, atol=0)
                if index % 3 == 0:
                    anchor = helper.cached_output[helper.boundary_key].clone()
                else:
                    torch.testing.assert_close(helper.cached_output[helper.boundary_key], anchor, rtol=0, atol=0)
            self.assertEqual([r["age"] for r in helper.trace], [0, 1, 2, 0, 1])
            self.assertGreater(helper.trace[1]["boundary_error"], 0)
            helper.disable()
            self.assertEqual(unet.forward, original)
            self.assertEqual(len(unet.conv_in._forward_hooks), 0)
            torch.testing.assert_close(unet(inputs[0], scheduler.timesteps[0]), expected[0])

    def test_relative_l1_denominator(self):
        self.assertEqual(relative_l1(torch.tensor([3.]), torch.tensor([2.])), .5)
        self.assertEqual(relative_l1(torch.zeros(2), torch.zeros(2)), 0)

    def test_bypass_does_not_read_or_write_cache(self):
        module = nn.Identity()
        helper = OracleCacheHelper(SimpleNamespace())
        helper.reset_states()
        helper.wrap_block_forward(module, "mid_block", 0, 0, "mid")
        helper.bypass = True
        self.assertEqual(module(torch.tensor(7.)).item(), 7)
        self.assertEqual(helper.cached_output, {})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
