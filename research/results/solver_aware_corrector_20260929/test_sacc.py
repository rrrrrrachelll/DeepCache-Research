import unittest
from types import SimpleNamespace

import torch
from torch import nn
from diffusers import DPMSolverMultistepScheduler

from .prompt_set import build_prompts, prompt_hash
from .sacc import (REFRESH_BALANCED8, Round0CollectorHelper, SolverAwareCorrector,
                   guided_noise, parameter_count, scheduler_history, x0_from_epsilon)


class Block(nn.Module):
    def __init__(self, up=False):
        super().__init__()
        count = 3 if up else 1
        self.resnets = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(count)])
        self.attentions = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(count if up else 0)])
        self.downsamplers = self.upsamplers = None

    def forward(self, value):
        for index, resnet in enumerate(self.resnets):
            value = resnet(value)
            if self.attentions:
                value = self.attentions[index](value)
        return value


class ToyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 4, 1)
        self.down_blocks = nn.ModuleList([Block()])
        self.mid_block = Block()
        self.up_blocks = nn.ModuleList([Block(up=True)])

    def forward(self, value, timestep):
        shallow = self.conv_in(value)
        value = self.down_blocks[0](shallow)
        value = self.mid_block(value)
        return self.up_blocks[0](value) + shallow


class SACCTests(unittest.TestCase):
    def test_prompt_split_is_frozen_and_disjoint(self):
        rows = build_prompts()
        self.assertEqual(len(rows), 180)
        self.assertEqual(len(prompt_hash()), 64)
        split_ids = {split: {r["id"] for r in rows if r["split"] == split}
                     for split in ("train", "validation", "development")}
        self.assertEqual([len(split_ids[s]) for s in split_ids], [120, 30, 30])
        self.assertFalse(split_ids["train"] & split_ids["validation"])
        self.assertFalse(split_ids["train"] & split_ids["development"])
        self.assertFalse(split_ids["validation"] & split_ids["development"])

    def test_corrector_shapes_zero_initialization_and_budget(self):
        batch = {name: torch.randn(2, 4, 8, 8) for name in
                 ("latent", "cached", "x0_cached", "history1", "history2")}
        batch["scalars"] = torch.randn(2, 7)
        for variant in ("c0", "c1", "c2", "c3"):
            model = SolverAwareCorrector(variant)
            output = model(batch)
            self.assertEqual(output.shape, (2, 4, 8, 8))
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            self.assertLess(parameter_count(model), 1_000_000)

    def test_guidance_is_applied_once(self):
        noise = torch.cat([torch.ones(1, 4, 2, 2), torch.ones(1, 4, 2, 2) * 3])
        self.assertTrue(torch.equal(guided_noise(noise, 2), torch.ones(1, 4, 2, 2) * 5))

    def test_history_order_and_x0_conversion(self):
        scheduler = DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_order=2)
        scheduler.set_timesteps(20)
        reference = torch.randn(1, 4, 8, 8)
        first, second = torch.randn_like(reference), torch.randn_like(reference)
        scheduler.model_outputs = [first, second]
        newest, older, valid = scheduler_history(scheduler, reference)
        torch.testing.assert_close(newest, second)
        torch.testing.assert_close(older, first)
        self.assertEqual(valid, (1, 1))
        epsilon = torch.randn_like(reference)
        actual = x0_from_epsilon(scheduler, 0, reference, epsilon)
        alpha, sigma = scheduler._sigma_to_alpha_sigma_t(scheduler.sigmas[0])
        torch.testing.assert_close(actual, (reference - sigma * epsilon) / alpha)

    def test_shadow_does_not_advance_scheduler_and_reuse_is_nonzero(self):
        torch.manual_seed(4)
        scheduler = DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_order=2)
        scheduler.set_timesteps(20)
        unet = ToyUNet().eval()
        pipe = SimpleNamespace(unet=unet, scheduler=scheduler)
        original = unet.forward
        helper = Round0CollectorHelper(pipe, REFRESH_BALANCED8)
        inputs = [torch.randn(2, 4, 8, 8) for _ in range(20)]
        helper.enable()
        with torch.no_grad():
            for index, (value, timestep) in enumerate(zip(inputs, scheduler.timesteps)):
                result = unet(value, timestep)
                guided = guided_noise(result)
                scheduler.step(guided, timestep, value[:1])
        helper.disable()
        self.assertEqual(unet.forward, original)
        self.assertEqual(len(helper.trace), 20)
        self.assertEqual(sum(row["refresh"] for row in helper.trace), 8)
        self.assertEqual(len(helper.samples), 12)
        self.assertTrue(all(row["guided_error"] == 0 for row in helper.trace if row["refresh"]))
        self.assertTrue(any(row["guided_error"] > 0 for row in helper.trace if not row["refresh"]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
