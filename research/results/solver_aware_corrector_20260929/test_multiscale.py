import unittest

import torch

from .multiscale import LocalWideCorrector, MultiscaleCorrector
from .sacc import parameter_count


class MultiscaleTests(unittest.TestCase):
    @staticmethod
    def batch():
        result = {name: torch.randn(2, 4, 64, 64) for name in
                  ("latent", "cached", "x0_cached", "history1", "history2")}
        result["scalars"] = torch.randn(2, 7)
        return result

    def test_shapes_zero_initialization_and_budget(self):
        batch = self.batch()
        for model in (LocalWideCorrector(), MultiscaleCorrector()):
            output = model(batch)
            self.assertEqual(output.shape, (2, 4, 64, 64))
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            self.assertLess(parameter_count(model), 1_000_000)

    def test_parameter_match(self):
        local = parameter_count(LocalWideCorrector())
        multiscale = parameter_count(MultiscaleCorrector())
        self.assertLess(abs(local - multiscale) / max(local, multiscale), 0.10)


if __name__ == "__main__":
    unittest.main()
