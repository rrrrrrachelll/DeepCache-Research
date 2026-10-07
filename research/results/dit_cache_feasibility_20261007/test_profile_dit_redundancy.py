import unittest

import torch

from research.results.dit_cache_feasibility_20261007.profile_dit_redundancy import (
    relative_norm,
    tensor_similarity,
)


class SimilarityTest(unittest.TestCase):
    def test_identity(self):
        value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = tensor_similarity(value, value.clone())
        self.assertAlmostEqual(float(result["cosine"]), 1.0, places=6)
        self.assertEqual(float(result["relative_l2_change"]), 0.0)

    def test_known_scale_and_norm(self):
        current = torch.tensor([[[2.0, 0.0]]])
        previous = torch.tensor([[[1.0, 0.0]]])
        result = tensor_similarity(current, previous)
        self.assertAlmostEqual(float(result["cosine"]), 1.0, places=6)
        self.assertAlmostEqual(float(result["relative_l2_change"]), 0.5, places=6)
        self.assertAlmostEqual(float(relative_norm(previous, current)), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
