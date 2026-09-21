import unittest
import torch

from .block_replacement_oracle import tree_error


class TreeErrorTests(unittest.TestCase):
    def test_identical_nested_tree_is_zero(self):
        value = (torch.tensor([1.0, -2.0]), (torch.tensor([3.0]),))
        result = tree_error(value, value)
        self.assertAlmostEqual(result["relative_l1"], 0.0)
        self.assertAlmostEqual(result["cosine_distance"], 0.0)
        self.assertEqual(result["numel"], 3)

    def test_relative_l1_is_aggregated_by_tensor_values(self):
        reference = (torch.ones(2), torch.ones(2) * 2)
        estimate = (torch.ones(2) * 2, torch.ones(2) * 4)
        result = tree_error(reference, estimate)
        self.assertAlmostEqual(result["relative_l1"], 1.0)

    def test_shape_mismatch_returns_none(self):
        self.assertIsNone(tree_error(torch.ones(2), torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
