import unittest

import torch

from .prompt_set import prompt_hash
from .sacc import parameter_count
from .train_text_conditioned import (PromptConditionedCorrector,
                                     attach_embeddings)


class TextConditionedTests(unittest.TestCase):
    def test_zero_initialization_shape_and_budget(self):
        batch = {name: torch.randn(2, 4, 8, 8) for name in
                 ("latent", "cached", "x0_cached", "history1", "history2")}
        batch["scalars"] = torch.randn(2, 7)
        batch["text_embedding"] = torch.randn(2, 768)
        model = PromptConditionedCorrector()
        output = model(batch)
        self.assertEqual(output.shape, (2, 4, 8, 8))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertLess(parameter_count(model), 1_000_000)

    def test_text_permutation_has_no_fixed_prompt(self):
        class Dataset:
            rows = [{"prompt_id": value} for value in ("a", "b", "c", "d")]
        artifact = {
            "prompt_hash": prompt_hash(),
            "embeddings": {value: torch.full((3,), index, dtype=torch.float32)
                           for index, value in enumerate(("a", "b", "c", "d"))},
        }
        dataset = Dataset()
        attach_embeddings(dataset, artifact)
        for row in dataset.rows:
            self.assertFalse(torch.equal(row["text_embedding"],
                                         row["text_embedding_shuffled"]))


if __name__ == "__main__":
    unittest.main()
