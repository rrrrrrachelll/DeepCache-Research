import unittest
from types import SimpleNamespace

from .data import PromptGroupedBatchSampler


class DataTests(unittest.TestCase):
    def test_prompt_grouped_sampler_has_one_step_per_case(self):
        dataset = SimpleNamespace(rows=[{"case": case} for case in ("a", "b", "c") for _ in range(4)])
        sampler = PromptGroupedBatchSampler(dataset, batch_size=3, seed=2)
        for batch in sampler:
            cases = [dataset.rows[index]["case"] for index in batch]
            self.assertEqual(len(cases), len(set(cases)))


if __name__ == "__main__":
    unittest.main()
