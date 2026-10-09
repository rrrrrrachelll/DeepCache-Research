from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).parent))
from run_experiment14b import BlockResidualReuse, build_policies


class ToyBlock(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment
        self.real_calls = 0

    def forward(self, hidden_states, *args, **kwargs):
        self.real_calls += 1
        return hidden_states + self.increment


class ToyTransformer:
    def __init__(self):
        self.transformer_blocks = torch.nn.ModuleList([ToyBlock(1.0), ToyBlock(2.0)])


def test_residual_reuse_skips_real_block_computation():
    transformer = ToyTransformer()
    selected = {(1, 0), (2, 1)}
    controller = BlockResidualReuse(transformer, selected)
    with controller:
        for _ in range(3):
            value = torch.zeros(1)
            for block in transformer.transformer_blocks:
                value = block(value)
    controller.validate(expected_steps=3)
    assert controller.skipped == 2
    assert controller.computed == 4
    assert [block.real_calls for block in transformer.transformer_blocks] == [2, 2]
    assert value.item() == 3.0


def test_policy_controls_match_each_budget():
    ranked = [(step, block) for step in range(1, 5) for block in range(3)]
    policies, metadata = build_policies(ranked, {"small": 0.25, "large": 0.5}, seed=5)
    assert len(policies["small"]) == len(policies["small_shuffled"]) == 3
    assert len(policies["large"]) == len(policies["large_shuffled"]) == 6
    assert metadata["large"]["actual_fraction"] == 0.5
