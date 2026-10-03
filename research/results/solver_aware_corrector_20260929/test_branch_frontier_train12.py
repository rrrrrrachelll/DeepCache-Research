from .run_branch_frontier_train12 import category_prompts


def test_formal_prompt_selection_is_one_unseen_train_prompt_per_category():
    prompts = category_prompts(1)
    assert len(prompts) == 12
    assert len({row["category"] for row in prompts}) == 12
    assert all(row["split"] == "train" for row in prompts)
    assert all(row["id"].endswith("_01") for row in prompts)
