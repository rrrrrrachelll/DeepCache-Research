from .run_single_step_intervention import POLICIES, StageBranchHelper
from .sacc import REFRESH_BALANCED8


def test_all_interventions_target_reuse_steps():
    refresh = set(REFRESH_BALANCED8)
    assert len(POLICIES) == 11
    assert all(not (set(policy) & refresh) for policy in POLICIES.values())


def test_policy_set_contains_early_control_and_late_candidates():
    assert POLICIES["b8_s01"] == {1: 8}
    assert POLICIES["b8_s18"] == {18: 8}
    assert POLICIES["b10_s18"] == {18: 10}
