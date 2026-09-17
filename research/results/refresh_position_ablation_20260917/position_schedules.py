"""Predeclared one-refresh dropout schedules around uniform9."""
from research.results.stage_aware_v2_dev_20260915.stage_aware_cache import (
    UNIFORM9_REFRESH, validate_refresh_indices,
)

DROPPED_REFRESH = {
    "drop_early8": 2,
    "drop_middle8": 10,
    "drop_late8": 17,
}
SCHEDULES = {
    name: validate_refresh_indices(i for i in UNIFORM9_REFRESH if i != dropped)
    for name, dropped in DROPPED_REFRESH.items()
}


def validate_ablation():
    uniform = set(UNIFORM9_REFRESH)
    for name, schedule in SCHEDULES.items():
        if len(schedule) != 8 or uniform - set(schedule) != {DROPPED_REFRESH[name]}:
            raise ValueError(f"{name} is not a one-refresh dropout")
        left = max(i for i in schedule if i < DROPPED_REFRESH[name])
        right = min(i for i in schedule if i > DROPPED_REFRESH[name])
        if right - left != 5:
            raise ValueError(f"{name} does not create the predeclared five-step span")
    return True


validate_ablation()
