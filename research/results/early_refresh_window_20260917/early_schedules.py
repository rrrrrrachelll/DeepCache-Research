"""Equal-budget schedules for the early refresh-window sweep."""

from research.results.stage_aware_v2_dev_20260915.stage_aware_cache import UNIFORM9_REFRESH

SECOND_REFRESH_STEPS = (1, 2, 3, 4)
REFERENCE_MODE = "early2"
SCHEDULES = {
    f"early{step}": (0, step, 5, 7, 10, 12, 14, 17, 19)
    for step in SECOND_REFRESH_STEPS
}
MEASURED_SCHEDULES = {name: schedule for name, schedule in SCHEDULES.items()
                      if name != REFERENCE_MODE}


def validate_schedules():
    assert SCHEDULES[REFERENCE_MODE] == tuple(UNIFORM9_REFRESH)
    fixed = (5, 7, 10, 12, 14, 17, 19)
    for step in SECOND_REFRESH_STEPS:
        schedule = SCHEDULES[f"early{step}"]
        assert len(schedule) == 9
        assert schedule[:2] == (0, step)
        assert schedule[2:] == fixed
        assert len(set(schedule)) == len(schedule)
    return True


validate_schedules()
