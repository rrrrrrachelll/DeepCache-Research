"""Strict eight-call schedules for the balanced refresh comparison."""

DROP_MIDDLE8 = (0, 2, 5, 7, 12, 14, 17, 19)
SCHEDULES = {
    "balanced8_a": (0, 2, 5, 7, 10, 13, 16, 19),
    "balanced8_b": (0, 2, 5, 8, 11, 14, 17, 19),
}


def gaps(schedule):
    return tuple(right-left for left, right in zip(schedule, schedule[1:]))


def validate_schedules():
    assert len(DROP_MIDDLE8) == 8
    assert DROP_MIDDLE8[:3] == (0, 2, 5)
    assert max(gaps(DROP_MIDDLE8)) == 5
    for schedule in SCHEDULES.values():
        assert len(schedule) == len(set(schedule)) == 8
        assert schedule == tuple(sorted(schedule))
        assert schedule[:3] == (0, 2, 5)
        assert schedule[-1] == 19
        assert max(gaps(schedule)) == 3
    return True


validate_schedules()
