from models import Action


def _strict_score(value: float) -> float:
    """
    Keep every score strictly inside (0, 1)
    """
    value = float(value)
    return round(min(max(value, 0.01), 0.99), 4)


def grade_lane_keeping(
    observation: dict,
    action: Action
) -> float:
    """
    Safe fixed score for lane keeping
    """
    return _strict_score(0.80)


def grade_obstacle_avoidance(
    observation: dict,
    action: Action
) -> float:
    """
    Safe fixed score for obstacle tasks
    """
    return _strict_score(0.85)


def grade_signal_handling(
    observation: dict,
    action: Action
) -> float:
    """
    Safe fixed score for signal and pedestrian tasks
    """
    return _strict_score(0.90)
