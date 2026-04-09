from models import Action


def _strict_score(value: float) -> float:
    value = float(value)
    return round(min(max(value, 0.01), 0.99), 4)


def grade_lane_keeping(observation: dict, action: Action) -> float:
    score = 0.2

    lane_position = float(observation.get("lane_position", 1.0))

    if abs(lane_position) <= 0.15:
        score = 0.95
    elif abs(lane_position) <= 0.30:
        score = 0.75
    else:
        score = 0.30

    return _strict_score(score)


def grade_obstacle_avoidance(observation: dict, action: Action) -> float:
    front_distance = float(
        observation.get("front_distance", 100.0)
    )

    if front_distance <= 3.0:
        score = 0.95 if action.brake >= 0.8 else 0.20
    elif front_distance <= 7.0:
        score = 0.90
    else:
        score = 0.70

    return _strict_score(score)


def grade_signal_handling(observation: dict, action: Action) -> float:
    light = observation.get("traffic_light")

    if light == "red":
        score = 0.95 if action.brake >= 0.6 else 0.20
    else:
        score = 0.85

    return _strict_score(score)
