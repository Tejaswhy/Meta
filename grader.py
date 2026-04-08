from models import Action


def _strict_score(value: float) -> float:
    """
    Keep score strictly inside (0, 1).
    Validator does NOT allow 0.0 or 1.0
    """
    return round(min(max(value, 0.01), 0.99), 4)


def grade_lane_keeping(observation: dict, action: Action) -> float:
    score = 0.0

    lane_position = float(observation.get("lane_position", 1.0))
    speed = float(observation.get("speed", 0.0))
    speed_limit = float(observation.get("speed_limit", 50.0))

    if abs(lane_position) <= 0.15:
        score += 0.5
    elif abs(lane_position) <= 0.30:
        score += 0.3
    else:
        score += 0.1

    if speed <= speed_limit:
        score += 0.3
    elif speed - speed_limit <= 5:
        score += 0.1

    if abs(action.steering) <= 0.30:
        score += 0.2
    elif abs(action.steering) <= 0.60:
        score += 0.1

    return _strict_score(score)


def grade_obstacle_avoidance(observation: dict, action: Action) -> float:
    front_distance = float(observation.get("front_distance", 100.0))
    left_lane_clear = bool(observation.get("left_lane_clear", False))
    right_lane_clear = bool(observation.get("right_lane_clear", False))

    score = 0.5

    if front_distance <= 3.0:
        if (
            action.action_type in {"emergency_stop", "brake"}
            and action.brake >= 0.8
        ):
            score = 0.95
        else:
            score = 0.05

    elif front_distance <= 7.0:
        if action.action_type == "change_lane":
            if action.lane_change == "left" and left_lane_clear:
                score = 0.92
            elif action.lane_change == "right" and right_lane_clear:
                score = 0.92
            else:
                score = 0.2

        elif action.action_type == "brake" and action.brake >= 0.5:
            score = 0.8

        else:
            score = 0.1

    else:
        if action.action_type in {"proceed", "maintain"}:
            score = 0.9
        else:
            score = 0.7

    return _strict_score(score)


def grade_signal_handling(observation: dict, action: Action) -> float:
    light = observation.get("traffic_light")
    crossing = bool(observation.get("pedestrian_crossing", False))
    pedestrian_distance = float(
        observation.get("pedestrian_distance", 999.0)
    )

    must_stop = light == "red" or (
        crossing and pedestrian_distance <= 8.0
    )

    soft_caution = light == "yellow"

    score = 0.5

    if must_stop:
        if (
            action.action_type in {"stop", "brake"}
            and action.brake >= 0.6
        ):
            score = 0.95
        else:
            score = 0.05

    elif soft_caution:
        if action.action_type in {"brake", "proceed"}:
            score = 0.8
        else:
            score = 0.4

    else:
        if (
            action.action_type in {"proceed", "maintain"}
            and action.brake < 0.3
        ):
            score = 0.92
        else:
            score = 0.6

    return _strict_score(score)
