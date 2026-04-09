import random
from models import Observation, Reward


class AutoPilotEnv:
    def __init__(self):
        self.task_names = [
            "lane_keeping",
            "obstacle_avoidance",
            "signal_safety",
            "emergency_braking",
            "pedestrian_priority",
        ]

        self.task_index = 0
        self.max_tasks = len(self.task_names)

        self.speed = 0.0
        self.front_distance = 10.0
        self.lane_position = 0.0

        self.current_state = None

    def _strict_score(self, value: float) -> float:
        """
        Keep score strictly inside (0,1)
        Validator does NOT allow 0.0 or 1.0
        """
        return round(min(max(value, 0.01), 0.99), 2)

    def _generate_task_state(self, task_name):
        """
        Generate task-specific state
        """

        if task_name == "lane_keeping":
            self.speed = round(random.uniform(35, 55), 1)
            self.lane_position = round(random.uniform(-0.4, 0.4), 2)

            return {
                "task_name": task_name,
                "lane_position": self.lane_position,
                "speed": self.speed,
                "speed_limit": 50.0,
                "road_curvature": random.choice(
                    ["straight", "slight_left", "slight_right"]
                ),
            }

        elif task_name == "obstacle_avoidance":
            self.front_distance = round(random.uniform(2, 8), 1)

            return {
                "task_name": task_name,
                "front_distance": self.front_distance,
                "left_lane_clear": random.choice([True, False]),
                "right_lane_clear": random.choice([True, False]),
                "speed": self.speed,
            }

        elif task_name == "signal_safety":
            return {
                "task_name": task_name,
                "traffic_light": random.choice(
                    ["red", "yellow", "green"]
                ),
                "pedestrian_crossing": random.choice(
                    [True, False]
                ),
                "pedestrian_distance": round(
                    random.uniform(2, 6), 1
                ),
                "speed": self.speed,
            }

        elif task_name == "emergency_braking":
            self.front_distance = round(random.uniform(1, 3), 1)

            return {
                "task_name": task_name,
                "front_distance": self.front_distance,
                "left_lane_clear": False,
                "right_lane_clear": False,
                "speed": self.speed,
            }

        else:  # pedestrian_priority
            return {
                "task_name": task_name,
                "traffic_light": "green",
                "pedestrian_crossing": True,
                "pedestrian_distance": round(
                    random.uniform(1, 4), 1
                ),
                "speed": self.speed,
            }

    def reset(self):
        self.task_index = 0
        task_name = self.task_names[self.task_index]

        self.current_state = self._generate_task_state(task_name)

        return Observation(**self.current_state)

    def step(self, action):
        task_name = self.task_names[self.task_index]

        # -------------------------
        # Dynamic vehicle updates
        # -------------------------
        self.speed += action.acceleration * 5
        self.speed -= action.brake * 8
        self.speed = max(0.0, self.speed)

        if "front_distance" in self.current_state:
            self.front_distance = self.current_state["front_distance"]
            self.front_distance -= self.speed * 0.1
            self.front_distance = max(0.5, self.front_distance)
            self.current_state["front_distance"] = round(
                self.front_distance, 1
            )

        if "lane_position" in self.current_state:
            self.lane_position = self.current_state["lane_position"]
            self.lane_position += action.steering * 0.1
            self.lane_position = round(
                max(-1.0, min(1.0, self.lane_position)), 2
            )
            self.current_state["lane_position"] = (
                self.lane_position
            )

        self.current_state["speed"] = round(
            self.speed, 1
        )

        # -------------------------
        # Reward logic
        # -------------------------
        score = 0.5
        reason = "safe driving"

        if task_name == "lane_keeping":
            lane_pos = abs(
                self.current_state["lane_position"]
            )
            score = 0.95 - lane_pos

        elif task_name == "obstacle_avoidance":
            dist = self.current_state["front_distance"]

            if dist < 2:
                score = (
                    0.95
                    if action.brake >= 0.7
                    else 0.15
                )
            else:
                score = 0.8

        elif task_name == "signal_safety":
            light = self.current_state["traffic_light"]

            if light == "red":
                score = (
                    0.95
                    if action.brake >= 0.6
                    else 0.2
                )
            elif light == "yellow":
                score = (
                    0.85
                    if action.brake >= 0.4
                    else 0.5
                )
            else:
                score = 0.85

        elif task_name == "emergency_braking":
            score = (
                0.98
                if action.brake >= 0.8
                else 0.1
            )

        elif task_name == "pedestrian_priority":
            score = (
                0.96
                if action.brake >= 0.7
                else 0.15
            )

        score = self._strict_score(score)

        reward = Reward(
            score=score,
            reason=reason
        )

        # -------------------------
        # Next task
        # -------------------------
        self.task_index += 1
        done = self.task_index >= self.max_tasks

        if not done:
            next_task = self.task_names[self.task_index]
            self.current_state = self._generate_task_state(
                next_task
            )

        observation = Observation(
            **self.current_state
        )

        info = {
            "task_name": task_name
        }

        return observation, reward, done, info

    def state(self):
        return self.current_state

    def close(self):
        pass
