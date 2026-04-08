from typing import Any, Dict, List, Tuple

from grader import (
    grade_lane_keeping,
    grade_obstacle_avoidance,
    grade_signal_handling,
)
from models import Action, Observation, Reward


class AutoPilotEnv:
    ENV_ID = "autopilotenv"

    def __init__(self):
        self.tasks = self._build_tasks()
        self.task_index = 0
        self.current_task = {}
        self.done = False
        self.closed = False

    def _build_tasks(self):
        return [
            {
                "task_name": "lane_keeping",
                "step": 0,
                "lane_position": 0.1,
                "speed": 42.0,
                "speed_limit": 50.0,
                "road_curvature": "slight_left",
            },
            {
                "task_name": "obstacle_avoidance",
                "step": 0,
                "front_distance": 5.0,
                "left_lane_clear": True,
                "right_lane_clear": False,
            },
            {
                "task_name": "signal_safety",
                "step": 0,
                "traffic_light": "red",
                "pedestrian_crossing": True,
                "pedestrian_distance": 4.0,
            },
            {
                "task_name": "emergency_braking",
                "step": 0,
                "front_distance": 2.0,
                "left_lane_clear": False,
                "right_lane_clear": False,
            },
            {
                "task_name": "pedestrian_priority",
                "step": 0,
                "traffic_light": "green",
                "pedestrian_crossing": True,
                "pedestrian_distance": 3.0,
            },
        ]

    def _obs(self):
        return Observation(**self.current_task)

    def reset(self):
        self.task_index = 0
        self.current_task = dict(self.tasks[0])
        self.done = False
        return self._obs()

    def _strict_reward(self, value):
        return round(min(max(value, 0.01), 0.99), 4)

    def step(
        self,
        action: Action
    ) -> Tuple[Observation, Reward, bool, Dict]:
        task_name = self.current_task["task_name"]

        if task_name == "lane_keeping":
            grade = grade_lane_keeping(
                self.current_task,
                action
            )
        elif task_name in {
            "obstacle_avoidance",
            "emergency_braking"
        }:
            grade = grade_obstacle_avoidance(
                self.current_task,
                action
            )
        else:
            grade = grade_signal_handling(
                self.current_task,
                action
            )

        reward = Reward(
            score=self._strict_reward(grade),
            reason="task_progress"
        )

        if self.task_index < len(self.tasks) - 1:
            self.task_index += 1
            self.current_task = dict(
                self.tasks[self.task_index]
            )
            done = False
        else:
            done = True

        return self._obs(), reward, done, {
            "task_name": task_name
        }

    def close(self):
        self.closed = True
