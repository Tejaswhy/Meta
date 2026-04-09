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
        self.current_task: Dict[str, Any] = {}
        self.done = False

    def _build_tasks(self) -> List[Dict[str, Any]]:
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

    def reset(self) -> Observation:
        self.task_index = 0
        self.current_task = dict(self.tasks[0])
        self.done = False
        return Observation(**self.current_task)

    def state(self) -> Dict[str, Any]:
        return {
            "env_id": self.ENV_ID,
            "task_index": self.task_index,
            "current_task": self.current_task,
            "done": self.done,
        }

    def _safe_reward(self, score: float) -> Reward:
        """
        Keep reward strictly inside (0,1)
        """
        safe = max(0.01, min(0.99, float(score)))
        safe = round(safe, 4)

        return Reward(
            score=safe,
            reason="task_progress"
        )

    def step(
        self,
        action: Action,
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if not self.current_task:
            self.reset()

        # Save current task before moving ahead
        current = dict(self.current_task)
        task_name = current["task_name"]

        # Grade based on CURRENT task
        if task_name == "lane_keeping":
            grade = grade_lane_keeping(current, action)

        elif task_name in {
            "obstacle_avoidance",
            "emergency_braking",
        }:
            grade = grade_obstacle_avoidance(current, action)

        else:
            grade = grade_signal_handling(current, action)

        reward = self._safe_reward(grade)

        # Final task check
        done = self.task_index == len(self.tasks) - 1

        info = {
            "task_name": task_name,
            "task_index": self.task_index,
            "reward_score": reward.score,
        }

        # Return CURRENT observation
        response_obs = Observation(**current)

        # Advance for next API call
        if not done:
            self.task_index += 1
            self.current_task = dict(
                self.tasks[self.task_index]
            )
        else:
            self.done = True

        return response_obs, reward, done, info

    def close(self):
        self.done = True
