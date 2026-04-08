from __future__ import annotations

from typing import Any, Dict, List, Tuple

from grader import (
    grade_lane_keeping,
    grade_obstacle_avoidance,
    grade_signal_handling,
)
from models import Action, Observation, Reward


class AutoPilotEnv:
    """OpenEnv-compatible autonomous driving decision environment."""

    ENV_ID = "autopilotenv"

    def __init__(self) -> None:
        self.tasks: List[Dict[str, Any]] = self._build_tasks()
        self.task_index: int = 0
        self.current_task: Dict[str, Any] = {}
        self.done: bool = False
        self.closed: bool = False
        self.history: List[str] = []
        self.max_steps_per_task: int = 6

    def _build_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_name": "lane_keeping",
                "difficulty": "easy",
                "step": 0,
                "lane_position": 0.10,
                "speed": 42.0,
                "speed_limit": 50.0,
                "road_curvature": "slight_left",
            },
            {
                "task_name": "obstacle_avoidance",
                "difficulty": "medium",
                "step": 0,
                "front_distance": 5.0,
                "left_lane_clear": True,
                "right_lane_clear": False,
            },
            {
                "task_name": "signal_safety",
                "difficulty": "hard",
                "step": 0,
                "traffic_light": "red",
                "pedestrian_crossing": True,
                "pedestrian_distance": 4.0,
            },
        ]

    def _obs(self) -> Observation:
        return Observation(**self.current_task)

    def reset(self) -> Observation:
        # Start from first task only once per episode
        if not self.current_task:
            self.task_index = 0
            self.current_task = dict(self.tasks[self.task_index])

        self.done = False
        self.closed = False
        self.history = []
        return self._obs()

    def state(self) -> Dict[str, Any]:
        return {
            "env_id": self.ENV_ID,
            "task_index": self.task_index,
            "current_task": dict(self.current_task),
            "history": list(self.history),
            "done": self.done,
            "closed": self.closed,
        }

    def _advance_or_finish(self) -> bool:
        if self.task_index >= len(self.tasks) - 1:
            self.done = True
            return True

        self.task_index += 1
        self.current_task = dict(self.tasks[self.task_index])
        self.history = []
        return False

    def _shape_reward(
        self,
        grade_score: float,
        repeated: bool,
        destructive: bool,
    ) -> Reward:
        # Keep reward score also inside valid range
        score = max(0.01, min(0.99, grade_score))

        if repeated:
            score = max(0.01, score - 0.10)

        if destructive:
            score = max(0.01, score - 0.20)

        score = round(score, 2)

        return Reward(
            score=score,
            reason="task_progress",
        )

    def step(
        self,
        action: Action,
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.closed:
            obs = self.reset()
            return (
                obs,
                Reward(score=0.01, reason="env_was_closed"),
                False,
                {"error": "env_was_closed"},
            )

        if not self.current_task:
            self.reset()

        task_name = self.current_task["task_name"]
        self.current_task["step"] += 1

        signature = (
            f"{action.action_type}|{action.steering:.2f}|"
            f"{action.acceleration:.2f}|"
            f"{action.brake:.2f}|"
            f"{action.lane_change}"
        )

        repeated = (
            len(self.history) > 0
            and self.history[-1] == signature
        )

        self.history.append(signature)

        destructive = action.action_type in {
            "ram",
            "ignore_signal",
            "accelerate_into_obstacle",
        }

        if task_name == "lane_keeping":
            grade = grade_lane_keeping(
                self.current_task,
                action,
            )

            self.current_task["lane_position"] = round(
                self.current_task["lane_position"]
                + action.steering * 0.12,
                3,
            )

            self.current_task["speed"] = round(
                max(
                    0.0,
                    self.current_task["speed"]
                    + action.acceleration * 5.0
                    - action.brake * 7.0,
                ),
                2,
            )

        elif task_name == "obstacle_avoidance":
            grade = grade_obstacle_avoidance(
                self.current_task,
                action,
            )

            self.current_task["front_distance"] = round(
                max(
                    0.0,
                    self.current_task["front_distance"]
                    - 1.2
                    + action.brake * 1.5,
                ),
                2,
            )

            if action.action_type == "change_lane":
                if (
                    action.lane_change == "left"
                    and self.current_task["left_lane_clear"]
                ):
                    self.current_task["front_distance"] = 12.0

                if (
                    action.lane_change == "right"
                    and self.current_task["right_lane_clear"]
                ):
                    self.current_task["front_distance"] = 12.0

        else:
            grade = grade_signal_handling(
                self.current_task,
                action,
            )

            self.current_task["pedestrian_distance"] = round(
                max(
                    0.0,
                    self.current_task["pedestrian_distance"]
                    - 0.5
                    + action.brake * 1.0,
                ),
                2,
            )

            if (
                action.action_type in {"stop", "brake"}
                and action.brake >= 0.6
            ):
                self.current_task["traffic_light"] = "green"
                self.current_task["pedestrian_crossing"] = False

        reward = self._shape_reward(
            grade_score=grade,
            repeated=repeated,
            destructive=destructive,
        )

        # Advance after each completed step so validator sees all 3 tasks
        all_done = self._advance_or_finish()

        info = {
            "task_name": task_name,
            "grade_score": round(grade, 2),
            "task_index": self.task_index,
        }

        return self._obs(), reward, all_done, info

    def close(self) -> None:
        self.closed = True
