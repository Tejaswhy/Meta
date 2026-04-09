from grader import (
    grade_lane_keeping,
    grade_obstacle_avoidance,
    grade_signal_handling,
)
from models import Observation, Reward, Action


class AutoPilotEnv:
    def __init__(self):
        self.tasks = [
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
                "step": 1,
                "front_distance": 5.0,
                "left_lane_clear": True,
                "right_lane_clear": False,
            },
            {
                "task_name": "signal_safety",
                "step": 2,
                "traffic_light": "red",
                "pedestrian_crossing": True,
                "pedestrian_distance": 4.0,
            },
            {
                "task_name": "emergency_braking",
                "step": 3,
                "front_distance": 2.0,
                "left_lane_clear": False,
                "right_lane_clear": False,
            },
            {
                "task_name": "pedestrian_priority",
                "step": 4,
                "traffic_light": "green",
                "pedestrian_crossing": True,
                "pedestrian_distance": 3.0,
            },
        ]

        self.task_index = 0
        self.current_task = self.tasks[0]

    def reset(self):
        self.task_index = 0
        self.current_task = self.tasks[0]
        return Observation(**self.current_task)

    def step(self, action: Action):
        task_name = self.current_task["task_name"]

        if task_name == "lane_keeping":
            grade = grade_lane_keeping(self.current_task, action)

        elif task_name in {
            "obstacle_avoidance",
            "emergency_braking",
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
            score=min(max(grade, 0.01), 0.99),
            reason="task_progress",
        )

        done = False

        if self.task_index < len(self.tasks) - 1:
            self.task_index += 1
            self.current_task = self.tasks[self.task_index]
        else:
            done = True

        return (
            Observation(**self.current_task),
            reward,
            done,
            {"task_name": task_name},
        )

    def close(self):
        pass
