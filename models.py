from typing import Literal, Optional
from pydantic import BaseModel, Field

TaskName = Literal[
    "lane_keeping",
    "obstacle_avoidance",
    "signal_safety",
    "emergency_braking",
    "pedestrian_priority"
]

TrafficLight = Literal["red", "yellow", "green"]
RoadCurvature = Literal[
    "straight",
    "slight_left",
    "slight_right",
    "sharp_left",
    "sharp_right"
]


class Observation(BaseModel):
    task_name: TaskName
    step: int = Field(ge=0)

    lane_position: Optional[float] = None
    speed: Optional[float] = None
    speed_limit: Optional[float] = None
    road_curvature: Optional[RoadCurvature] = None

    front_distance: Optional[float] = None
    left_lane_clear: Optional[bool] = None
    right_lane_clear: Optional[bool] = None

    traffic_light: Optional[TrafficLight] = None
    pedestrian_crossing: Optional[bool] = None
    pedestrian_distance: Optional[float] = None


class Action(BaseModel):
    steering: float = 0.0
    acceleration: float = 0.0
    brake: float = 0.0
    action_type: Optional[str] = None
    lane_change: Optional[
        Literal["left", "right", "none"]
    ] = "none"


class Reward(BaseModel):
    score: float = Field(gt=0.0, lt=1.0)
    reason: str
