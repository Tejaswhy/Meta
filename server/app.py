from __future__ import annotations

import json
import io
import contextlib
from typing import Dict

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from inference import main
from env import AutoPilotEnv as _AutoPilotEnv
from models import Action


class AutoPilotEnv(_AutoPilotEnv):
    """Entrypoint class for OpenEnv validation."""


# Global environment instance for API routes
_env = AutoPilotEnv()

# FastAPI app instance
api = FastAPI()


def render_car_html(
    lane_position: float,
    steering: float,
    speed: float,
) -> str:
    offset_x = lane_position * 120
    tilt = steering * 18
    offset_y = -(speed * 2)

    return f"""
    <div style="
        position:relative;
        width:620px;
        height:320px;
        background:#f3f4f6;
        border-radius:20px;
        overflow:hidden;
        border:1px solid #ddd;
        margin:auto;
    ">
        <div style="
            position:absolute;
            left:50%;
            top:0;
            width:4px;
            height:100%;
            background:#bbb;
        "></div>

        <div style="
            position:absolute;
            left:25%;
            top:0;
            width:2px;
            height:100%;
            background:#ddd;
        "></div>

        <div style="
            position:absolute;
            right:25%;
            top:0;
            width:2px;
            height:100%;
            background:#ddd;
        "></div>

        <div style="
            position:absolute;
            left:50%;
            top:50%;
            transform:translate({offset_x}px, {offset_y}px) rotate({tilt}deg);
            transition:all 0.15s ease;
        ">
            <div style="
                width:120px;
                height:60px;
                background:black;
                border-radius:16px;
                box-shadow:0 10px 20px rgba(0,0,0,0.3);
                position:relative;
            ">
                <div style="
                    position:absolute;
                    top:8px;
                    left:20px;
                    width:80px;
                    height:20px;
                    background:#444;
                    border-radius:8px;
                "></div>

                <div style="
                    position:absolute;
                    width:18px;
                    height:18px;
                    background:#222;
                    border-radius:50%;
                    left:5px;
                    top:-8px;
                "></div>

                <div style="
                    position:absolute;
                    width:18px;
                    height:18px;
                    background:#222;
                    border-radius:50%;
                    right:5px;
                    top:-8px;
                "></div>

                <div style="
                    position:absolute;
                    width:18px;
                    height:18px;
                    background:#222;
                    border-radius:50%;
                    left:5px;
                    bottom:-8px;
                "></div>

                <div style="
                    position:absolute;
                    width:18px;
                    height:18px;
                    background:#222;
                    border-radius:50%;
                    right:5px;
                    bottom:-8px;
                "></div>
            </div>
        </div>
    </div>
    """


def run_joystick(
    steering: float,
    throttle: float,
):
    env = AutoPilotEnv()
    obs = env.reset()

    action_type = "maintain"
    brake = 0.0
    acceleration = max(0.0, throttle)

    if throttle < 0:
        action_type = "brake"
        brake = abs(throttle)
        acceleration = 0.0

    action = Action(
        action_type=action_type,
        steering=steering,
        acceleration=acceleration,
        brake=brake,
        lane_change="none",
    )

    next_obs, reward, done, info = env.step(action)

    payload: Dict[str, object] = {
        "action": action.model_dump(),
        "next_observation": next_obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }

    car_view = render_car_html(
        next_obs.lane_position or 0.0,
        steering,
        next_obs.speed or 0.0,
    )

    env.close()

    return car_view, json.dumps(payload, indent=2)


with gr.Blocks(title="AutoPilotEnv Live Dashboard") as demo:
    gr.Markdown("# AutoPilotEnv Live 3D Joystick Dashboard")

    gr.Markdown(
        "Move sliders like a joystick — updates instantly."
    )

    with gr.Row():
        steering = gr.Slider(
            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.1,
            label="Steering",
        )

        throttle = gr.Slider(
            minimum=-1.0,
            maximum=1.0,
            value=0.2,
            step=0.1,
            label="Throttle / Brake",
        )

    car_view = gr.HTML()
    output = gr.Code(
        language="json",
        label="Environment Output"
    )

    steering.change(
        run_joystick,
        inputs=[steering, throttle],
        outputs=[car_view, output],
    )

    throttle.change(
        run_joystick,
        inputs=[steering, throttle],
        outputs=[car_view, output],
    )

    demo.load(
        run_joystick,
        inputs=[steering, throttle],
        outputs=[car_view, output],
    )


@api.get("/", response_class=PlainTextResponse)
def home():
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        main()

    return buffer.getvalue()


@api.post("/reset")
def reset():
    obs = _env.reset()
    return obs.model_dump()


@api.post("/step")
def step(action: dict):
    action_obj = Action(**action)

    obs, reward, done, info = _env.step(action_obj)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@api.get("/state")
def state():
    return _env.state()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
