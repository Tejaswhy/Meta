
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from env import AutoPilotEnv
from models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _fmt_reward(value: float) -> str:
    return f"{value:.2f}"


def _fallback_action(task_name: str) -> Action:
    if task_name == "lane_keeping":
        return Action(
            action_type="maintain",
            steering=0.0,
            acceleration=0.2,
            brake=0.0,
        )

    if task_name == "obstacle_avoidance":
        return Action(
            action_type="change_lane",
            lane_change="left",
            brake=0.2,
        )

    return Action(
        action_type="stop",
        brake=1.0,
    )


def main() -> None:
    env = AutoPilotEnv()

    rewards = []
    steps = 0
    success = False

    print(
        f"[START] task=lane_keeping "
        f"env=autopilotenv model={MODEL_NAME}"
    )

    try:
        obs = env.reset()

        max_steps = 6

        for step in range(1, max_steps + 1):
            task_name = obs.task_name

            action = _fallback_action(task_name)

            obs, reward, done, info = env.step(action)

            steps = step
            rewards.append(_fmt_reward(reward.score))

            print(
                f"[STEP] step={step} "
                f"action={action.action_type} "
                f"reward={_fmt_reward(reward.score)} "
                f"done={_bool_str(done)} "
                f"error=null"
            )

            if done:
                success = reward.score > 0
                break

    except Exception as exc:
        steps = 1
        rewards.append("-1.00")

        print(
            f"[STEP] step=1 action=none "
            f"reward=-1.00 "
            f"done=true "
            f"error={str(exc).replace(chr(10), ' ')[:120]}"
        )

        success = False

    finally:
        env.close()

        print(
            f"[END] success={_bool_str(success)} "
            f"steps={steps} "
            f"rewards={','.join(rewards)}"
        )






if __name__ == "__main__":
    main()


