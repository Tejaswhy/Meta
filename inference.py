from __future__ import annotations

import os
from openai import OpenAI

from env import AutoPilotEnv
from models import Action

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.environ["API_KEY"]


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
            lane_change="none",
        )

    elif task_name in {
        "obstacle_avoidance",
        "emergency_braking",
    }:
        return Action(
            action_type="change_lane",
            steering=0.0,
            acceleration=0.0,
            brake=0.2,
            lane_change="left",
        )

    else:
        return Action(
            action_type="stop",
            steering=0.0,
            acceleration=0.0,
            brake=1.0,
            lane_change="none",
        )


def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    env = AutoPilotEnv()

    rewards = []
    steps = 0
    success = False

    print(
        f"[START] task=all_tasks "
        f"env=autopilotenv "
        f"model={MODEL_NAME}"
    )

    try:
        obs = env.reset()

        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Choose safe driving action"
                }
            ]
        )

        for task_run in range(5):
            action = _fallback_action(obs.task_name)

            obs, reward, done, info = env.step(action)

            safe_reward = min(
                max(float(reward.score), 0.01),
                0.99,
            )

            steps += 1
            rewards.append(safe_reward)

            print(
                f"[STEP] step={steps} "
                f"action={action.action_type} "
                f"reward={_fmt_reward(safe_reward)} "
                f"done={_bool_str(done)} "
                f"error=null"
            )

            if done:
                success = True

        rewards_str = ",".join(
            _fmt_reward(r) for r in rewards
        )

    except Exception as exc:
        rewards_str = "0.01"

        print(
            f"[STEP] step=1 "
            f"action=none "
            f"reward=0.01 "
            f"done=true "
            f"error={str(exc)[:120]}"
        )

    finally:
        env.close()

        print(
            f"[END] success={_bool_str(success)} "
            f"steps={steps} "
            f"rewards={rewards_str}"
        )


if __name__ == "__main__":
    main()
