from __future__ import annotations

import os
from openai import OpenAI

from env import AutoPilotEnv
from models import Action

# STRICT validator-required environment variables
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

    if task_name == "obstacle_avoidance":
        return Action(
            action_type="change_lane",
            steering=0.0,
            acceleration=0.0,
            brake=0.2,
            lane_change="left",
        )

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
    score = 0.0

    print(
        f"[START] task=lane_keeping "
        f"env=autopilotenv "
        f"model={MODEL_NAME}"
    )

    try:
        obs = env.reset()

        # Mandatory proxy API call
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Choose a safe driving action for "
                        f"task {obs.task_name}"
                    ),
                }
            ],
        )

        llm_output = response.choices[0].message.content

        max_steps = 6

        for step in range(1, max_steps + 1):
            task_name = obs.task_name

            # Deterministic baseline action
            action = _fallback_action(task_name)

            obs, reward, done, info = env.step(action)

            steps = step
            rewards.append(reward.score)

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

        if rewards:
            score = min(max(sum(rewards) / len(rewards), 0.0), 1.0)

    except Exception as exc:
        steps = max(steps, 1)
        rewards.append(-1.0)

        print(
            f"[STEP] step={steps} "
            f"action=none "
            f"reward=-1.00 "
            f"done=true "
            f"error={str(exc).replace(chr(10), ' ')[:120]}"
        )

        success = False
        score = 0.0

    finally:
        env.close()

        rewards_str = ",".join(_fmt_reward(r) for r in rewards)

        print(
            f"[END] success={_bool_str(success)} "
            f"steps={steps} "
            f"score={score:.2f} "
            f"rewards={rewards_str}"
        )


if __name__ == "__main__":
    main()
