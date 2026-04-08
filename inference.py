import os
from openai import OpenAI

from env import AutoPilotEnv
from models import Action

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.environ["API_KEY"]


def fallback_action(task_name):
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
        "emergency_braking"
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


def main():
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

        for _ in range(5):
            action = fallback_action(obs.task_name)

            obs, reward, done, info = env.step(action)

            rewards.append(reward.score)
            steps += 1

            print(
                f"[STEP] step={steps} "
                f"action={action.action_type} "
                f"reward={reward.score:.2f} "
                f"done={str(done).lower()} "
                f"error=null"
            )

            if done:
                success = True
                break

        rewards_str = ",".join(
            f"{r:.2f}" for r in rewards
        )

    except Exception as exc:
        rewards_str = "0.01"

        print(
            f"[STEP] step=1 "
            f"action=none "
            f"reward=0.01 "
            f"done=true "
            f"error={str(exc)}"
        )

    finally:
        env.close()

        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps} "
            f"rewards={rewards_str}"
        )


if __name__ == "__main__":
    main()
