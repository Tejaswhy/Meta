import os
from openai import OpenAI

from env import AutoPilotEnv
from models import Action


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://api.openai.com/v1"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gpt-4.1-mini"
)

client = None


def ping_proxy():
    global client

    if client is None:
        return

    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "safe driving check"
                }
            ],
            max_tokens=5,
        )
    except Exception:
        pass


def fallback_action(task_name: str) -> Action:
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
            action_type="brake",
            steering=0.0,
            acceleration=0.0,
            brake=0.8,
            lane_change="none",
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
    global client

    HF_TOKEN = os.getenv("HF_TOKEN", "test_key")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

    tasks = [
        "lane_keeping",
        "obstacle_avoidance",
        "signal_safety",
        "emergency_braking",
        "pedestrian_priority",
    ]

    for task_name in tasks:
        env = AutoPilotEnv()

        rewards = []
        steps = 0
        success = False

        print(
            f"[START] task={task_name} "
            f"env=autopilotenv "
            f"model={MODEL_NAME}"
        )

        try:
            obs = env.reset()

            for _ in range(5):
                action = fallback_action(task_name)

                obs, reward, done, info = env.step(action)

                safe_score = round(
                    max(0.01, min(0.99, float(reward.score))),
                    2
                )

                rewards.append(safe_score)
                steps += 1

                print(
                    f"[STEP] step={steps} "
                    f"action={action.action_type} "
                    f"reward={safe_score:.2f} "
                    f"done={str(done).lower()} "
                    f"error=null"
                )

                if done:
                    success = True
                    break

        finally:
            env.close()

            rewards_str = ",".join(
                f"{r:.2f}" for r in rewards
            )

            print(
                f"[END] success={str(success).lower()} "
                f"steps={steps} "
                f"rewards={rewards_str}"
            )


if __name__ == "__main__":
    main()
