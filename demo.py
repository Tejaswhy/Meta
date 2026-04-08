from env import AutoPilotEnv
from models import Action


def main():
    env = AutoPilotEnv()

    obs = env.reset()
    print("Initial Observation:")
    print(obs.model_dump())

    action = Action(
        action_type="maintain",
        steering=0.0,
        acceleration=0.2,
        brake=0.0,
        lane_change="none",
    )

    next_obs, reward, done, info = env.step(action)

    print("\nAfter One Step:")
    print("Observation:", next_obs.model_dump())
    print("Reward:", reward.model_dump())
    print("Done:", done)
    print("Info:", info)

    env.close()


if __name__ == "__main__":
    main()
