import numpy as np

from .action import Action
from .env import Env
from .policy import BestPolicy, Policy, RandomPolicy


def main(gamma: float, policy_name: str, iter_num: int) -> None:
    revenue = 0.0
    env = Env()
    env.step(Action.LEFT)
    policy_registory: dict[str, type[Policy]] = {
        "best": BestPolicy,
        "random": RandomPolicy,
    }
    policy = policy_registory[policy_name]()
    for _ in range(iter_num):
        action_probs = policy.get(env.state)
        action_idx: int = np.random.choice(len(Action), p=list(action_probs.values()))
        action = list(action_probs.keys())[action_idx]
        _, reward = env.step(action)
        revenue = reward + gamma * revenue

    print(f"Policy: {policy_name}, Revenue: {revenue:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MDP experiment.")
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="Discount factor (gamma)"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="best",
        choices=["best", "random"],
        help="Policy name",
    )
    parser.add_argument(
        "--iter_num", type=int, default=1000, help="Number of iterations"
    )
    args = parser.parse_args()
    main(
        gamma=args.gamma,
        policy_name=args.policy,
        iter_num=args.iter_num,
    )
