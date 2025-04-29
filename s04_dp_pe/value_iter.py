from .dp import ValueFunction
from .policy import Policy


def main(gamma: float = 0.9, threshold: float = 1e-4) -> None:
    policy = Policy()
    value_function = ValueFunction(policy, gamma)

    while True:
        diff = value_function.update()
        value_function.render()
        best_policy_dict = value_function.argmax_policy()
        policy.update_greedy(best_policy_dict)
        if diff < threshold:
            break


if __name__ == "__main__":
    main()
