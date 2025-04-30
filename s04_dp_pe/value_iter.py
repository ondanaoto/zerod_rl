from .dp import ValueFunction
from .policy import Policy


def main(gamma: float = 0.9, threshold: float = 1e-4) -> None:
    policy = Policy()
    value_function = ValueFunction(policy, gamma)

    while True:
        diff = value_function.update(policy)
        value_function.render()
        policy = value_function.get_greedy_policy()
        if diff < threshold:
            break


if __name__ == "__main__":
    main()
