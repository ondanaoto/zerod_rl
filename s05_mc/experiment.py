from .optimizer import MonteCarloOptimizer


def main(epsilon: float = 0.1) -> None:
    optimizer = MonteCarloOptimizer(epsilon=epsilon)
    optimizer.update_policy()
    optimizer.render_pi()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo Policy Iteration")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon for epsilon-greedy policy",
    )
    main(
        epsilon=parser.parse_args().epsilon,
    )
