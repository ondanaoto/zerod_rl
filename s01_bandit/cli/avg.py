from s01_bandit.action_value_estimator import AverageActionRewardEstimator
from s01_bandit.bandit.fix import FixBandit


def main():
    rates = [0.1, 0.5, 0.9]

    bandit = FixBandit(rates=rates)
    estimator = AverageActionRewardEstimator(n_actions=len(rates))
    n_trials = 1000
    for i in range(n_trials):
        action = i % len(rates)
        reward = bandit.play(action)
        estimator.update_estimates(action, reward)

    print("Estimated action rewards:")
    for action in range(bandit.arm_count):
        print(f"Action {action}: {estimator.estimate_action_reward(action)}")


if __name__ == "__main__":
    main()
