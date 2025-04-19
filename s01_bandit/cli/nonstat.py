from s01_bandit.action_value_estimator.exp import ExponentialActionRewardEstimator
from s01_bandit.agent.eplison_greedy import EpsilonGreedyAgent
from s01_bandit.bandit.nonstat import NonStatBandit


def main(
    arm_count: int, sigma: float, alpha: float, epsilon: float, iterate: int, epoch: int
):
    avg_reward_history = [[0.0 for _ in range(iterate)] for _ in range(epoch)]
    avg_final_rates = []
    init_rate = [0.1, 0.4, 0.7, 0.9]
    for e in range(epoch):
        bandit = NonStatBandit(init_rate=init_rate, sigma=sigma)
        arm_count = len(init_rate)
        estimator = ExponentialActionRewardEstimator(n_action=arm_count, alpha=alpha)
        agent = EpsilonGreedyAgent(estimator=estimator, epsilon=epsilon)
        total_reward = 0.0
        for i in range(iterate):
            action = agent.get_action()
            reward = bandit.play(action)
            estimator.update_estimates(action, reward)
            total_reward += reward
            avg_reward_history[e][i] = total_reward / (i + 1)
        avg_final_rates.append([slot.rate for slot in bandit.slots])
    for i in range(iterate):
        avg_avg_reward = sum([avg_reward_history[e][i] for e in range(epoch)]) / epoch
        print("avg_reward per epoch: ", avg_avg_reward)
    print("rate_avg: ")
    for i in range(arm_count):
        print(f"arm {i}: {sum([avg_final_rates[e][i] for e in range(epoch)]) / epoch}")


if __name__ == "__main__":
    main(arm_count=4, sigma=0.01, alpha=0.1, epsilon=0.05, iterate=1000, epoch=100)
