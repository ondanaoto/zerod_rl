from s01_bandit.action_value_estimator import AverageActionRewardEstimator
from s01_bandit.agent.eplison_greedy import EpsilonGreedyAgent
from s01_bandit.bandit.fix import FixBandit


def main(rates: list[float], epsilon: float, iterate: int, epoch: int):
    bandit = FixBandit(rates=rates)
    avg_reward_history = [[0.0 for _ in range(iterate)] for _ in range(epoch)]
    for e in range(epoch):
        estimator = AverageActionRewardEstimator(len(rates))
        agent = EpsilonGreedyAgent(estimator=estimator, epsilon=epsilon)
        total_reward = 0.0
        for i in range(iterate):
            action = agent.get_action()
            reward = bandit.play(action)
            estimator.update_estimates(action, reward)
            total_reward += reward
            avg_reward_history[e][i] = total_reward / (i + 1)
    for i in range(iterate):
        avg_avg_reward = sum([avg_reward_history[e][i] for e in range(epoch)]) / epoch
        print(avg_avg_reward)


if __name__ == "__main__":
    main(rates=[0.1, 0.5, 0.9], epsilon=0.05, iterate=1000, epoch=200)
