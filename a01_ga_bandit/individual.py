from a01_ga_bandit.gene import Gene
from s01_bandit.action_value_estimator.exp import ExponentialActionRewardEstimator
from s01_bandit.agent.eplison_greedy import EpsilonGreedyAgent


class Individual:
    def __init__(self, n_action: int, gene: Gene):
        self.n_action = n_action
        self.estimator = ExponentialActionRewardEstimator(
            n_action=n_action, alpha=gene.alpha
        )
        self.agent = EpsilonGreedyAgent(estimator=self.estimator, epsilon=gene.epsilon)
        self.gene = gene
