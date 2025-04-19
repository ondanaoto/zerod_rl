import numpy as np

from s01_bandit.bandit.nonstat import NonStatBandit

from .individual import Individual


class Environment:
    def __init__(self, init_rate: list[float], sigma: float, time: int):
        self.bandit = NonStatBandit(init_rate=init_rate, sigma=sigma)
        if time < 0:
            raise ValueError("time must be positive. time: ", time)
        self.time = time

    def reset(self) -> None:
        init_rate: list[float] = [
            np.random.random() for _ in range(self.bandit.arm_count)
        ]
        self.bandit.reset_init_rate(init_rate=init_rate)

    def eval(self, individual: Individual) -> float:
        total_reward = 0.0
        for _ in range(self.time):
            action = individual.agent.get_action()
            reward = self.bandit.play(action)
            total_reward += reward
            individual.estimator.update_estimates(action, reward)
        return total_reward / self.time

    @property
    def n_action(self) -> int:
        return self.bandit.arm_count
