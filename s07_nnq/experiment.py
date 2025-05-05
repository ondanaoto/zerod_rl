from c01_grid_world.env import GridWorld

from .agent import DeepQLearningAgent


def main():
    agent = DeepQLearningAgent()
    env = GridWorld()
    for _ in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

    agent.render_pi()


if __name__ == "__main__":
    main()
