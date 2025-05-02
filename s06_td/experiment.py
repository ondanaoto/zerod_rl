from c01_grid_world.env import GridWorld
from .agent import QLearningAgent

def main():
    agent = QLearningAgent()
    env = GridWorld()
    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
    agent.render_pi()


if __name__ == "__main__":
    main()
