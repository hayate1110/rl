from train_cartpole import Agent
import gymnasium as gym
import torch
import imageio

def main():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent('cuda' if torch.cuda.is_available() else 'cpu', observation_size, action_size)
    agent.load("../outputs/cartpole.pth")

    render_data = []

    agent.reset()
    state, info = env.reset()
    render_data.append(env.render())
    
    terminated = False
    truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            action = agent.get_action_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        render_data.append(env.render())
    
    with imageio.get_writer('../outputs/cartpole.gif', fps=24) as writer:
        for frame in render_data:
            writer.append_data(frame)

if __name__ == "__main__":
    main()