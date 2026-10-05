from train_mountain_car_continuous import Agent
import gymnasium as gym
import torch
import imageio

def main():
    env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent('cuda' if torch.cuda.is_available() else 'cpu', observation_size, action_size)
    agent.load("../outputs/mountain_car_continuous.pth")
    
    render_data = []

    agent.reset()
    state, info = env.reset()
    render_data.append(env.render())
    
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.get_action_greedy(state)
        action_scaled = (action + 1.0) / 2.0 * (env.action_space.high - env.action_space.low) + env.action_space.low
        next_state, reward, terminated, truncated, info = env.step(action_scaled)
        state = next_state
        render_data.append(env.render())
    
    with imageio.get_writer('../outputs/mountain_car_continuous.gif', fps=24) as writer:
        for frame in render_data:
            writer.append_data(frame)

if __name__ == "__main__":
    main()