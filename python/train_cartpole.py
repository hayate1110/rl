import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import copy
import gymnasium as gym
import tqdm
import random
import matplotlib.pyplot as plt
import imageio


class PolicyNet(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(observation_size, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x

class ValueNet(nn.Module):
    def __init__(self, observation_size):
        super().__init__()
        self.l1 = nn.Linear(observation_size, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self, device='cpu', observation_size=4, action_size=2):
        self.device = device
        self.gamma = 0.99
        self.lam = 0.95
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.normalize_advantage = True
        self.clip_range = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.reward_clip = [-5, 5]

        self.max_grad_norm = 0.5
        
        self.new_pi = PolicyNet(observation_size, action_size).to(self.device)
        self.old_pi = copy.deepcopy(self.new_pi).to(self.device).eval()
        self.v = ValueNet(observation_size).to(self.device)

        self.optimizer_new_pi = optim.Adam(self.new_pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.advantages = []
        self.returns = []
        self.dones = []
        self.terminateds = []

    def get_action(self, state):
        """
        state: ndarray of shape (state_dim,)
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) # shape: (1, state_dim)
        # old_piを使ってアクションを選択する
        probs = self.old_pi(state) # shape: (1, num_actions)
        probs = probs[0] # shape: (num_actions,)
        dist = Categorical(probs)
        action = dist.sample().item()
        return action

    def get_action_greedy(self, state):
        """
        state: ndarray of shape (state_dim,)
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) # shape: (1, state_dim)
        # new_piを使ってアクションを選択する
        probs = self.new_pi(state) # shape: (1, num_actions)
        probs = probs[0] # shape: (num_actions,)
        action = torch.argmax(probs).item()
        return action

    def get_ratio(self, states, actions):
        """
        states: Tensor of shape (batch_size, state_dim)
        actions: Tensor of shape (batch_size,) with integer action indices
        returns: Tensor of shape (batch_size,) containing the log probability of each action
        """
        with torch.no_grad():
            old_probs = self.old_pi(states)  # shape: (B, num_actions)
        new_probs = self.new_pi(states)  # shape: (B, num_actions)
        
        old_dist = Categorical(old_probs)
        new_dist = Categorical(new_probs)
        
        old_log_probs = old_dist.log_prob(actions)  # shape: (B,)
        new_log_probs = new_dist.log_prob(actions)  # shape: (B,)

        old_log_probs = old_log_probs.detach()

        ratio = torch.exp(new_log_probs - old_log_probs)
        
        return ratio

    def get_entropy(self, states):
        """
        states (Tensor or ndarray): shape (B, state_dim)
        returns: Tensor of shape (B,) — entropy values for each state
        """    
        # モデルからアクション確率を取得
        probs = self.new_pi(states)  # shape: (B, num_actions)
    
        # 分布を構築し、エントロピーを取得
        dist = Categorical(probs)
        entropy = dist.entropy()  # shape: (B,)
    
        return entropy

    def add(self, state, action, reward, next_state, terminated, truncated):
        """
        state: ndarray of shape (state_dim,)
        action: int
        reward: float
        next_state: ndarray of shape (state_dim,)
        terminated: bool
        truncated: bool
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(terminated or truncated)
        self.terminateds.append(terminated)

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.advantages.clear()
        self.dones.clear()
        self.terminateds.clear()
        self.returns.clear()

    def call_on_step(self, state, action, reward, next_state, terminated, truncated):
        if self.reward_clip is not None:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])
            
        self.add(state, action, reward, next_state, terminated, truncated)
    
    def finish_trajectory(self):
        advantage = 0
        states = torch.tensor(np.stack(self.states), dtype=torch.float32).to(self.device) # shape: (T, state_dim)
        next_states = torch.tensor(np.stack(self.next_states), dtype=torch.float32).to(self.device) # shape: (T, state_dim)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device) # shape: (T,)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device) # shape: (T,)
        terminateds = torch.tensor(self.terminateds, dtype=torch.float32).to(self.device) # shape: (T,)

        with torch.no_grad():
            v = self.v(states).squeeze(1) # shape: (T,)
            next_v = self.v(next_states).squeeze(1) # shape: (T,)

        for i in reversed(range(len(states))):
            # δ_t = r_t + γ(1-done)V(s_{t+1}) - V(s_t)
            delta = rewards[i] + self.gamma * next_v[i] * (1 - terminateds[i]) - v[i]
            # A_t = δ_t + γλ(1-done)A_{t+1}
            advantage = delta + self.gamma * self.lam * advantage * (1 - dones[i])
            self.advantages.append(advantage.item())

        self.advantages = list(reversed(self.advantages)) # shape: (T,)

        self.returns = torch.tensor(self.advantages, dtype=torch.float32).to(self.device) + v
        self.returns = self.returns.detach().tolist() # shape: (T,)

    def get_samples(self, idx):
        states = []
        actions = []
        advantages = []
        returns = []
        
        for i in idx:
            states.append(self.states[i])
            actions.append(self.actions[i])
            advantages.append(self.advantages[i])
            returns.append(self.returns[i])

        return states, actions, advantages, returns
        
    def update(self, idx):
        states, actions, advantages, returns = self.get_samples(idx)
        
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device) # shape: (B, state_dim)
        actions = torch.tensor(np.stack(actions), dtype=torch.long).to(self.device) # shape: (B,)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device) # shape: (B,)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device) # shape: (B,)

        ratio = self.get_ratio(states, actions)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        v = self.v(states).squeeze(1)  # shape: (B,)

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(returns, v)
        
        entropy = self.get_entropy(states)

        entropy_loss = -torch.mean(entropy)

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        self.optimizer_v.zero_grad()
        self.optimizer_new_pi.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.new_pi.parameters(), self.max_grad_norm) # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), self.max_grad_norm) # 勾配クリッピング
        self.optimizer_v.step()
        self.optimizer_new_pi.step()

    def save(self, path):
        torch.save({
            'policy_state_dict': self.new_pi.state_dict(),
            'value_state_dict': self.v.state_dict(),
            'optimizer_pi_state_dict': self.optimizer_new_pi.state_dict(),
            'optimizer_v_state_dict': self.optimizer_v.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.new_pi.load_state_dict(checkpoint['policy_state_dict'])
        self.v.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer_new_pi.load_state_dict(checkpoint['optimizer_pi_state_dict'])
        self.optimizer_v.load_state_dict(checkpoint['optimizer_v_state_dict'])

def main():
    print("Using device:", 'cuda' if torch.cuda.is_available() else 'cpu')
    n_updates = 500
    n_steps_per_update = 1024
    epochs_per_update = 3
    batch_size = 32
    env = gym.make('CartPole-v1')
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent('cuda' if torch.cuda.is_available() else 'cpu', observation_size, action_size)
    reward_history = []

    moving_avg_window = 100
    solved_reward = 475

    for _ in tqdm.tqdm(range(n_updates)):
        state, info = env.reset()
        episode_reward = 0
        agent.reset()

        for _ in range(n_steps_per_update):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.call_on_step(state, action, reward, next_state, terminated, truncated)
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                state, info = env.reset()
                reward_history.append(episode_reward)
                episode_reward = 0

        agent.finish_trajectory()
        
        for _ in range(epochs_per_update):
            indices = np.arange(len(agent.states))
            np.random.shuffle(indices)
            for start in range(0, len(agent.states), batch_size):
                batch_idx = indices[start:start+batch_size]
                agent.update(batch_idx)
        
        agent.old_pi = copy.deepcopy(agent.new_pi)
        agent.old_pi.eval()

        if len(reward_history) >= moving_avg_window:
            recent_avg = sum(reward_history[-moving_avg_window:]) / moving_avg_window
            if recent_avg >= solved_reward:
                print(f"Solved! Average reward: {recent_avg}")
                break
    
    plt.plot(reward_history)
    plt.savefig('../outputs/reward_history_cartpole.png')

    agent.save('../outputs/cartpole.pth')

if __name__ == "__main__":
    main()