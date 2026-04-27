import torch
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from .base import BaseAgent
from .networks import NatureCNN

class Critic(nn.Module):
    def __init__(self, dim_state, dim_act, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_state + dim_act, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class TwinCritic(nn.Module):
    def __init__(self, observation_space, dim_act, hidden_dim=256):
        super().__init__()
        shape = observation_space.shape
        if len(shape) == 3:
            self.extractor = NatureCNN(observation_space, hidden_dim)
            dim_state = hidden_dim
        else:
            self.extractor = nn.Identity()
            dim_state = shape[0]

        self.q1 = Critic(dim_state, dim_act, hidden_dim)
        self.q2 = Critic(dim_state, dim_act, hidden_dim)

    def forward(self, state, action):
        features = self.extractor(state)
        return self.q1(features, action), self.q2(features, action)

    def q1_forward(self, state, action):
        features = self.extractor(state)
        return self.q1(features, action)

class Actor(nn.Module):
    def __init__(self, observation_space, dim_act, hidden_dim=256, action_low=None, action_high=None):
        super().__init__()
        shape = observation_space.shape
        if len(shape) == 3:
            self.extractor = NatureCNN(observation_space, hidden_dim)
            dim_state = hidden_dim
        else:
            self.extractor = nn.Identity()
            dim_state = shape[0]

        self.net = nn.Sequential(
            nn.Linear(dim_state, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_act),
            nn.Tanh()
        )
        self.register_buffer("action_low", torch.FloatTensor(action_low))
        self.register_buffer("action_high", torch.FloatTensor(action_high))

    def forward(self, state):
        features = self.extractor(state)
        y_t = self.net(features)
        # Robust scaling to [low, high]
        return self.action_low + (y_t + 1.0) * 0.5 * (self.action_high - self.action_low)

class TD3Agent(BaseAgent):
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.99, tau=0.005, batch_size:int=256,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, min_samples=1000):
        super().__init__(observation_space, action_space)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.total_it = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("TD3Agent currently only supports continuous action spaces (Box).")
            
        dim_act = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high

        self.actor = Actor(observation_space, dim_act, action_low=self.action_low, action_high=self.action_high).to(self.device)
        self.actor_target = Actor(observation_space, dim_act, action_low=self.action_low, action_high=self.action_high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TwinCritic(observation_space, dim_act).to(self.device)
        self.critic_target = TwinCritic(observation_space, dim_act).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        if len(state.shape) == len(self.observation_space.shape):
            state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_ts = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action = self.actor(state_ts).cpu().numpy()
            
        if not evaluate:
            # Scale noise width to the range width for consistent exploration
            range_width = (self.action_high - self.action_low)
            noise = np.random.normal(0, 0.1 * range_width, size=action.shape)
            action = (action + noise).clip(self.action_low, self.action_high)
            
        if len(state.shape) == len(self.observation_space.shape):
            return action[0]
        return action

    def _update(self, batch):
        self.total_it += 1
        states, actions, rewards, next_states, dones = batch
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(
                self.actor_target.action_low, self.actor_target.action_high
            )

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss_val = 0.0
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            actor_loss_val = actor_loss.item()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val
        }

    def train(self, env, num_epochs: int, logger, render: bool, results_file: str = "results.csv"):
        from src.utils.buffer import ReplayBuffer
        from tqdm import tqdm
        
        buffer = ReplayBuffer(capacity=100_000, state_shape=self.observation_space.shape, 
                             action_shape=self.action_space.shape, device=self.device)
        
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        
        n_envs = getattr(env, "num_envs", 1)
        episode_rewards = np.zeros(n_envs)
        episode_count = 0
        
        results = []
        epoch_losses = []
        
        for epoch in tqdm(range(num_epochs)):
            for _ in range(2000 // n_envs):
                # Warmup: sample random actions if buffer is too small
                if len(buffer) < self.min_samples:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(state)
                    
                next_state, rewards, dones, truncateds, infos = env.step(action)
                masks = np.logical_or(dones, truncateds)
                
                real_next_state = next_state.copy()
                if isinstance(infos, dict) and "final_observation" in infos:
                    has_final = infos.get("_final_observation", masks)
                    for i in range(n_envs):
                        if has_final[i] and infos["final_observation"][i] is not None:
                            real_next_state[i] = infos["final_observation"][i]
                            
                # For off-policy agents, we only want to stop bootstrapping if terminated.
                # If truncated, we bootstrap from the final_observation.
                buffer.add_batch(state, action, rewards, real_next_state, dones)

                if len(buffer) >= self.min_samples:
                    for _ in range(n_envs):
                        losses = self._update(buffer.sample(self.batch_size))
                        epoch_losses.append(losses["critic_loss"])
                    
                state = next_state
                episode_rewards += rewards
                
                for i in range(n_envs):
                    if masks[i]:
                        episode_count += 1
                        episode_rewards[i] = 0
            
            # Evaluation every 10 epochs
            if (epoch + 1) % 10 == 0:
                eval_rewards = self._evaluate(env, num_episodes=5)
                # Convert numpy scalars to standard floats for clean CSV saving
                eval_rewards = [float(r) for r in eval_rewards]
                avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

                results.append({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "eval_reward_mean": float(np.mean(eval_rewards)),
                    "eval_reward_std": float(np.std(eval_rewards)),
                    "eval_reward_1": eval_rewards[0],
                    "eval_reward_2": eval_rewards[1],
                    "eval_reward_3": eval_rewards[2],
                    "eval_reward_4": eval_rewards[3],
                    "eval_reward_5": eval_rewards[4],
                })
                epoch_losses = [] # Reset losses for next period
                
                # Periodically save results
                with open(results_file, 'w', newline='') as f:
                    if results:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
                logger.info(f"Epoch {epoch+1} | Eval Reward: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
            
        # Final save
        with open(results_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

    def save(self, filepath: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        if 'actor_optim_state_dict' in checkpoint:
            self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
