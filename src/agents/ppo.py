import torch
import csv
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from src.utils.data_collector import DataCollector
import logging
from src.utils.buffer import RolloutBuffer

from .base import BaseAgent
from .networks import NatureCNN

class Actor(nn.Module):
    def __init__(self, observation_space, dim_act, hidden_dim=256):
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
            nn.Linear(hidden_dim, dim_act)
        )

    def forward(self, state):
        features = self.extractor(state)
        return self.net(features)

class Critic(nn.Module):
    def __init__(self, observation_space, hidden_dim=256):
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
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.extractor(state)
        return self.net(features)

class PPOAgent(BaseAgent):
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.999, gae_lambda=0.99,
                 clip_ratio=0.2, epochs=10, K_epochs=100, batch_size=128, entropy_coef=0.001, max_grad_norm=0.5, **kwargs):
        super().__init__(observation_space, action_space)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.is_discrete = False
        dim_act = action_space.shape[0]

        self.actor = Actor(observation_space, dim_act).to(self.device)
        self.critic = Critic(observation_space).to(self.device)
        
        self.action_logstd = nn.Parameter(torch.zeros(dim_act, device=self.device))

        self.actor_optim = optim.Adam(list(self.actor.parameters()) + ([self.action_logstd] if self.action_logstd is not None else []), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def get_value(self, state):
        if len(state.shape) == len(self.observation_space.shape):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
            
        with torch.no_grad():
            value = self.critic(state)
        return value.cpu().numpy().flatten() # Return array of values

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        if len(state.shape) == len(self.observation_space.shape):
            state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_ts = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            mean = self.actor(state_ts)
            std = self.action_logstd.exp().expand_as(mean)
            dist = Normal(mean, std)
            
            if evaluate:
                action = mean
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Return as batch if input was batch
        if len(state.shape) == len(self.observation_space.shape):
            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def train(self, env, num_epochs: int, logger, render: bool, results_file: str = "results.csv"):
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        n_envs = getattr(env, "num_envs", 1)
        steps_per_rollout = 2000 // n_envs
        rollout_capacity = steps_per_rollout * n_envs

        # Initialize buffer with the correct device
        buffer = RolloutBuffer(capacity=rollout_capacity, state_shape=self.observation_space.shape, 
                              action_shape=self.action_space.shape, device=self.device)
        
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        
        episode_rewards = np.zeros(n_envs)
        episode_counts = 0
        
        results = []
        epoch_losses = []

        for epoch in tqdm(range(num_epochs)):
            # 1. Collect Rollouts
            for _ in range(steps_per_rollout):
                action, log_prob = self.select_action(state)
                value = self.get_value(state)
                
                next_state, rewards, dones, truncateds, infos = env.step(action)
                
                # Combine done and truncated for transition storage and reward resetting
                masks = np.logical_or(dones, truncateds)
                
                # Handle final observations for auto-resetting vectorized environments
                # We bootstrap the value of the terminal state if an episode was truncated (timeout)
                if isinstance(infos, dict) and "final_observation" in infos:
                    has_final = infos.get("_final_observation", truncateds)
                    for i in range(n_envs):
                        if has_final[i] and infos["final_observation"][i] is not None and truncateds[i]:
                            v_terminal = self.get_value(infos["final_observation"][i]).item()
                            rewards[i] += self.gamma * v_terminal
                
                buffer.add_batch(state, action, rewards, value, log_prob, masks)
                
                state = next_state
                episode_rewards += rewards
                
                for i in range(n_envs):
                    if masks[i]:
                        episode_counts += 1
                        episode_rewards[i] = 0
            
            # Compute returns and advantages for the collected rollout
            last_value = self.get_value(state) # Last value for each env
            last_dones = masks.copy()
            
            buffer.compute_returns_and_advantages(last_value, last_dones, 
                                                   gamma=self.gamma, gae_lambda=self.gae_lambda)
            
            # 2. Update Policy and Value Networks
            states, actions, old_log_probs, returns, advantages = buffer.get_all()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            dataset_size = states.size(0)
            indices = np.arange(dataset_size)

            update_losses = []
            for _ in range(self.K_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    batch_idx = indices[start:end]
                    
                    b_states = states[batch_idx]
                    b_actions = actions[batch_idx]
                    b_old_log_probs = old_log_probs[batch_idx]
                    b_returns = returns[batch_idx]
                    b_advantages = advantages[batch_idx]

                    # Actor update
                    logits_or_mean = self.actor(b_states)
                    std = self.action_logstd.exp().expand_as(logits_or_mean)
                    dist = Normal(logits_or_mean, std)
                    
                    new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    # Correct ratio: exp(new_log_prob - old_log_prob)
                    ratio = torch.exp(new_log_probs - b_old_log_probs)

                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                    
                    # Missing entropy was added back here
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.action_logstd], self.max_grad_norm)
                    self.actor_optim.step()

                    # Critic update
                    values = self.critic(b_states).squeeze(-1)
                    value_loss = F.mse_loss(values, b_returns)

                    self.critic_optim.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()
                    update_losses.append(value_loss.item())
            
            epoch_losses.append(np.mean(update_losses))
            buffer.reset()

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
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
        }
        if self.action_logstd is not None:
            save_dict['action_logstd'] = self.action_logstd
        torch.save(save_dict, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if self.action_logstd is not None and 'action_logstd' in checkpoint:
            self.action_logstd.data.copy_(checkpoint['action_logstd'].data)
        if 'actor_optim_state_dict' in checkpoint:
            self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
