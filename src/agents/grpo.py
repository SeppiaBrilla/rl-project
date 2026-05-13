import torch
import csv
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym
import logging
import copy
from typing import List, Dict, Any

from .base import BaseAgent
from .networks import NatureCNN
from src.utils.buffer import ReplayBuffer

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
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_dim, dim_act)
        self.log_std_linear = nn.Linear(hidden_dim, dim_act)
        
        # Orthogonal initialization
        self.apply(self._init_weights)
        
        # Action space bounds (assumed [-1, 1] for dm_control after tanh)
        self.register_buffer("action_scale", torch.tensor(1.0))
        self.register_buffer("action_bias", torch.tensor(0.0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        features = self.extractor(state)
        x = self.net(features)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state, group_size=1):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # Sample in pre-tanh space
        x_t = dist.sample((group_size,)) # (G, B, D)
        
        # Action in post-tanh space
        action = torch.tanh(x_t)
        scaled_action = action * self.action_scale + self.action_bias
        
        # Log-prob with tanh correction: log_prob(scaled_action) = log_prob(x_t) - log(|det(J)|)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1) # (G, B)
        
        return scaled_action, log_prob, dist

class Critic(nn.Module):
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
            nn.Linear(dim_state + dim_act, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        features = self.extractor(state)
        return self.net(torch.cat([features, action], dim=-1))

class GRPOAgent(BaseAgent):
    """
    Revised GRPO for continuous control.
    Uses Squashed Gaussian policy, Group-Relative Advantages from Critics, 
    and n-step TD with Target Policy Smoothing.
    """
    def __init__(self, observation_space, action_space, 
                 lr=3e-4, gamma=0.99, 
                 group_size=16,
                 n_step=3,
                 eps_clip=0.2, 
                 beta=0.1, # KL penalty coefficient (increased for stability)
                 entropy_coef=0.01,
                 max_grad_norm=1.0,
                 tau=0.005, # Soft update for target networks
                 batch_size=256,
                 buffer_size=1000000,
                 learning_starts=5000,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 **kwargs):
        super().__init__(observation_space, action_space)
        self.gamma = gamma
        self.group_size = group_size
        self.n_step = n_step
        self.eps_clip = eps_clip
        self.beta = beta
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.total_it = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim_act = action_space.shape[0]

        # Actor and reference policy
        self.actor = Actor(observation_space, self.dim_act).to(self.device)
        self.ref_actor = copy.deepcopy(self.actor)
        for p in self.ref_actor.parameters(): p.requires_grad = False

        # Double Q-Critics
        self.critics = nn.ModuleList([
            Critic(observation_space, self.dim_act).to(self.device),
            Critic(observation_space, self.dim_act).to(self.device)
        ])
        self.target_critics = copy.deepcopy(self.critics)
        for p in self.target_critics.parameters(): p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size, observation_space.shape, action_space.shape, device=str(self.device))

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_ts = torch.FloatTensor(state).to(self.device)
        if len(state_ts.shape) == len(self.observation_space.shape):
            state_ts = state_ts.unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state_ts)
                action = torch.tanh(mean)
                log_prob = torch.zeros(action.shape[0], device=self.device)
            else:
                # Use group_size=1 for collection
                action, log_prob, _ = self.actor.sample(state_ts, group_size=1)
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
        
        if len(state.shape) == len(self.observation_space.shape):
            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def _collect_step(self, env, state):
        actions, _ = self.select_action(state)
        next_state, rewards, dones, truncateds, infos = env.step(actions)
        
        # Use only terminations for bootstrapping
        self.replay_buffer.add_batch(state, actions, rewards, next_state, dones)
        return next_state, rewards, np.logical_or(dones, truncateds)

    def _update(self, n_envs):
        self.total_it += 1
        
        # 1. Update Critics using n-step TD and Target Policy Smoothing
        states, actions, n_step_rewards, n_step_next_states, n_step_dones = \
            self.replay_buffer.sample_n_step(self.batch_size, self.n_step, n_envs, self.gamma)
        
        with torch.no_grad():
            # Target Policy Smoothing
            next_mean, next_log_std = self.actor(n_step_next_states)
            noise = (torch.randn_like(next_mean) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = torch.tanh(next_mean + noise)
            
            q1_target = self.target_critics[0](n_step_next_states, next_actions)
            q2_target = self.target_critics[1](n_step_next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            
            target_Q = n_step_rewards.unsqueeze(1) + (1 - n_step_dones.unsqueeze(1)) * (self.gamma ** self.n_step) * min_q_target

        current_Q1 = self.critics[0](states, actions)
        current_Q2 = self.critics[1](states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Update Actor using Group-Relative Advantage
        # Use detached samples for score-function gradient
        group_actions, log_probs, dist = self.actor.sample(states, self.group_size)
        
        # Align for evaluation: log_probs is (G, B), group_actions is (G, B, D)
        # We want (B, G) for log_probs
        log_probs = log_probs.transpose(0, 1)
        
        flat_actions = group_actions.transpose(0, 1).reshape(-1, self.dim_act)
        flat_states = states.repeat_interleave(self.group_size, dim=0)
        
        with torch.no_grad():
            q1 = self.critics[0](flat_states, flat_actions).reshape(self.batch_size, self.group_size)
            q2 = self.critics[1](flat_states, flat_actions).reshape(self.batch_size, self.group_size)
            group_Q = torch.min(q1, q2)
            
            # Group-Relative Advantage
            mean_Q = group_Q.mean(dim=1, keepdim=True)
            std_Q = group_Q.std(dim=1, keepdim=True) + 1e-6
            advantages = ((group_Q - mean_Q) / std_Q).detach()

        # KL Penalty from reference policy
        with torch.no_grad():
            ref_mean, ref_log_std = self.ref_actor(states)
            ref_dist = Normal(ref_mean, ref_log_std.exp())

        kl_div = torch.distributions.kl_divergence(dist, ref_dist).mean()
        entropy = dist.entropy().mean()

        # Actor Loss (Score-function gradient + regularization)
        actor_loss = -(advantages * log_probs).mean() + self.beta * kl_div - self.entropy_coef * entropy
        
        # Logging some info periodically
        if self.total_it % 500 == 0:
             print(f"Update {self.total_it}: L_actor={actor_loss.item():.4f}, KL={kl_div.item():.4f}, Adv={advantages.abs().mean().item():.4f}, Q={group_Q.mean().item():.4f}")

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # 3. Soft Updates
        with torch.no_grad():
            for param, target_param in zip(self.critics.parameters(), self.target_critics.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Update reference policy much more slowly (every 100 updates)
            if self.total_it % 100 == 0:
                for param, target_param in zip(self.actor.parameters(), self.ref_actor.parameters()):
                    target_param.data.copy_(0.05 * param.data + (0.95) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "kl_div": kl_div.item(),
            "entropy": entropy.item(),
            "mean_std": dist.stddev.mean().item()
        }

    def train(self, env, num_epochs: int, logger, render: bool, results_file: str = "results.csv"):
        results = []
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        
        total_steps = 0
        n_envs = env.num_envs
        pbar = tqdm(range(num_epochs), desc="Training GRPO")
        
        for epoch in pbar:
            epoch_metrics = []
            for _ in range(2001 // n_envs):
                total_steps += n_envs
                state, rewards, dones = self._collect_step(env, state)
                
                if len(self.replay_buffer) > self.learning_starts:
                    # Perform n_envs updates to maintain 1:1 update-to-step ratio
                    for _ in range(n_envs):
                        metrics = self._update(n_envs)
                        epoch_metrics.append(metrics)
            
            if epoch_metrics:
                avg_critic_loss = np.mean([m['critic_loss'] for m in epoch_metrics])
                avg_std = np.mean([m['mean_std'] for m in epoch_metrics])
                pbar.set_description(f"Epoch {epoch+1} | L_critic: {avg_critic_loss:.4f} | std: {avg_std:.4f}")

            if (epoch + 1) % 10 == 0:
                eval_rewards = self._evaluate(env, num_episodes=5)
                eval_rewards = [float(r) for r in eval_rewards]
                
                results.append({
                    "epoch": epoch + 1,
                    "eval_reward_mean": float(np.mean(eval_rewards)),
                    "eval_reward_std": float(np.std(eval_rewards)),
                    "eval_reward_1": eval_rewards[0],
                    "eval_reward_2": eval_rewards[1],
                    "eval_reward_3": eval_rewards[2],
                    "eval_reward_4": eval_rewards[3],
                    "eval_reward_5": eval_rewards[4]
                })
                
                logger.info(f"Epoch {epoch+1} | Eval Reward: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
                
                with open(results_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)

    def save(self, filepath: str):
        state_dict = {
            'actor': self.actor.state_dict(),
            'ref_actor': self.ref_actor.state_dict(),
            'critics': self.critics.state_dict(),
            'target_critics': self.target_critics.state_dict()
        }
        torch.save(state_dict, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.ref_actor.load_state_dict(ckpt['ref_actor'])
        self.critics.load_state_dict(ckpt['critics'])
        self.target_critics.load_state_dict(ckpt['target_critics'])
