import torch
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
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
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_dim, dim_act)
        self.log_std_linear = nn.Linear(hidden_dim, dim_act)
        
        # Store bounds as buffers for GPU/CPU consistency
        self.register_buffer("action_low", torch.FloatTensor(action_low))
        self.register_buffer("action_high", torch.FloatTensor(action_high))

    def forward(self, state):
        features = self.extractor(state)
        x = self.net(features)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        # Robust scaling to [low, high]
        action = self.action_low + (y_t + 1.0) * 0.5 * (self.action_high - self.action_low)
        
        # Log-prob correction for tanh + scaling
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(0.5 * (self.action_high - self.action_low) * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = self.action_low + (torch.tanh(mean) + 1.0) * 0.5 * (self.action_high - self.action_low)
        return action, log_prob, mean

class SACAgent(BaseAgent):
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, 
                 batch_size=256, target_entropy=None, min_samples=1000, **kwargs):
        super().__init__(observation_space, action_space)
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.min_samples = min_samples
        
        if isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("SACAgent currently only supports continuous action spaces (Box).")
        
        dim_act = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high

        self.actor = Actor(observation_space, dim_act, action_low=self.action_low, action_high=self.action_high).to(self.device)
        self.critic = TwinCritic(observation_space, dim_act).to(self.device)
        self.critic_target = TwinCritic(observation_space, dim_act).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = target_entropy if target_entropy is not None else -float(dim_act)
        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=self.device, dtype=torch.float32)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        if len(state.shape) == len(self.observation_space.shape):
            state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_ts = torch.FloatTensor(state).to(self.device)
            
        with torch.no_grad():
            action, _, mean = self.actor.sample(state_ts)
        action = mean if evaluate else action
        
        if len(state.shape) == len(self.observation_space.shape):
            return action.cpu().numpy()[0]
        return action.cpu().numpy()

    def _update(self, batch):
        states, actions, rewards, next_states, dones = batch
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * min_q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        curr_actions, log_probs, _ = self.actor.sample(states)
        q1_curr, q2_curr = self.critic(states, curr_actions)
        min_q_curr = torch.min(q1_curr, q2_curr)
        
        actor_loss = ((self.log_alpha.exp() * log_probs) - min_q_curr).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.log_alpha.exp().item()
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
            'log_alpha': self.log_alpha,
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            'alpha_optim_state_dict': self.alpha_optim.state_dict()
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha.data = checkpoint['log_alpha'].data
        if 'actor_optim_state_dict' in checkpoint:
            self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict'])
