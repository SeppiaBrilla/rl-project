import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

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
        self.action_logstd = nn.Parameter(torch.zeros(dim_act))

    def forward(self, state):
        features = self.extractor(state)
        mean = self.net(features)
        std = self.action_logstd.exp().expand_as(mean)
        return mean, std

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
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, epochs=10, batch_size=64, entropy_coef=0.01):
        super().__init__(observation_space, action_space)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dim_act = action_space.shape[0]

        self.actor = Actor(observation_space, dim_act).to(self.device)
        self.critic = Critic(observation_space).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def get_value(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state)
        return value.cpu().numpy()[0, 0]

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.actor(state_ts)
            dist = Normal(mean, std)
            if evaluate:
                action = mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def update(self, rollout_buffer):
        states, actions, old_log_probs, returns, advantages = rollout_buffer.get_all()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                mean, std = self.actor(b_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                values = self.critic(b_states).squeeze(-1)
                critic_loss = F.mse_loss(values, b_returns)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                
        rollout_buffer.reset()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
        
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
        if 'actor_optim_state_dict' in checkpoint:
            self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
