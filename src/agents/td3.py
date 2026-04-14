import torch
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
    def __init__(self, observation_space, dim_act, hidden_dim=256, max_action=1.0):
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
        self.max_action = max_action

    def forward(self, state):
        features = self.extractor(state)
        return self.max_action * self.net(features)

class TD3Agent(BaseAgent):
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        super().__init__(observation_space, action_space)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("TD3Agent currently only supports continuous action spaces (Box).")
            
        dim_act = action_space.shape[0]
        self.max_action = float(action_space.high[0]) if hasattr(action_space, 'high') else 1.0

        self.actor = Actor(observation_space, dim_act, max_action=self.max_action).to(self.device)
        self.actor_target = Actor(observation_space, dim_act, max_action=self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TwinCritic(observation_space, dim_act).to(self.device)
        self.critic_target = TwinCritic(observation_space, dim_act).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        if not evaluate:
            noise = np.random.normal(0, self.max_action * 0.1, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def update(self, batch):
        self.total_it += 1
        states, actions, rewards, next_states, dones = batch
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

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
