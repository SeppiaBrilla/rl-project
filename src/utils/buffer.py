"""
Replay Buffer Utility
"""
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple, action_shape: tuple = (), device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32) # Changed to float32 and added action_shape
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        )

    def __len__(self):
        return self.size

class RolloutBuffer:
    def __init__(self, capacity: int, state_shape: tuple, action_shape: tuple = (), device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, value, log_prob, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def compute_returns_and_advantages(self, last_value, done, gamma=0.99, gae_lambda=0.95):
        last_gae_lam = 0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
                
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.returns = self.advantages + self.values[:self.size]

    def get_all(self):
        return (
            torch.as_tensor(self.states[:self.size], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actions[:self.size], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.log_probs[:self.size], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.returns[:self.size], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.advantages[:self.size], dtype=torch.float32, device=self.device)
        )
        
    def reset(self):
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size
