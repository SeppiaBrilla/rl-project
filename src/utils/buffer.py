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

    def add_batch(self, states, actions, rewards, next_states, dones):
        """
        Add a batch of transitions (from vectorized environments).
        We assume transitions are ordered [env0, env1, ..., envN]
        """
        n = len(states)
        if self.pos + n <= self.capacity:
            self.states[self.pos:self.pos+n] = states
            self.actions[self.pos:self.pos+n] = actions
            self.rewards[self.pos:self.pos+n] = rewards
            self.next_states[self.pos:self.pos+n] = next_states
            self.dones[self.pos:self.pos+n] = dones
            self.pos = (self.pos + n) % self.capacity
        else:
            # Wrap around safely to maintain interleaving if n_envs divides capacity
            first_part = self.capacity - self.pos
            second_part = n - first_part
            
            self.states[self.pos:] = states[:first_part]
            self.actions[self.pos:] = actions[:first_part]
            self.rewards[self.pos:] = rewards[:first_part]
            self.next_states[self.pos:] = next_states[:first_part]
            self.dones[self.pos:] = dones[:first_part]
            
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:]
            
            self.pos = second_part
            
        self.size = min(self.size + n, self.capacity)

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

    def add_batch(self, states, actions, rewards, values, log_probs, dones):
        """
        Add a batch of transitions. For RolloutBuffer, we assume capacity is matched 
        to (steps * n_envs).
        """
        n = len(states)
        if self.pos + n > self.capacity:
            # For RolloutBuffer we usually don't want to wrap around like this, 
            # but for safety:
            n = self.capacity - self.pos
            if n <= 0: return

        self.states[self.pos:self.pos+n] = states[:n]
        self.actions[self.pos:self.pos+n] = actions[:n]
        self.rewards[self.pos:self.pos+n] = rewards[:n]
        self.values[self.pos:self.pos+n] = values[:n]
        self.log_probs[self.pos:self.pos+n] = log_probs[:n]
        self.dones[self.pos:self.pos+n] = dones[:n]
        
        self.pos += n
        self.size = min(self.size + n, self.capacity)

    def compute_returns_and_advantages(self, last_value, last_done, gamma=0.99, gae_lambda=0.95):
        n_envs = len(last_value) if hasattr(last_value, "__len__") else 1
        last_gae_lam = np.zeros(n_envs, dtype=np.float32)
        
        # In GAE: delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
        # done_t should be the termination flag for the step t. 
        
        for step in reversed(range(0, self.size, n_envs)):
            if step == self.size - n_envs:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step : step + n_envs]
                next_value = self.values[step + n_envs : step + 2 * n_envs]

            delta = self.rewards[step : step + n_envs] + gamma * next_value * next_non_terminal - self.values[step : step + n_envs]
            self.advantages[step : step + n_envs] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]

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

    @property
    def full(self):
        return self.size == self.capacity