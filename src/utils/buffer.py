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

    def sample_n_step(self, batch_size: int, n: int, n_envs: int, gamma: float):
        """
        Sample n-step transitions. 
        Returns: states, actions, n_step_rewards, n_step_next_states, n_step_dones
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        
        n_step_rewards = np.zeros(batch_size, dtype=np.float32)
        n_step_dones = np.zeros(batch_size, dtype=np.float32)
        
        curr_indices = indices.copy()
        gamma_power = 1.0
        for i in range(n):
            n_step_rewards += gamma_power * self.rewards[curr_indices]
            n_step_dones = np.maximum(n_step_dones, self.dones[curr_indices])
            
            # Mask to avoid moving past a terminal state
            mask = (n_step_dones == 0)
            if not np.any(mask):
                break
                
            next_indices = (curr_indices[mask] + n_envs) % self.capacity
            curr_indices[mask] = next_indices
            gamma_power *= gamma

        batch_next_states = self.next_states[curr_indices]
        
        return (
            torch.as_tensor(batch_states, dtype=torch.float32, device=self.device),
            torch.as_tensor(batch_actions, dtype=torch.float32, device=self.device),
            torch.as_tensor(n_step_rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(batch_next_states, dtype=torch.float32, device=self.device),
            torch.as_tensor(n_step_dones, dtype=torch.float32, device=self.device)
        )


    def __len__(self):
        return self.size

class HERReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, state_shape: tuple, action_shape: tuple, goal_shape: tuple, 
                 compute_reward_fn, device: str = "cpu", her_ratio: float = 0.8):
        super().__init__(capacity, state_shape, action_shape, device)
        self.goal_shape = goal_shape
        self.her_ratio = her_ratio
        self.compute_reward_fn = compute_reward_fn
        
        self.achieved_goals = np.zeros((capacity, *goal_shape), dtype=np.float32)
        self.next_achieved_goals = np.zeros((capacity, *goal_shape), dtype=np.float32)
        self.desired_goals = np.zeros((capacity, *goal_shape), dtype=np.float32)
        
        # Track episodes efficiently
        self.episode_id = np.full(capacity, -1, dtype=np.int64)
        self.step_indices = np.zeros(capacity, dtype=np.int32)
        self.episode_indices = {} # ep_id -> np.array of buffer indices
        self.current_episodes = {} # env_idx -> list of buffer indices in current episode
        self.next_episode_id = 0

    def add_batch(self, states, actions, rewards, next_states, terminateds, truncateds, 
                  achieved_goals, next_achieved_goals, desired_goals, env_indices=None):
        n = len(states)
        if env_indices is None:
            env_indices = np.arange(n)
            
        for i in range(n):
            idx = self.pos
            
            # Invalidate old episode if we are overwriting
            old_ep_id = self.episode_id[idx]
            if old_ep_id != -1 and old_ep_id in self.episode_indices:
                del self.episode_indices[old_ep_id]

            self.states[idx] = states[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_states[idx] = next_states[i]
            self.dones[idx] = terminateds[i] # Store only terminations for correct bootstrapping
            self.achieved_goals[idx] = achieved_goals[i]
            self.next_achieved_goals[idx] = next_achieved_goals[i]
            self.desired_goals[idx] = desired_goals[i]
            
            env_idx = env_indices[i]
            if env_idx not in self.current_episodes:
                self.current_episodes[env_idx] = []
            
            # Store step index within episode for O(1) lookup during sampling
            self.step_indices[idx] = len(self.current_episodes[env_idx])
            self.current_episodes[env_idx].append(idx)
            
            # Episode finished if either terminated or truncated
            if terminateds[i] or truncateds[i]:
                # Episode finished (either termination or truncation)
                # Episode finished, record boundaries with a unique ID
                ep_id = self.next_episode_id
                self.next_episode_id += 1
                
                ep_indices = np.array(self.current_episodes[env_idx])
                self.episode_indices[ep_id] = ep_indices
                self.episode_id[ep_indices] = ep_id
                
                self.current_episodes[env_idx] = []

            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        res_states = self.states[indices].copy()
        res_actions = self.actions[indices].copy()
        res_next_states = self.next_states[indices].copy()
        res_rewards = self.rewards[indices].copy()
        res_dones = self.dones[indices].copy()
        res_desired_goals = self.desired_goals[indices].copy()
        
        # HER relabeling (Vectorized)
        her_indices = np.where(np.random.uniform(0, 1, batch_size) < self.her_ratio)[0]
        
        # Pre-filter indices that belong to finished/valid episodes
        valid_her_mask = np.zeros(len(her_indices), dtype=bool)
        for i, idx_in_batch in enumerate(her_indices):
            buffer_idx = indices[idx_in_batch]
            ep_id = self.episode_id[buffer_idx]
            if ep_id != -1 and ep_id in self.episode_indices:
                valid_her_mask[i] = True
        
        her_indices = her_indices[valid_her_mask]
        
        if len(her_indices) > 0:
            her_buffer_indices = indices[her_indices]
            ep_ids = self.episode_id[her_buffer_indices]
            
            # Select future steps for each HER transition
            future_buffer_indices = []
            for i, buffer_idx in enumerate(her_buffer_indices):
                ep_id = ep_ids[i]
                target_ep = self.episode_indices[ep_id]
                step_idx = self.step_indices[buffer_idx]
                
                # Pick a future step (including current)
                future_step_idx = np.random.randint(step_idx, len(target_ep))
                future_buffer_indices.append(target_ep[future_step_idx])
            
            future_buffer_indices = np.array(future_buffer_indices)
            
            # Relabel goals
            new_goals = self.next_achieved_goals[future_buffer_indices]
            res_desired_goals[her_indices] = new_goals
            
            # Recompute rewards for all HER transitions at once
            new_rewards = self.compute_reward_fn(self.next_achieved_goals[her_buffer_indices], new_goals, None)
            res_rewards[her_indices] = new_rewards
            
            # Fix done signal: for relabeled transitions, terminal iff goal reached
            # (In sparse reward 0.0 is success, -1.0 is failure)
            res_dones[her_indices] = (new_rewards == 0).astype(np.float32)

        # Concatenate state and goal for the agent
        combined_states = np.concatenate([res_states, res_desired_goals], axis=-1)
        combined_next_states = np.concatenate([res_next_states, res_desired_goals], axis=-1)
        
        return (
            torch.as_tensor(combined_states, dtype=torch.float32, device=self.device),
            torch.as_tensor(res_actions, dtype=torch.float32, device=self.device),
            torch.as_tensor(res_rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(combined_next_states, dtype=torch.float32, device=self.device),
            torch.as_tensor(res_dones, dtype=torch.float32, device=self.device)
        )

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