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
        
        # Track episode boundaries for 'future' sampling
        self.episode_indices = [] # List of (start_idx, end_idx)
        self.current_episodes = {} # env_idx -> list of buffer indices in current episode

    def add_batch(self, states, actions, rewards, next_states, dones, 
                  achieved_goals, next_achieved_goals, desired_goals, env_indices=None):
        n = len(states)
        if env_indices is None:
            env_indices = np.arange(n)
            
        for i in range(n):
            idx = self.pos
            self.states[idx] = states[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_states[idx] = next_states[i]
            self.dones[idx] = dones[i]
            self.achieved_goals[idx] = achieved_goals[i]
            self.next_achieved_goals[idx] = next_achieved_goals[i]
            self.desired_goals[idx] = desired_goals[i]
            
            env_idx = env_indices[i]
            if env_idx not in self.current_episodes:
                self.current_episodes[env_idx] = []
            self.current_episodes[env_idx].append(idx)
            
            if dones[i]:
                # Episode finished, record boundaries
                ep_indices = self.current_episodes[env_idx]
                self.episode_indices.append(np.array(ep_indices))
                self.current_episodes[env_idx] = []
                
                # Cleanup old episodes that might have been overwritten
                # (Simple heuristic: if the first index of an episode is overwritten, 
                # the episode is invalid for HER future sampling)
                # In practice, we just keep a list and periodically filter or use a better structure.
                if len(self.episode_indices) > 1000: # Keep memory in check
                    self.episode_indices = self.episode_indices[-1000:]

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
        
        # HER relabeling
        her_indices = np.where(np.random.uniform(0, 1, batch_size) < self.her_ratio)[0]
        for idx_in_batch in her_indices:
            buffer_idx = indices[idx_in_batch]
            
            # Find which episode this buffer_idx belongs to
            # This search can be slow, but for a prototype it's okay.
            # Optimized version would store ep_id for each buffer index.
            target_ep = None
            for ep in reversed(self.episode_indices):
                if buffer_idx in ep:
                    target_ep = ep
                    break
            
            if target_ep is not None:
                # Find current step in episode
                step_idx = np.where(target_ep == buffer_idx)[0][0]
                # Pick a future step (including current one is sometimes allowed, but future is better)
                future_step_idx = np.random.randint(step_idx, len(target_ep))
                future_buffer_idx = target_ep[future_step_idx]
                
                # Relabel goal
                new_goal = self.next_achieved_goals[future_buffer_idx]
                res_desired_goals[idx_in_batch] = new_goal
                
                # Recompute reward and done
                res_rewards[idx_in_batch] = self.compute_reward_fn(self.next_achieved_goals[buffer_idx], new_goal, None)
                # For HER, done is typically only true if the goal is reached, 
                # but most implementations just recompute reward and leave done as original 
                # or recompute it too if the env provides it.
                # Here we assume standard GoalEnv behavior.

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