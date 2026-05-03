"""
Environment wrappers and utilities
"""
import gymnasium as gym
import numpy as np

class NormalizeRewardWrapper(gym.RewardWrapper):
    """
    Example wrapper to normalize reward.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def reward(self, reward):
        return reward / 10.0

class CarRacingActionWrapper(gym.ActionWrapper):
    """
    Simplifies CarRacing action space from 3D ([steer, gas, brake]) 
    to 2D ([steer, throttle/brake]).
    Combined throttle/brake:
    - Positive values: Gas
    - Negative values: Brake
    """
    def __init__(self, env):
        super().__init__(env)
        # Original: Box([-1, 0, 0], [1, 1, 1], (3,))
        # New: Box([-1, -1], [1, 1], (2,))
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

    def action(self, action):
        steer = action[0]
        throttle_brake = action[1]
        
        gas = 0.0
        brake = 0.0
        
        if throttle_brake > 0:
            gas = throttle_brake
        else:
            brake = -throttle_brake
            
        return np.array([steer, gas, brake], dtype=np.float32)

class AcrobotRewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping for Acrobot-swingup to encourage:
    1. Swinging up (tip height)
    2. Building momentum (angular velocity)
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        try:
            # Access underlying DMC physics
            physics = self.env.unwrapped.physics
            
            # Height of the tip of the second link
            # In DMC Acrobot, the vertical axis is Z
            tip_pos = physics.named.data.xpos['tip']
            tip_height = tip_pos[2] 
            
            # Angular velocity (sum of absolute velocities of the joints)
            ang_vel = np.sum(np.abs(physics.data.qvel))
            
            # Apply shaping: height + damped angular velocity
            # Height ranges roughly from -2 to 2.
            # We cap the ang_vel contribution to prevent it from dominating.
            reward += 0.5 * tip_height + 0.1 * np.clip(ang_vel, 0, 10)
            
        except Exception:
            # Fallback if physics access fails
            pass
            
        return obs, reward, terminated, truncated, info

class GoalConditionedWrapper(gym.Wrapper):
    """
    Wraps a standard Box environment into a Goal-Conditioned environment.
    By default, it treats the entire observation as both the state and the goal.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError(f"GoalConditionedWrapper only supports Box observation spaces, got {type(obs_space)}")
            
        self.observation_space = gym.spaces.Dict({
            "observation": obs_space,
            "achieved_goal": obs_space,
            "desired_goal": obs_space
        })
        # For Acrobot-swingup, a common 'goal' state is roughly [1, 0, 1, 0, 0, 0] 
        # (upright position with zero velocity).
        # We initialize it to zeros but it can be set externally.
        self.goal = np.zeros(obs_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Combine the base environment reward (e.g. shaping) with the goal-conditioned one.
        # This allows the agent to see 'progress' even before hitting the sparse goal.
        her_reward = self.compute_reward(obs, self.goal, info)
        return self._convert_obs(obs), reward + her_reward, terminated, truncated, info

    def _convert_obs(self, obs):
        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": obs.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32)
        }

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        # Distance-based sparse reward (standard for HER)
        # Returns 0 if close to goal, -1 otherwise.
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)
        
        # For Acrobot/pendulum tasks, we often care more about the orientation (cos/sin)
        # than the velocity for the 'success' condition.
        # Here we use the first 4 elements (cos1, sin1, cos2, sin2)
        if achieved_goal.shape[-1] >= 4:
            dist = np.linalg.norm(achieved_goal[..., :4] - desired_goal[..., :4], axis=-1)
        else:
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            
        # Relaxed threshold (0.3 is ~15-20 degrees error)
        reward = -(dist > 0.3).astype(np.float32)
        return reward[0] if reward.shape[0] == 1 else reward
