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
            
            # Apply shaping: 0.1 * height + 0.01 * velocity
            # Height ranges roughly from -2 to 2, ang_vel can be large
            reward += 0.1 * tip_height + 0.01 * ang_vel
            
        except Exception:
            # Fallback if physics access fails
            pass
            
        return obs, reward, terminated, truncated, info
