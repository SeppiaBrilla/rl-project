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
