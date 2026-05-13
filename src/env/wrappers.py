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
            tip_pos = physics.named.data.site_xpos['tip']
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
        # We now calculate the combined reward (shaping + sparse) inside compute_reward
        # to ensure consistency for HER relabeling.
        combined_reward = self.compute_reward(obs, self.goal, info)
        return self._convert_obs(obs), combined_reward, terminated, truncated, info

    def _convert_obs(self, obs):
        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": obs.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32)
        }

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        # Distance-based sparse reward (standard for HER)
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)
        
        # 1. Sparse Goal Component
        if achieved_goal.shape[-1] >= 4:
            dist = np.linalg.norm(achieved_goal[..., :4] - desired_goal[..., :4], axis=-1)
        else:
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        sparse_reward = -(dist > 0.3).astype(np.float32)
        
        # 2. Dense Shaping Component (Consistent for all goals)
        # In Acrobot, achieved_goal is [cos1, sin1, cos2, sin2, v1, v2]
        shaping_reward = 0.0
        if achieved_goal.shape[-1] >= 6:
            # DMC orientations order: [sin1, sin12, cos1, cos12]
            # In Acrobot-swingup, theta=0 (cos=1) is UP and theta=pi (cos=-1) is DOWN.
            cos1 = achieved_goal[..., 2]
            cos12 = achieved_goal[..., 3]
            tip_height = cos1 + cos12
            
            # Angular velocity magnitude
            v1 = achieved_goal[..., 4]
            v2 = achieved_goal[..., 5]
            ang_vel = np.abs(v1) + np.abs(v2)
            
            shaping_reward = 0.5 * tip_height + 0.1 * np.clip(ang_vel, 0, 10)
            
        reward = sparse_reward + shaping_reward
        return reward[0] if reward.shape[0] == 1 else reward

class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeats the same action for N steps and accumulates rewards.
    Helps the agent build momentum in underactuated systems.
    """
    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class AcrobotUprightStartWrapper(gym.Wrapper):
    """
    Debug wrapper that initializes the Acrobot near the upright position.
    Used to verify if the agent can learn to balance before tackling the swing-up.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            # Access DMC physics through Shimmy wrapper
            physics = self.env.unwrapped.physics
            # Joint positions: [theta1, theta2]
            # In dm_control Acrobot, upright is roughly [0, 0]
            qpos = np.zeros(2) + np.random.uniform(-0.1, 0.1, size=2)
            qvel = np.random.uniform(-0.05, 0.05, size=2)
            
            physics.data.qpos[:] = qpos
            physics.data.qvel[:] = qvel
            physics.forward()
            
            # Reconstruct the native DMC observation dictionary for Acrobot
            # Order: [sin(theta1), sin(theta1+theta2), cos(theta1), cos(theta1+theta2)]
            theta1 = qpos[0]
            theta12 = qpos[0] + qpos[1]
            obs = {
                "orientations": np.array([
                    np.sin(theta1), np.sin(theta12),
                    np.cos(theta1), np.cos(theta12)
                ], dtype=np.float32),
                "velocity": qvel.astype(np.float32)
            }
        except Exception as e:
            print(f"Warning: Failed to set upright start: {e}")
            
        return obs, info

class StateCloningWrapper(gym.Wrapper):
    """
    Exposes get_state() and set_state() for environments that support cloning.
    Specifically implemented for dm_control (physics state).
    """
    def get_state(self, **kwargs):
        try:
            # dm_control (via shimmy)
            physics = self.env.unwrapped.physics
            return physics.get_state().copy()
        except Exception:
            return None

    def set_state(self, state, **kwargs):
        if state is None: return
        try:
            physics = self.env.unwrapped.physics
            physics.set_state(state)
            physics.forward()
        except Exception:
            pass
