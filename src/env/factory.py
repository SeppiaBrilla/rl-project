import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, MaxAndSkipObservation, NormalizeObservation, TransformObservation
from .wrappers import CarRacingActionWrapper, AcrobotRewardShapingWrapper

class PyTorchImageWrapper(gym.ObservationWrapper):
    """
    Normalizes image pixels to [0, 1] and transposes dimensions from (H, W, C) to (C, H, W).
    Gym FrameStack stacks on the first dimension for vectors but keeps it contiguous for images.
    If the input is (H, W, C), PyTorch expects (C, H, W).
    If using FrameStack, Gym outputs (4, 84, 84, 1). This wrapper squashes the last channel 
    and returns (4, 84, 84).
    """
    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape
        if len(shape) == 4 and shape[-1] == 1:
            # Output of FrameStack on Grayscale
            new_shape = (shape[0], shape[1], shape[2])
        elif len(shape) == 3:
            # Output of standard Image
            new_shape = (shape[2], shape[0], shape[1])
        else:
            new_shape = shape
            
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, obs):
        obs = np.array(obs)
        if len(obs.shape) == 4 and obs.shape[-1] == 1:
            obs = obs.squeeze(-1) # (4, 84, 84)
        elif len(obs.shape) == 3:
            obs = np.transpose(obs, (2, 0, 1)) # (C, H, W)
        return obs.astype(np.float32) / 255.0

def create_env(env_id: str, render_mode: str = None, flatten_obs: bool = True, normalize_obs: bool = False, 
               shape_reward: bool = False, use_her: bool = False, action_repeat: int = 1, 
               upright_start: bool = False, **kwargs):
    """
    Factory function to create Gymnasium environments.
    """
    if env_id.startswith("dm_control/"):
        try:
            import shimmy
        except ImportError:
            raise ImportError("shimmy is required for DMC environments.")
            
    # Create the environment
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    
    from .wrappers import ActionRepeatWrapper, AcrobotUprightStartWrapper
    
    # Debug Upright Start (Apply at the bottom so it returns DMC Dict)
    if upright_start and "acrobot" in env_id.lower():
        env = AcrobotUprightStartWrapper(env)
    
    obs_space = env.observation_space
    is_image = hasattr(obs_space, "shape") and obs_space.shape is not None and len(obs_space.shape) == 3

    # Apply normalization and shaping to the base environment
    if normalize_obs and not is_image:
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObservation(env)
        
        env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        flatten_obs = False 

    if shape_reward and "acrobot" in env_id.lower():
        # Allow shaping even with HER, but HER will still use the sparse reward for relabeling
        env = AcrobotRewardShapingWrapper(env)

    # Apply HER wrapper if requested
    if use_her:
        # Ensure it's flattened if it's a DMC dict
        if isinstance(env.observation_space, gym.spaces.Dict) and "observation" not in env.observation_space:
            env = FlattenObservation(env)
            
        from .wrappers import GoalConditionedWrapper
        env = GoalConditionedWrapper(env)
        # Specific goal for Acrobot swingup (upright position)
        if "acrobot" in env_id.lower():
            # Target upright: sin1=0, sin12=0, cos1=1, cos12=1, dots=0
            env.goal = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        flatten_obs = False 
    
    # Handle remaining non-HER flattening/image logic
    if is_image:
        env = MaxAndSkipObservation(env, 4)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        env = FrameStackObservation(env, 4)
        env = PyTorchImageWrapper(env)
        
        if "CarRacing" in env_id:
            env = CarRacingActionWrapper(env)
    elif flatten_obs:
        if isinstance(env.observation_space, (gym.spaces.Dict, gym.spaces.Tuple)):
            env = FlattenObservation(env)
        elif hasattr(env.observation_space, "shape") and len(env.observation_space.shape) > 1:
            env = FlattenObservation(env)
            
    # Action Repeat (Momentum builder - apply at the end of the stack)
    if "acrobot" in env_id.lower() and action_repeat == 1:
        action_repeat = 4
        
    if action_repeat > 1:
        env = ActionRepeatWrapper(env, action_repeat)
    
    # Enable state cloning for GRPO synchronization
    from .wrappers import StateCloningWrapper
    env = StateCloningWrapper(env)
            
    return env

def create_vector_env(env_id: str, num_envs: int = 1, render_mode: str = None, normalize_obs: bool = False, 
                      shape_reward: bool = False, use_her: bool = False, action_repeat: int = 1, 
                      upright_start: bool = False, **kwargs):
    """
    Creates a vectorized environment.
    """
    def make_env():
        return create_env(env_id, render_mode=render_mode, normalize_obs=normalize_obs, 
                          shape_reward=shape_reward, use_her=use_her, action_repeat=action_repeat,
                          upright_start=upright_start, **kwargs)
    
    if num_envs > 1:
        # Use AsyncVectorEnv for true parallelism
        env = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    else:
        # Use SyncVectorEnv for consistency even with 1 env
        env = gym.vector.SyncVectorEnv([make_env])
    
    # Store the env_id for easier discovery by agents (e.g. for HER reward function)
    env.env_id = env_id
        
    return env
