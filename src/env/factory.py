import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation
from .wrappers import CarRacingActionWrapper

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

def create_env(env_id: str, render_mode: str = None, flatten_obs: bool = True, **kwargs):
    """
    Factory function to create Gymnasium environments, including support for 
    DeepMind Control Suite (via Shimmy) and standard Gymnasium environments.
    
    Args:
        env_id: The environment ID (e.g., 'CartPole-v1', 'CarRacing-v2', 
                'dm_control/cartpole-swingup-v0')
        render_mode: Rendering mode ('human', 'rgb_array', etc.)
        flatten_obs: Whether to flatten dict/tuple observations (default True)
        **kwargs: Additional arguments passed to gym.make
        
    Returns:
        A Gymnasium environment instance.
    """
    if env_id.startswith("dm_control/"):
        try:
            import shimmy
        except ImportError:
            raise ImportError(
                "shimmy is required for DMC environments. "
                "Install it with `pip install shimmy[dm-control]`"
            )
            
    # Create the environment
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    
    obs_space = env.observation_space
    is_image = hasattr(obs_space, "shape") and obs_space.shape is not None and len(obs_space.shape) == 3
    
    if is_image:
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        env = FrameStackObservation(env, 4)
        env = PyTorchImageWrapper(env)
        
        # Apply simplified actions for CarRacing by default
        if "CarRacing" in env_id:
            env = CarRacingActionWrapper(env)
    elif flatten_obs:
        if isinstance(env.observation_space, (gym.spaces.Dict, gym.spaces.Tuple)):
            env = FlattenObservation(env)
        elif hasattr(env.observation_space, "shape") and len(env.observation_space.shape) > 1:
            env = FlattenObservation(env)
        
    return env
