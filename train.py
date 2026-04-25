import argparse
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
# import pandas as pd (No longer required)

import src
from src.utils.logger import setup_logger
from src.utils.buffer import ReplayBuffer, RolloutBuffer
from src.agents import DQNAgent, SACAgent, TD3Agent, PPOAgent
from src.env import create_env
from src.utils.data_collector import DataCollector

configs = {
    'PPO': {
        "CarRacing-v3": {
            # Agent initialization parameters
            "lr": 1e-4,                    # Learning rate (lower for CNN stability)
            "gamma": 0.99,                 # Discount factor
            "gae_lambda": 0.95,            # GAE lambda for advantage estimation
            "clip_ratio": 0.2,             # PPO clipping parameter
            "epochs": 10,                  # This seems unused in your code - remove or clarify
            "K_epochs": 10,                # Number of optimization epochs per rollout (you use this)
            "batch_size": 64,              # Mini-batch size for updates
            "entropy_coef": 0.01,          # Entropy coefficient for exploration
            "max_grad_norm": 0.5,          # Gradient clipping
            
            # Training loop parameters
            # "num_epochs": 4000,            # Total training epochs (episodes in your loop)
            # "rollout_buffer_capacity": 512, # Smaller due to memory with images
            
            # Environment info
            # "observation_type": "image",   # Triggers NatureCNN in your Actor/Critic
            # "expected_episodes": "~4000 episodes, ~2M timesteps",
        },
        
        "dm_control/cartpole-swingup-v0": {
            # Agent initialization parameters
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "epochs": 10,                  # Unused - consider removing
            "K_epochs": 10,                # Optimization epochs per rollout
            "batch_size": 64,
            "entropy_coef": 0.001,         # Light exploration bonus
            "max_grad_norm": 0.5,
            
            # Training loop parameters
            # "num_epochs": 1000,            # Total training epochs
            # "rollout_buffer_capacity": 2048, # Full rollouts
            
            # Environment info
            # "observation_type": "vector",  # Uses Identity extractor
            # "expected_episodes": "~1000 episodes, ~500K timesteps",
        },

        "dm_control/acrobot-swingup-v0": {
            # Agent initialization parameters
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "epochs": 10,
            "K_epochs": 10,
            "batch_size": 64,
            "entropy_coef": 0.005,         # More exploration for complex dynamics
            "max_grad_norm": 0.5,

            # Training loop parameters
            # "num_epochs": 2000,
            # "rollout_buffer_capacity": 2048,

            # Environment info
            # "observation_type": "vector",
            # "expected_episodes": "~2000 episodes, ~1M timesteps",
        }
    },
    'SAC': {
        "CarRacing-v3": {
            # Agent initialization parameters
            "lr": 1e-4,                    # Learning rate
            "gamma": 0.99,                 # Discount factor
            "tau": 0.005,                  # Soft update coefficient
            "alpha": 0.2,                  # Initial entropy coefficient
            "target_entropy": None,        # Auto: -dim(action_space)

            # Training loop parameters
            # "num_epochs": 4000,            # Episodes to train
            # "buffer_capacity": 200_000,    # Smaller for image observations
            "batch_size": 256,             # Batch size for updates (you use 256 hardcoded)
            # "min_buffer_size": 256,        # Start training after this many samples

            # Environment info
            # "observation_type": "image",
            # "expected_episodes": "~4000 episodes, ~2M timesteps",

            # Note: Your code hardcodes batch_size=256 in train() - consider parameterizing
        },

        "dm_control/cartpole-swingup-v0": {
            # Agent initialization parameters
            "lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "target_entropy": None,        # Auto: -1 for 1D action space
            
            # Training loop parameters
            # "num_epochs": 600,             # Fewer episodes needed (off-policy is efficient)
            # "buffer_capacity": 1_000_000,
            # "batch_size": 256,
            # "min_buffer_size": 256,
            
            # Environment info
            # "observation_type": "vector",
            # "expected_episodes": "~600 episodes, ~300K timesteps",
        },
        
        "dm_control/acrobot-swingup-v0": {
            # Agent initialization parameters
            "lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "target_entropy": None,
            
            # Training loop parameters
            # "num_epochs": 1000,
            # "buffer_capacity": 1_000_000,
            # "batch_size": 256,
            # "min_buffer_size": 256,
            
            # Environment info
            # "observation_type": "vector",
            # "expected_episodes": "~1000 episodes, ~500K timesteps",
        }
    },
    'TD3':{
        "CarRacing-v3": {
            # Agent initialization parameters
            "lr": 1e-4,                    # Learning rate
            "gamma": 0.99,                 # Discount factor
            "tau": 0.005,                  # Soft update coefficient
            "policy_noise": 0.2,           # Noise added to target policy
            "noise_clip": 0.5,             # Clip range for target noise
            "policy_freq": 2,              # Update policy every N critic updates
            
            # Training loop parameters
            # "num_epochs": 4000,            # Episodes to train
            # "buffer_capacity": 200_000,    # You use 10_000 - increase this!
            "batch_size": 256,             # You hardcode 256 in train()
            # "min_buffer_size": 256,
            # "exploration_noise": 0.1,      # You use max_action * 0.1 in select_action
            
            # Environment info
            # "observation_type": "image",
            # "expected_episodes": "~4000 episodes, ~2M timesteps",
            
            # CRITICAL: Your buffer_capacity is only 10_000 - this is too small!
        },
        
        "dm_control/cartpole-swingup-v0": {
            # Agent initialization parameters
            "lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            
            # Training loop parameters
            # "num_epochs": 600,
            # "buffer_capacity": 1_000_000,  # Much larger than your 10_000!
            # "batch_size": 256,
            # "min_buffer_size": 256,
            # "exploration_noise": 0.1,
            
            # Environment info
            # "observation_type": "vector",
            # "expected_episodes": "~600 episodes, ~300K timesteps",
        },
        
        "dm_control/acrobot-swingup-v0": {
            # Agent initialization parameters
            "lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            
            # Training loop parameters
            # "num_epochs": 1000,
            # "buffer_capacity": 1_000_000,
            # "batch_size": 256,
            # "min_buffer_size": 256,
            # "exploration_noise": 0.1,
            
            # Environment info
            # "observation_type": "vector",
            # "expected_episodes": "~1000 episodes, ~500K timesteps",
        }
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="RL Framework Training Script")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN", "SAC", "TD3", "PPO"], help="Algorithm to train")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--render", action="store_true", help="Enable rendering for debugging/visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the model upon completion")
    parser.add_argument("--results-file", type=str, default="results.csv", help="Path to save the training results (csv)")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("Training")
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # If --render is passed, we use 'human' rendering which usually toggles display window.
    # Otherwise, render_mode=None makes the env step as fast as possible.
    render_mode = "human" if args.render else None
    
    logger.info(f"Initializing {args.env} with render_mode={render_mode}")
    env = create_env(args.env, render_mode=render_mode)
    
    # Initialize Agent
    if args.algo == "DQN":
        agent = DQNAgent(env.observation_space, env.action_space)
    elif args.algo == "SAC":
        agent = SACAgent(env.observation_space, env.action_space, **configs["SAC"][args.env])
    elif args.algo == "TD3":
        agent = TD3Agent(env.observation_space, env.action_space, **configs["TD3"][args.env])
    elif args.algo == "PPO":
        agent = PPOAgent(env.observation_space, env.action_space, **configs["PPO"][args.env])
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not implemented")
        

    agent.train(env, args.epochs, logger, args.render, results_file=args.results_file)

    if args.save_model:
        logger.info(f"Saving model to {args.save_model}")
        agent.save(args.save_model)
    env.close()

if __name__ == "__main__":
    main()
