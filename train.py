import argparse
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
# import pandas as pd (No longer required)

import src
from src.utils.logger import setup_logger
from src.utils.buffer import ReplayBuffer, RolloutBuffer
from src.agents import DQNAgent, SACAgent, TD3Agent, PPOAgent, GRPOAgent
from src.env import create_env
from src.utils.data_collector import DataCollector

configs = {
    'PPO': {
        "CarRacing-v3": {
            # Agent initialization parameters
            "lr": 5e-5,                    # Stable learning rate for CNN
            "gamma": 0.99,                 # Standard discount
            "gae_lambda": 0.95,            # GAE lambda for advantage estimation
            "clip_ratio": 0.2,             # PPO clipping
            "K_epochs": 10,                # 10 optimization epochs per rollout
            "batch_size": 128,             # Increased for image stability
            "entropy_coef": 0.01,          # Exploration bonus
            "max_grad_norm": 0.5,          # Gradient clipping
            
            # Environment info
            # "observation_type": "image",   # Triggers NatureCNN in your Actor/Critic
            # "expected_episodes": "~4000 episodes, ~2M timesteps",
        },
        
        "dm_control/cartpole-swingup-v0": {
            # Agent initialization parameters
            "lr": 5e-5,
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
            "lr": 5e-5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "epochs": 10,
            "K_epochs": 10,
            "batch_size": 64,
            "entropy_coef": 0.005,         # More exploration for complex dynamics
            "max_grad_norm": 0.5,
            "normalize_obs": False,
            "shape_reward": True,

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
            "lr": 1e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.1,
            "batch_size": 256,
            "min_samples": 5000,
        },

        "dm_control/cartpole-swingup-v0": {
            "lr": 5e-5,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 1.0,
            "min_samples": 5000,
            "target_entropy": None,
        },
        
        "dm_control/acrobot-swingup-v0": {
            "lr": 5e-5,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 1.0,
            "min_samples": 10000,
            "target_entropy": None,
            "normalize_obs": False,
            "shape_reward": True,
        }
    },
    'TD3':{
        "CarRacing-v3": {
            "lr": 1e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "batch_size": 256,
            "min_samples": 5000,
        },
        
        "dm_control/cartpole-swingup-v0": {
            # Agent initialization parameters
            "lr": 5e-5,
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
            "lr": 1e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "use_her": True,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "min_samples": 10000,
            "batch_size": 1024,
            "normalize_obs": True,
            "shape_reward": True,
        }
    },
    'GRPO': {
        "CarRacing-v3": {
            "lr": 5e-5,
            "gamma": 0.99,
            "eps_clip": 0.2,
            "entropy_coef": 0.01,
            "normalize_obs": True,
            "n_policies": 4,
            "n_groups": 2,
        },
        "dm_control/cartpole-swingup-v0": {
            "lr": 3e-4,
            "gamma": 0.99,
            "group_size": 16,
            "n_step": 3,
            "beta": 0.1,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
            "batch_size": 256,
            "learning_starts": 5000,
            "tau": 0.005,
        },
        "dm_control/acrobot-swingup-v0": {
            "lr": 3e-4,
            "gamma": 0.99,
            "eps_clip": 0.2,
            "entropy_coef": 0.01,
            "normalize_obs": True,
            "shape_reward": True,
            "beta": 0.05,
            "n_policies": 4,
            "n_groups": 2,
            "lambda_s": 0.01,
        }
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="RL Framework Training Script")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN", "SAC", "TD3", "PPO", "GRPO"], help="Algorithm to train")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--render", action="store_true", help="Enable rendering for debugging/visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the model upon completion")
    parser.add_argument("--results-file", type=str, default="results.csv", help="Path to save the training results (csv)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--group-size", type=int, default=8, help="Group size (M) for GRPO")
    parser.add_argument("--upright-start", action="store_true", help="Initialize Acrobot near upright (debug)")
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
    
    normalize_obs = configs.get(args.algo, {}).get(args.env, {}).get("normalize_obs", False)
    shape_reward = configs.get(args.algo, {}).get(args.env, {}).get("shape_reward", False)
    use_her = configs.get(args.algo, {}).get(args.env, {}).get("use_her", False)
    
    from src.env import create_vector_env
    env = create_vector_env(args.env, num_envs=args.n_envs, render_mode=render_mode, 
                            normalize_obs=normalize_obs, shape_reward=shape_reward, use_her=use_her,
                            upright_start=args.upright_start)
    
    # Initialize Agent using single_observation_space/single_action_space
    if args.algo == "DQN":
        agent = DQNAgent(env.single_observation_space, env.single_action_space)
    elif args.algo == "SAC":
        agent = SACAgent(env.single_observation_space, env.single_action_space, **configs["SAC"][args.env])
    elif args.algo == "TD3":
        agent = TD3Agent(env.single_observation_space, env.single_action_space, **configs["TD3"][args.env])
    elif args.algo == "PPO":
        agent = PPOAgent(env.single_observation_space, env.single_action_space, **configs["PPO"][args.env])
    elif args.algo == "GRPO":
        grpo_config = configs["GRPO"][args.env].copy()
        if args.group_size:
            grpo_config["group_size"] = args.group_size
        agent = GRPOAgent(env.single_observation_space, env.single_action_space, **grpo_config)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not implemented")
        

    agent.train(env, args.epochs, logger, args.render, results_file=args.results_file)

    if args.save_model:
        logger.info(f"Saving model to {args.save_model}")
        agent.save(args.save_model)
    env.close()

if __name__ == "__main__":
    main()
