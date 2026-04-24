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
        agent = SACAgent(env.observation_space, env.action_space)
    elif args.algo == "TD3":
        agent = TD3Agent(env.observation_space, env.action_space)
    elif args.algo == "PPO":
        agent = PPOAgent(env.observation_space, env.action_space)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not implemented")
        

    agent.train(env, args.epochs, logger, args.render, results_file=args.results_file)

    if args.save_model:
        logger.info(f"Saving model to {args.save_model}")
        agent.save(args.save_model)
    env.close()

if __name__ == "__main__":
    main()
