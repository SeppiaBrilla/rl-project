import argparse
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

import src
from src.utils.logger import setup_logger
from src.utils.buffer import ReplayBuffer, RolloutBuffer
from src.agents import DQNAgent, SACAgent, TD3Agent, PPOAgent
from src.env import create_env

def parse_args():
    parser = argparse.ArgumentParser(description="RL Framework Training Script")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN", "SAC", "TD3", "PPO"], help="Algorithm to train")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--render", action="store_true", help="Enable rendering for debugging/visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the model upon completion")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("Training")
    
    episode_rewards = []
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # If --render is passed, we use 'human' rendering which usually toggles display window.
    # Otherwise, render_mode=None makes the env step as fast as possible.
    render_mode = "human" if args.render else None
    
    logger.info(src.__file__)
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
        
    action_shape = env.action_space.shape if hasattr(env.action_space, "shape") else ()
    if args.algo == "PPO":
        buffer = RolloutBuffer(capacity=2048, state_shape=env.observation_space.shape, action_shape=action_shape)
    else:
        buffer = ReplayBuffer(capacity=10000, state_shape=env.observation_space.shape, action_shape=action_shape)
    
    # Training Loop
    logger.info(f"Starting training for {args.episodes} episodes...")
    
    pbar = tqdm(range(args.episodes), disable=args.render)
    for episode in pbar:
        # We pass seed only on reset if needed, but typically standard sets handles it.
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            if args.algo == "PPO":
                action, log_prob = agent.select_action(state, evaluate=False)
                value = agent.get_value(state)
            else:
                action = agent.select_action(state, evaluate=False)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            if args.algo == "PPO":
                buffer.add(state, action, reward, value, log_prob, done or truncated)
                if len(buffer) == buffer.capacity:
                    last_value = agent.get_value(next_state)
                    buffer.compute_returns_and_advantages(last_value, done or truncated)
                    agent.update(buffer)
            else:
                buffer.add(state, action, reward, next_state, done or truncated)
                if len(buffer) > 256:
                    agent.update(buffer.sample(256))
            
            state = next_state
            episode_reward += reward

        if args.render:
            logger.info(f"Episode {episode + 1} | Reward: {episode_reward}")
        else:
            pbar.set_postfix({'Reward': f"{episode_reward:.2f}"})
        episode_rewards.append(episode_reward)
    
    df = pd.DataFrame({"episode": range(len(episode_rewards)), "reward": episode_rewards})
    df.to_csv(f"results_{args.algo}_{args.env}.csv", index=False)   

    logger.info("Training finished.")

    if args.save_model:
        logger.info(f"Saving model to {args.save_model}")
        agent.save(args.save_model)
    env.close()

if __name__ == "__main__":
    main()
