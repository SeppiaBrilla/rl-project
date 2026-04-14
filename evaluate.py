import argparse
import torch
import numpy as np
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.agents import DQNAgent, SACAgent, TD3Agent, PPOAgent
from src.env import create_env

def parse_args():
    parser = argparse.ArgumentParser(description="RL Framework Evaluation Script")
    parser.add_argument("--env", type=str, required=True, help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "SAC", "TD3", "PPO"], help="Algorithm name")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model file (.pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("Evaluation")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    render_mode = None if args.no_render else "human"
    
    logger.info(f"Setting up evaluation for {args.env} with {args.algo}")
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

    # Load Model
    logger.info(f"Loading weights from {args.model_path}")
    try:
        agent.load(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Evaluation Loop
    total_rewards = []
    
    for ep in range(args.episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Use deterministic/greedy action for evaluation
            if args.algo == "PPO":
                action, _ = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state, evaluate=True)
            
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        logger.info(f"Episode {ep + 1}: Final Reward = {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    logger.info(f"Evaluation finished over {args.episodes} episodes.")
    logger.info(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
