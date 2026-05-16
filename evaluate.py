import argparse
torch = None # Delayed import
import numpy as np
from tqdm import tqdm
import logging

from src.utils.logger import setup_logger
from src.agents import DQNAgent, SACAgent, TD3Agent, PPOAgent, GRPOAgent
from src.env import create_vector_env
from train import configs

def parse_args():
    parser = argparse.ArgumentParser(description="RL Framework Evaluation Script")
    parser.add_argument("--env", type=str, required=True, help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "SAC", "TD3", "PPO", "GRPO"], help="Algorithm name")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model file (.pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--legacy", action="store_true", help="Non-strict loading (for older models)")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("Evaluation")
    
    import torch # Import here to avoid issues with some envs
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    render_mode = None if args.no_render else "human"
    
    # Get environment specific configs
    env_config = configs.get(args.algo, {}).get(args.env, {})
    normalize_obs = env_config.get("normalize_obs", False)
    shape_reward = env_config.get("shape_reward", False)
    
    logger.info(f"Setting up evaluation for {args.env} with {args.algo}")
    # Use create_vector_env for 1 env to match training wrapper stack exactly
    env = create_vector_env(args.env, num_envs=1, render_mode=render_mode, 
                            normalize_obs=normalize_obs, shape_reward=shape_reward)
    
    # Initialize Agent
    if args.algo == "DQN":
        agent = DQNAgent(env.single_observation_space, env.single_action_space)
    elif args.algo == "SAC":
        agent = SACAgent(env.single_observation_space, env.single_action_space, **env_config)
    elif args.algo == "TD3":
        agent = TD3Agent(env.single_observation_space, env.single_action_space, **env_config)
    elif args.algo == "PPO":
        agent = PPOAgent(env.single_observation_space, env.single_action_space, **env_config)
    elif args.algo == "GRPO":
        agent = GRPOAgent(env.single_observation_space, env.single_action_space, **env_config)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not implemented")

    # Load Model
    logger.info(f"Loading weights from {args.model_path}")
    try:
        # We need to handle the state_dict directly if legacy mode is on
        if args.legacy:
            ckpt = torch.load(args.model_path, map_location=agent.device)
            agent.actor.load_state_dict(ckpt['actor'], strict=False)
            if hasattr(agent, 'critics'):
                agent.critics.load_state_dict(ckpt['critics'], strict=False)
            logger.warning("Loaded model in LEGACY mode (strict=False). Performance may be degraded.")
        else:
            agent.load(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Evaluation Loop
    total_rewards = []
    
    for ep in range(args.episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        
        episode_reward = 0
        done = False
        
        while not done:
            # Use deterministic/greedy action for evaluation
            if args.algo in ["PPO", "GRPO"]:
                action, _ = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state, evaluate=True)
            
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward[0] # VectorEnv rewards are arrays
            done = terminated[0] or truncated[0]
            
        total_rewards.append(episode_reward)
        logger.info(f"Episode {ep + 1}: Final Reward = {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    logger.info(f"Evaluation finished over {args.episodes} episodes.")
    logger.info(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
