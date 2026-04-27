"""
Base Agent definition
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    def __init__(self, observation_space, action_space, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """
        Select an action given the current state.
        Args:
            state: current environment state
            evaluate: if True, select action greedily, otherwise explore
        Returns:
            action
        """
        pass

    @abstractmethod
    def train(self, env, num_epochs: int, logger, render: bool, results_file: str = "results.csv"):
        """
        Train the agent.
        Args:
            env: environment to train on
            num_epochs: number of training epochs
            logger: logger instance
            render: whether to render the environment
            results_file: path to save training results
        """
        pass

    def _evaluate(self, env, num_episodes=5):
        """
        Evaluate the agent in the environment.
        Args:
            env: environment to evaluate on
            num_episodes: number of episodes to run
        Returns:
            list of cumulative rewards for each episode
        """
        is_vector = hasattr(env, "num_envs")
        n_envs = env.num_envs if is_vector else 1
        
        rewards = []
        while len(rewards) < num_episodes:
            state = env.reset()
            if isinstance(state, tuple): state = state[0]
            
            done = np.zeros(n_envs, dtype=bool)
            ep_rewards = np.zeros(n_envs)
            
            # For vector env, we run until all environments in the batch are done once
            # or we just collect finished episodes.
            # However, VectorEnv auto-resets. This makes sequential evaluation tricky.
            # Best is to run the vector env and collect 'info' which contains final rewards.
            
            finished_episodes = 0
            while finished_episodes < n_envs and len(rewards) < num_episodes:
                action = self.select_action(state, evaluate=True)
                if isinstance(action, tuple): action = action[0]
                
                state, reward, dones, truncateds, infos = env.step(action)
                masks = np.logical_or(dones, truncateds)
                
                ep_rewards += reward
                if is_vector:
                    for i in range(n_envs):
                        if masks[i]:
                            rewards.append(float(ep_rewards[i]))
                            ep_rewards[i] = 0
                            finished_episodes += 1
                else:
                    # Single env (or old gym style)
                    if masks if not np.isscalar(masks) else masks:
                        # Extract scalar if it's an array
                        val = ep_rewards.item() if hasattr(ep_rewards, "item") else ep_rewards
                        rewards.append(float(val))
                        break
            
            if not is_vector:
                # If it's a single env, we only need to break once
                if len(rewards) >= num_episodes: break

        return rewards[:num_episodes]

    def save(self, filepath: str):
        """
        Save the agent parameters.
        """
        pass

    def load(self, filepath: str):
        """
        Load the agent parameters.
        """
        pass
