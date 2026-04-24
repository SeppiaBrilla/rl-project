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
        rewards = []
        for _ in range(num_episodes):
            state, info = env.reset()
            done = False
            truncated = False
            ep_reward = 0
            while not (done or truncated):
                action = self.select_action(state, evaluate=True)
                # Unpack action if it's a tuple (e.g., PPO returns action and log_prob)
                if isinstance(action, tuple):
                    action = action[0]
                state, reward, done, truncated, info = env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)
        return rewards

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
