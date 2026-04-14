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
    def update(self, *args, **kwargs):
        """
        Update the agent model/policy.
        """
        pass

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
