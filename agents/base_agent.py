from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for AI agents.
    All agents must implement these methods.
    """
    def __init__(self, name="Agent"):
        self.name = name

    @abstractmethod
    def act(self, state, valid_actions):
        """
        Choose an action given the current state.
        Args:
            state: Current game state
            valid_actions: List of valid actions
        Returns:
            action: The chosen action
        """
        pass

    @abstractmethod
    def train(self, experience):
        """
        Train the agent on an experience.
        Args:
            experience: Tuple of (state, action, reward, next_state, done)
        """
        pass

    def save(self, filepath):
        """Save agent parameters to file"""
        pass

    def load(self, filepath):
        """Load agent parameters from file"""
        pass