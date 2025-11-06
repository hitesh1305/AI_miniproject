# environments/base_env.py
from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Standardized environment interface:
    - reset() -> initial_state
    - step(action) -> (next_state, reward, done, info)
    - get_valid_actions() -> list
    - render() for debug or headless printouts
    """
    def __init__(self):
        self.state = None
        self.done = False
        self.winner = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_valid_actions(self):
        pass

    @abstractmethod
    def render(self):
        pass

    def is_terminal(self):
        return self.done
