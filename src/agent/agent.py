from src.env import BalatroEnv

from abc import ABC, abstractmethod

# https://gymnasium.farama.org/introduction/train_agent/
class Agent(ABC):
    def __init__(
        self,
        env: BalatroEnv
    ):
        self.env = env

    @abstractmethod
    def get_action(self, state) -> int:
        pass

    @abstractmethod
    def update(
        self,
        cur_state, # obs type
        action: int,
        reward_f: float,
        terminated: bool,
        next_state, # obs type
    ):
        pass

