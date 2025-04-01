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
    def get_action(self, obs) -> int:
        pass

    @abstractmethod
    def update(
        self,
        obs, # obs type
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        pass

