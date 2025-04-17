from abc import ABC, abstractmethod
from src.game_state import GameState
from src.common import Action

class Reward(ABC):
    @abstractmethod
    def apply(self, prev: GameState, action: Action, cur: GameState):
        pass
