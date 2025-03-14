from src.common import Action
from src.observable_state import ObservableState
from src.strategy import Strategy


class Player:
    """
    Represents a Player (agent) of the game that has a single Strategy it defers to for
    taking actions.
    """
    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def take_action(self, state: ObservableState) -> Action:
        return self.strategy.strategize(state)
