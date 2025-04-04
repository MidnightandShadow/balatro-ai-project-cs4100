import random
import time
from math import sqrt, log2, log
from typing import Optional

from src.agent.agent import Agent
from src.env import BalatroEnv
from src.game_state import GameState
from src.player import Player
from src.strategy import (
    PrioritizeFlushSimple,
    FirstFiveCardsStrategy,
    RandomStrategy,
    PartRandomStrategy,
)



class HardcodedAgent(Agent):
    def __init__(self, env: BalatroEnv):
        super().__init__(env)
        self.epsilon = 0
        self.playout_player = Player(PartRandomStrategy(epsilon=self.epsilon, other_strategy=PrioritizeFlushSimple()))

    def get_action(self, obs) -> int:
        action = self.playout_player.take_action(self.env.get_observable_state())
        return self.env.action_to_action_index(action)

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        MCTS uses online planning, so there is no need to "update" the agent: self.env will already have
        all the updated information each time the agent takes a step in the environment.
        """
        pass

