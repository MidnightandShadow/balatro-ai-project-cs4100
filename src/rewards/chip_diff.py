from src.rewards.reward import Reward
from src.game_state import GameState
from src.common import Action

class ChipDiff(Reward):
    def apply(self, prev: GameState, action: Action, cur: GameState):
        """ 
        score_diff
        """
        score_diff = cur.scored_chips - prev.scored_chips
        return score_diff
