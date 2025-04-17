from src.rewards.reward import Reward
from src.game_state import GameState
from src.common import Action

class ChipsScored(Reward):
    def apply(self, prev: GameState, action: Action, cur: GameState):
        """ 
        score_diff + win/loss bonus
        """
        score_diff = cur.scored_chips - prev.scored_chips
        return score_diff + (
            # win
            cur.blind_chips * 100
            if cur.is_game_over() and cur.did_player_win()

            # loss
            else cur.blind_chips * -100
            if cur.is_game_over()

            # otherwise
            else 0
        )

