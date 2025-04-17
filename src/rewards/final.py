from src.rewards.reward import Reward
from src.game_state import GameState
from src.common import Action

class Final(Reward):
    def apply(self, prev: GameState, action: Action, cur: GameState):
        """ 
        only final win/loss reward
        """
        return (
            # win
            cur.blind_chips * 100
            if nxt_state.is_game_over() and self.game_state.did_player_win()

            # loss
            else cur.blind_chips * -100 + 50*(cur.blind_chips - cur.scored_chips)
            if nxt_state.is_game_over()

            # otherwise
            else 0
        )

