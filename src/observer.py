from src.common import Action, ScoredHand
from src.observable_state import ObservableState
from src.view import render_text_view_state, render_text_view_game_over, render_text_view_turn

# An Observer represents an object that receives relevant game updates and decides how to relay that information.
# Observer is a variant of the GoF Observer pattern.


class Observer:
    # Relays the update in the ObservableState of the game (when it has not been influenced by any Action)
    def notify_state(self, observable_state: ObservableState):
        pass

    # Relays the update in the ObservableState of the game when it has changed due to an Action (i.e. a turn made by
    # a player-agent).
    # This also can relay other turn-related information to allow for easy tracing of how the resulting state was
    # generated from the previous state.
    def notify_turn(self, observable_state: ObservableState, action: Action, scored_hand: ScoredHand):
        pass

    def notify_game_over(self, did_player_win: bool):
        pass


# PlayerObserver relays game-related information to human players by way of rendering to some human-readable view.
class PlayerObserver(Observer):
    def notify_state(self, observable_state: ObservableState):
        render_text_view_state(observable_state)

    def notify_turn(self, observable_state: ObservableState, action: Action, scored_hand: ScoredHand):
        render_text_view_turn(observable_state, action, scored_hand)

    def notify_game_over(self, player_won: bool):
        render_text_view_game_over(player_won)
