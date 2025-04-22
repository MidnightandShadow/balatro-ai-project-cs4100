from abc import abstractmethod, ABC

from src.common import Action, ScoredHand, PokerHand, ActionType
from src.observable_state import ObservableState
from src.view import (
    render_text_view_state,
    render_text_view_game_over,
    render_text_view_turn,
)


class Observer(ABC):
    """

    An Observer represents an object that receives relevant game updates and decides how 
    to relay that information.

    Observer is a variant of the GoF Observer pattern.

    """

    @abstractmethod
    def notify_state(self, observable_state: ObservableState):
        """
        Relays the update in the ObservableState of the game (when it has not been 
        influenced by any Action)
        """ 
        pass

    @abstractmethod
    def notify_turn(
        self, observable_state: ObservableState, action: Action, scored_hand: ScoredHand
    ):
        """
        Relays the update in the ObservableState of the game when it has changed due to
        an Action (i.e. a turn made by a player-agent).
        This also can relay other turn-related information to allow for easy tracing of
        how the resulting state was generated from the previous state.
        """
        pass

    @abstractmethod
    def notify_game_over(self, did_player_win: bool):
        pass


class PlayerObserver(Observer):
    """
    PlayerObserver relays game-related information to human players by way of rendering
    to some human-readable view.
    """

    def notify_state(self, observable_state: ObservableState):
        render_text_view_state(observable_state)

    def notify_turn(
        self, observable_state: ObservableState, action: Action, scored_hand: ScoredHand
    ):
        render_text_view_turn(observable_state, action, scored_hand)

    def notify_game_over(self, did_player_win: bool):
        render_text_view_game_over(did_player_win)


class CollectActionsTakenObserver(Observer):
    """
    Mutates the given dictionaries of poker hands to counts on every turn, logging the counts of all played and
    discarded poker hands in a simulated game.
    """
    def __init__(self, mutable_hand_action_map: dict[PokerHand, int], mutable_discard_action_map: dict[PokerHand, int]):
        self.hand_action_map = mutable_hand_action_map
        self.discard_action_map = mutable_discard_action_map

    def notify_state(self, observable_state: ObservableState):
        pass

    def notify_turn(
        self, observable_state: ObservableState, action: Action, scored_hand: ScoredHand
    ):
        matched_poker_hand = scored_hand.poker_hand
        action_type = action.action_type

        match action_type:
            case ActionType.HAND:
                self.hand_action_map[matched_poker_hand] += 1
            case ActionType.DISCARD:
                self.discard_action_map[matched_poker_hand] += 1

    def notify_game_over(self, did_player_win: bool):
        pass
