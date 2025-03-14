"""
This file contains the "Referee" of the game as a function "simulate_game()".

Given some initial GameState, a Player (agent to play the game), and an ObserverManager 
(to notify of game updates), it will simulate a game to completion roughly as according
to the Balatro rules listed in description.md.

NOTE: The current implementation partially deviates from the rules of Balatro, but not
      in any way that affects the semantics for the first ante of the game (excluding 
      boss modifiers and shops).
"""

from src.common import Action, ActionType, hand_to_scored_hand
from src.game_state import GameState
from src.observer_manager import ObserverManager
from src.player import Player


def simulate_game(
    initial_state: GameState, player: Player, observer_manager: ObserverManager
):
    """
    Simulates a game to completion
    INVARIANT: the first cards for the player's observable hand have already been 
    drawn from the deck and dealt
    """
    current_state = initial_state.copy()
    observer_manager.notify_observers_state(
        current_state.game_state_to_observable_state()
    )

    while not current_state.is_game_over():
        current_action = player.take_action(
            current_state.game_state_to_observable_state()
        )
        current_state = _simulate_turn(current_state, current_action, observer_manager)

    player_won = current_state.did_player_win()
    observer_manager.notify_observers_game_over(player_won)
    return player_won


def _simulate_turn(
    state: GameState, action: Action, observer_manager: ObserverManager
) -> GameState:
    """
    Simulates a single turn by applying the action to transform the state according to
    the game rules in description.md

    NOTE: Does not include terminal-state checking
    """
    _validate_action(state, action)
    new_score = state.scored_chips

    scored_hand = hand_to_scored_hand(action.played_hand)
    if action.action_type == ActionType.HAND:
        new_score += scored_hand.score()

    resulting_state = (
        state.update_actions_remaining(action.action_type)
        .update_score(new_score)
        .replace_played_cards(action.played_hand)
    )

    observer_manager.notify_observers_turn(
        resulting_state.game_state_to_observable_state(), action, scored_hand
    )
    return resulting_state


class IllegalActionException(Exception):
    def __init__(self):
        super().__init__("ILLEGAL ACTION")


def _validate_action(state: GameState, action: Action) -> None:
    """
    Raises an IllegalActionException if given an action that is illegal with respect
    to the state
    """
    if not _is_legal_action(state, action):
        raise IllegalActionException


def _is_legal_action(state: GameState, action: Action) -> bool:
    """
    An action is legal only when:
        1. There are enough Hands or Discards left to accommodate the ActionType
        2. The cards chosen are present in the state's observable_hand
        3. The cards chosen are between length 1 - 5 (inclusive)
    """
    enough_action_types_left = (
        state.hand_actions > 0
        if action.action_type == ActionType.HAND
        else state.discard_actions > 0
    )
    cards_chosen_are_available = all(
        card in state.observable_hand for card in action.played_hand
    )
    cards_chosen_are_valid_size_hand = 1 <= len(action.played_hand) <= 5
    return (
        enough_action_types_left
        and cards_chosen_are_available
        and cards_chosen_are_valid_size_hand
    )
