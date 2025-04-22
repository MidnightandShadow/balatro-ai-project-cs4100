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

WIN_BONUS = 600
LOSE_BONUS = -1000

def simulate_game(
    initial_state: GameState, player: Player, observer_manager: ObserverManager) -> bool:
    """
    Simulates a game to completion
    INVARIANT: the first cards for the player's observable hand have already been
    drawn from the deck and dealt
    """
    return bool(simulate_game_with_reward(initial_state, player, observer_manager))


def simulate_turn(
    state: GameState, action: Action, observer_manager: ObserverManager) -> GameState:
    """
    Simulates a single turn by applying the action to transform the state according to
    the game rules in description.md

    NOTE: Does not include terminal-state checking
    """
    return simulate_turn_with_reward(state, action, observer_manager, 0)[0]


def simulate_game_with_reward(
    initial_state: GameState, player: Player, observer_manager: ObserverManager) -> int:
    """
    Simulates a game to completion
    INVARIANT: the first cards for the player's observable hand have already been
    drawn from the deck and dealt
    """
    reward_acc = 0
    current_state = initial_state
    observer_manager.notify_observers_state(
        current_state.game_state_to_observable_state()
    )

    while not current_state.is_game_over():
        current_action = player.take_action(
            current_state.game_state_to_observable_state()
        )
        current_state, reward_acc =(
            simulate_turn_with_reward(current_state, current_action, observer_manager, reward_acc))

    player_won = current_state.did_player_win()
    observer_manager.notify_observers_game_over(player_won)
    return reward_acc + WIN_BONUS if player_won else reward_acc + LOSE_BONUS


def simulate_turn_with_reward(
    state: GameState, action: Action, observer_manager: ObserverManager, reward_acc: int) ->(
        tuple)[GameState, int]:
    """
    Simulates a single turn by applying the action to transform the state according to
    the game rules in description.md

    NOTE: Does not include terminal-state checking
    """
    _validate_action(state, action)
    new_score = state.scored_chips
    scored_hand = hand_to_scored_hand(action.played_hand)

    if action.action_type == ActionType.HAND:
        score_diff = scored_hand.score()
        new_score += score_diff
        reward_acc += score_diff

    resulting_state = state.update_state_for_turn(action.action_type, new_score, action.played_hand)

    observer_manager.notify_observers_turn(
        resulting_state.game_state_to_observable_state(), action, scored_hand
    )

    return resulting_state, reward_acc

class IllegalActionException(Exception):
    def __init__(self, message="ILLEGAL ACTION"):
        super().__init__(message)


def _validate_action(state: GameState, action: Action) -> None:
    """
    Raises an IllegalActionException if given an action that is illegal with respect
    to the state
    """
    if not _is_legal_action(state, action):
        raise IllegalActionException(f"\nILLEGAL ACTION:\nSTATE:"
                                     f"\nOBSERVABLE HAND: {state.observable_hand}"
                                     f"\nHANDS LEFT: {state.hand_actions}"
                                     f"\nDISCARDS LEFT: {state.discard_actions}"
                                     f"\nACTION: {action}\n")


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
