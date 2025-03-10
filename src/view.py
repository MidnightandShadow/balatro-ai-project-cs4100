from src.common import Action, ScoredHand, ActionType
from src.constants import *
from src.observable_state import ObservableState


# This file contains "View" functions that take in an ObservableState (and/or other relevant data)
# to render helpful game info in some form.


def render_text_view_state(state: ObservableState) -> None:
    blind_chips_label = f"Score at least: {state.blind_chips} chips"
    scored_chips_label = f"Round score: {state.scored_chips} chips"
    hand_actions_label = f"Hands: {state.hand_actions}"
    discard_actions_label = f"Discards: {state.discard_actions}"
    num_cards_left_in_deck_label = f"# of cards left in deck: {state.num_cards_left_in_deck}"
    observable_hand_label = f"Observable hand: {state.observable_hand}"
    print(blind_chips_label, scored_chips_label, hand_actions_label, discard_actions_label,
          num_cards_left_in_deck_label, observable_hand_label, sep="\n")


# Renders the state as above, but first renders the action the player took to transition to that state,
# as well as the relevant ScoredHand info.
# NOTE: either this function shouldn't take in the ScoredHand, or the representation of ScoredHand
#       should change to PlayedHand (to accommodate Discard action hand representations).
#       For now, it only displays ScoredHand info if the action is Hand.
def render_text_view_turn(state: ObservableState, action: Action, scored_hand: ScoredHand) -> None:
    print(TEXT_DASH_SEPARATOR)
    print(f"The action you took:\n{action}")

    if action.action_type == ActionType.HAND:
        print(f"The poker hand that was matched: {scored_hand.poker_hand.name}\n"
              f"The cards that were scored: {scored_hand.scored_cards}\n"
              f"The chips gained for this hand: {scored_hand.score()}")

    print(TEXT_EQUALS_SEPARATOR)
    render_text_view_state(state)


def render_text_view_game_over(player_won: bool) -> None:
    print(TEXT_EQUALS_SEPARATOR)
    if player_won:
        print("You won!\n")
    else:
        print("You lost :(\n")
