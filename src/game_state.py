from __future__ import annotations

import random

from ordered_set import OrderedSet

from src.common import *
from src.constants import (
    OBSERVABLE_HAND_SIZE,
    SMALL_BLIND_CHIPS,
    HAND_ACTIONS,
    DISCARD_ACTIONS,
)
from src.observable_state import ObservableState


class GameState:
    """
    Represents a complete state of Balatro as according to description.md
    INVARIANT: If you want the deck to be shuffled, it is the caller's responsibility
               to pre-shuffle it
    """
    def __init__(
        self,
        blind_chips: int,
        scored_chips: int,
        hand_actions: int,
        discard_actions: int,
        deck: OrderedSet[Card],
        observable_hand: list[Card] | None = None,
    ):
        self.blind_chips = blind_chips
        self.scored_chips = scored_chips
        self.hand_actions = hand_actions
        self.discard_actions = discard_actions
        self.deck = deck
        if observable_hand is None:
            self.observable_hand = _draw_cards(self.deck, OBSERVABLE_HAND_SIZE)
        else:
            self.observable_hand = observable_hand

    def replace_played_cards(self, played_cards: list[Card]) -> GameState:
        """
        Returns a new GameState with the played_cards removed from the current
        observable_hand and replaced by an equal number of cards from the deck.
        INVARIANT: there will always be enough cards in the deck to draw n cards, given
                   the rules of the game.
        """ 
        replacement_cards = _draw_cards(self.deck, len(played_cards))
        new_hand = (
            list(filter(lambda card: card not in played_cards, self.observable_hand))
            + replacement_cards
        )
        return self.update_observable_hand(new_hand)

    def is_game_over(self) -> bool:
        """The game is over when scored_chips >= blind_chips or hand_actions = 0"""
        return self.scored_chips >= self.blind_chips or self.hand_actions == 0

    def did_player_win(self) -> bool:
        """
        The player wins when the game is over, and they've scored at least the 
        required number of chips
        """
        return self.is_game_over() and self.scored_chips >= self.blind_chips

    def game_state_to_observable_state(self) -> ObservableState:
        return ObservableState(
            self.blind_chips,
            self.scored_chips,
            self.hand_actions,
            self.discard_actions,
            set(self.deck),
            self.observable_hand,
        )

    def copy(self) -> GameState:
        return GameState(
            self.blind_chips,
            self.scored_chips,
            self.hand_actions,
            self.discard_actions,
            self.deck.copy(),
            self.observable_hand.copy(),
        )

    def update_score(self, new_score: int) -> GameState:
        return GameState(
            self.blind_chips,
            new_score,
            self.hand_actions,
            self.discard_actions,
            self.deck.copy(),
            self.observable_hand.copy(),
        )

    def update_observable_hand(self, new_observable_hand: list[Card]) -> GameState:
        return GameState(
            self.blind_chips,
            self.scored_chips,
            self.hand_actions,
            self.discard_actions,
            self.deck.copy(),
            new_observable_hand.copy(),
        )

    def update_actions_remaining(self, action_type_taken: ActionType) -> GameState:
        match action_type_taken.value:
            case ActionType.HAND.value:
                return GameState(
                    self.blind_chips,
                    self.scored_chips,
                    self.hand_actions - 1,
                    self.discard_actions,
                    self.deck.copy(),
                    self.observable_hand.copy(),
                )

            case ActionType.DISCARD.value:
                return GameState(
                    self.blind_chips,
                    self.scored_chips,
                    self.hand_actions,
                    self.discard_actions - 1,
                    self.deck.copy(),
                    self.observable_hand.copy(),
                )


def generate_deck() -> OrderedSet[Card]:
    """
    Creates the card Deck, represented as a pre-shuffled OrderedSet of Cards
    The deck uses the typical 52 playing cards (based on 13 ranks and 4 suits)
    """
    initial_card_list = []
    for suit in Suit:
        for rank in Rank:
            initial_card_list.append(Card(rank, suit))

    random.shuffle(initial_card_list)  # Mutates the list

    return OrderedSet(initial_card_list)


def _draw_cards(deck: OrderedSet[Card], n: int) -> list[Card]:
    """
    Draws n cards from the front of the given deck and returns them as a list of Cards
    EFFECT: mutates deck by removing the cards drawn
    """
    result_cards_list = []
    for _ in range(n):
        result_cards_list.append(deck.pop(0))

    return result_cards_list


INITIAL_DECK = generate_deck()
INITIAL_GAME_STATE = GameState(
    blind_chips=SMALL_BLIND_CHIPS,
    scored_chips=0,
    hand_actions=HAND_ACTIONS,
    discard_actions=DISCARD_ACTIONS,
    deck=INITIAL_DECK,
)
