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
            self.observable_hand = self._draw_cards(OBSERVABLE_HAND_SIZE)
        else:
            self.observable_hand = observable_hand

    def _draw_cards(self, n: int) -> list[Card]:
        """
        Draws n cards from the front of the given deck and returns them as a list of Cards
        EFFECT: mutates deck by removing the cards drawn
        """
        result_cards_list = []
        for _ in range(n):
            result_cards_list.append(self.deck.pop(0))
    
        return result_cards_list

    def replace_played_cards(self, played_cards: list[Card]) -> list[Card]:
        """
        Returns a new list of Cards with the played_cards removed from the current
        observable_hand and replaced by an equal number of cards from the deck.
        INVARIANT: there will always be enough cards in the deck to draw n cards, given
                   the rules of the game.
        """ 
        replacement_cards = self._draw_cards(len(played_cards))
        new_hand = (
            list(filter(lambda card: card not in played_cards, self.observable_hand))
            + replacement_cards
        )
        return new_hand

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
            set(self.deck.copy()),
            self.observable_hand.copy(),
        )

    @staticmethod
    def game_state_from_observable_state(observable_state: ObservableState, shuffle_deck=False):
        new_deck = OrderedSet(random.sample(observable_state.cards_left_in_deck.copy(),
                                            len(observable_state.cards_left_in_deck)))\
            if shuffle_deck else OrderedSet(observable_state.cards_left_in_deck.copy())

        return GameState(
            observable_state.blind_chips,
            observable_state.scored_chips,
            observable_state.hand_actions,
            observable_state.discard_actions,
            new_deck,
            observable_state.observable_hand.copy()
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

    def update_state_for_turn(self, action_type_taken: ActionType, new_score: int, played_hand: list[Card]) -> GameState:

        new_observable_hand = self.replace_played_cards(played_hand)

        match action_type_taken:
            case ActionType.HAND:
                return GameState(
                    self.blind_chips,
                    new_score,
                    self.hand_actions - 1,
                    self.discard_actions,
                    self.deck.copy(),
                    new_observable_hand.copy(),
                )

            case ActionType.DISCARD:
                return GameState(
                    self.blind_chips,
                    new_score,
                    self.hand_actions,
                    self.discard_actions - 1,
                    self.deck.copy(),
                    new_observable_hand.copy(),
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


INITIAL_DECK = generate_deck()
INITIAL_GAME_STATE = GameState(
    blind_chips=SMALL_BLIND_CHIPS,
    scored_chips=0,
    hand_actions=HAND_ACTIONS,
    discard_actions=DISCARD_ACTIONS,
    deck=INITIAL_DECK,
)
