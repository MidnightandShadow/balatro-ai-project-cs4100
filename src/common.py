""" 

This file contains data representations and functions common to both the GameState
(and corresponding simulator) as well as the Player (and corresponding Strategy).
This has not been added to the GameState directly to avoid exposing the GameState
to the Player.

"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType


class Suit(Enum):
    SPADES = 1
    CLUBS = 2
    HEARTS = 3
    DIAMONDS = 4


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class CardAttribute(Enum):
    RANK = 1
    SUIT = 2


class Card:
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __repr__(self):
        return f"<{self.rank.name} of {self.suit.name}>"

    def score(self) -> int:
        """
        Returns the score (i.e. chips) for a Card based purely on its Rank.
        Relies on the punning between Rank value and score value for numbered cards.
        """
        if self.rank.value < 11:
            return self.rank.value
        if self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]:
            return 10
        if self.rank is Rank.ACE:
            return 11
        raise NotImplementedError("Invalid rank value")

    def to_int(self) -> int:
        # TWO -> 0, THREE -> 1, ... ACE -> 13
        # TWO SPADES -> 0, ..., ACE DIAMONDS -> 51
        return len(Rank) * (self.suit.value - 1) + (self.rank.value - 2)


class PokerHand(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


class ScoringTable:
    """
    Represents the scoring table for Poker Hands the table is a mapping of PokerHands to
    pairs of (base chips, chip multiplier)
    """
    _scoring_table: MappingProxyType[PokerHand, tuple[int, int]] = MappingProxyType(
        {
            PokerHand.HIGH_CARD: (5, 1),
            PokerHand.PAIR: (10, 2),
            PokerHand.TWO_PAIR: (20, 2),
            PokerHand.THREE_OF_A_KIND: (30, 3),
            PokerHand.STRAIGHT: (30, 4),
            PokerHand.FLUSH: (35, 4),
            PokerHand.FULL_HOUSE: (40, 4),
            PokerHand.FOUR_OF_A_KIND: (60, 7),
            PokerHand.STRAIGHT_FLUSH: (100, 8),
        }
    )

    @classmethod
    def get_chips(cls, poker_hand: PokerHand):
        return cls._scoring_table[poker_hand][0]

    @classmethod
    def get_mult(cls, poker_hand: PokerHand):
        return cls._scoring_table[poker_hand][1]


class ActionType(Enum):
    HAND = 1
    DISCARD = 2


class Action:
    def __init__(self, action_type: ActionType, played_hand: list[Card]):
        self.action_type = action_type
        self.played_hand = played_hand

    def __repr__(self):
        return f"Action: {self.action_type.name}\nCards Chosen: {self.played_hand}"

    def copy(self) -> Action:
        return Action(self.action_type, self.played_hand.copy())

class ScoredHand:
    """

    Represents a potential hand of 1-5 cards to be scored through:
        1. The corresponding matched Poker Hand
        2. The cards of the potential played hand that are scored (i.e. part of the
           Poker Hand)
        3. The cards of the potential played hand that are not scored (i.e. not part
           of the Poker Hand)

    """
    def __init__(
        self,
        poker_hand: PokerHand,
        scored_cards: list[Card],
        unscored_cards: list[Card],
    ):
        self.poker_hand = poker_hand
        self.scored_cards = scored_cards
        self.unscored_cards = unscored_cards

    def __eq__(self, other):
        if not isinstance(other, ScoredHand):
            return False
        return (
            self.poker_hand == other.poker_hand
            and self.scored_cards == other.scored_cards
            and self.unscored_cards == other.unscored_cards
        )

    def __hash__(self):
        return hash((self.poker_hand, self.scored_cards, self.unscored_cards))

    def __repr__(self):
        return f"[Hand: {self.poker_hand}; Scored Cards: {self.scored_cards}; "\
              + "Unscored Cards: {self.unscored_cards}]"

    def score(self) -> int:
        card_score_sum = sum(list(map(lambda card: card.score(), self.scored_cards)))
        poker_hand_base_chips = ScoringTable.get_chips(self.poker_hand)
        poker_hand_base_mult = ScoringTable.get_mult(self.poker_hand)
        return (poker_hand_base_chips + card_score_sum) * poker_hand_base_mult


class InvariantViolatedException(Exception):
    def __init__(self, message):
        super().__init__(f"INVARIANT VIOLATED: {message}")


"""

Returns a ScoredHand corresponding to the given played_hand as per the rules in
description.md

Matches are made in decreasing order of required number of cards in the hand
(e.g. full-house before four-of-a-kind).

The x-pair and x-of-a-kind checks rely on 
group_cards_by_attribute_and_get_four_longest_sublists_if_present() and the invariants
it establishes when grouping cards by rank in descending order.

INVARIANT: played_hand is of length 1 - 5

"""
def hand_to_scored_hand(played_hand: list[Card]) -> ScoredHand:
    is_straight, is_flush = _is_straight(played_hand), _is_flush(played_hand)

    if is_straight and is_flush:
        return ScoredHand(PokerHand.STRAIGHT_FLUSH, played_hand, [])
    if is_flush:
        return ScoredHand(PokerHand.FLUSH, played_hand, [])
    if is_straight:
        return ScoredHand(PokerHand.STRAIGHT, played_hand, [])

    (
        most_populated_rank,
        second_most_populated_rank,
        third_most_populated_rank,
        fourth_most_populated_rank,
    ) = group_cards_by_attribute_and_get_four_longest_sublists_if_present(
        played_hand, CardAttribute.RANK
    )

    if len(most_populated_rank) == 3 and len(second_most_populated_rank) == 2:
        return ScoredHand(
            PokerHand.FULL_HOUSE, most_populated_rank + second_most_populated_rank, []
        )

    if len(most_populated_rank) == 4:
        return ScoredHand(
            PokerHand.FOUR_OF_A_KIND, most_populated_rank, second_most_populated_rank
        )

    if len(most_populated_rank) == 2 and len(second_most_populated_rank) == 2:
        return ScoredHand(
            PokerHand.TWO_PAIR,
            most_populated_rank + second_most_populated_rank,
            third_most_populated_rank,
        )

    if len(most_populated_rank) == 3:
        return ScoredHand(
            PokerHand.THREE_OF_A_KIND,
            most_populated_rank,
            second_most_populated_rank + third_most_populated_rank,
        )

    if len(most_populated_rank) == 2:
        return ScoredHand(
            PokerHand.PAIR,
            most_populated_rank,
            second_most_populated_rank
            + third_most_populated_rank
            + fourth_most_populated_rank,
        )

    high_card, other_cards = high_card_and_unscored(played_hand)
    return ScoredHand(PokerHand.HIGH_CARD, [high_card], other_cards)


def _is_straight(played_hand: list[Card]) -> bool:
    if len(played_hand) != 5:
        return False

    rank_descending_hand = cards_by_rank_descending(played_hand)
    ranks: list[Rank] = list(map(lambda card: card.rank, rank_descending_hand))

    for i in range(len(ranks) - 1):
        if ranks[i].value - ranks[i + 1].value != 1:
            return False
    return True


def _is_flush(played_hand: list[Card]) -> bool:
    if len(played_hand) != 5:
        return False

    suits: list[Suit] = list(map(lambda card: card.suit, played_hand))
    return all(x == suits[0] for x in suits)


def high_card_and_unscored(played_hand: list[Card]) -> tuple[Card, list[Card]]:
    """
    Returns the highest rank card in played_hand, list of the other cards in played_hand
    """
    rank_descending_hand = cards_by_rank_descending(played_hand)
    return rank_descending_hand[0], rank_descending_hand[1:]


def cards_by_rank_descending(cards: list[Card]) -> list[Card]:
    return sorted(cards, key=lambda card: card.rank.value, reverse=True)


"""

Given a list of cards, returns a list of sublists of cards, where:

  - the sublists are the lists of cards in the hand that match to a particular card
    attribute (e.g. rank or suit).
  - The result is in descending order by length of the sublists.
    Examples for intuition:
      - Grouping by Rank, the cards "Ten of Spades, Ten of Diamonds, Ten of Hearts, 
        Ten of Clubs, Two of Spades" would return:
        [[Ten of Spades, Ten of Diamonds, Ten of Hearts, Ten of Clubs],
         [Two of Spades]].
      - Grouping by Suit, the cards "Ten of Spades, Ten of Diamonds, Ten of Hearts,
        Ten of Clubs, Two of Spades, Six of Clubs" would return:
        [[Ten of Spades, Two of Spades], 
         [Ten of Clubs, Six of Clubs], 
         [Ten of Diamonds], 
         [Ten of Hearts]]

"""
def group_cards_by_attribute(
    played_hand: list[Card], group_by: CardAttribute
) -> list[list[Card]]:
    result: list[list[Card]] = []
    match group_by.value:
        case CardAttribute.RANK.value:
            for rank in Rank:
                result.append(
                    list(
                        filter(lambda card: card.rank.value == rank.value, played_hand)
                    )
                )

        case CardAttribute.SUIT.value:
            for suit in Suit:
                result.append(
                    list(
                        filter(lambda card: card.suit.value == suit.value, played_hand)
                    )
                )

    result_filtered_to_present_cards = list(
        filter(lambda card_list: len(card_list) > 0, result)
    )
    return sorted(
        result_filtered_to_present_cards,
        key=lambda card_list: len(card_list),
        reverse=True,
    )


def group_cards_by_attribute_and_get_four_longest_sublists_if_present(
    played_hand: list[Card], group_by: CardAttribute
) -> tuple[list[Card], list[Card], list[Card], list[Card]]:
    """
    Returns a tuple of the four longest sublists from group_cards_by_attribute() in 
    decreasing order by length.

    If the nth-longest sublist is not present, returns the empty list for that sublist.
    """
    grouped_cards = group_cards_by_attribute(played_hand, group_by)
    return (
        grouped_cards[0],
        grouped_cards[1] if len(grouped_cards) >= 2 else [],
        grouped_cards[2] if len(grouped_cards) >= 3 else [],
        grouped_cards[3] if len(grouped_cards) >= 4 else [],
    )
