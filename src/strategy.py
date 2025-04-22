from abc import ABC, abstractmethod
from random import sample, random
from typing import Tuple

from src.common import (
    Action,
    ActionType,
    CardAttribute,
    group_cards_by_attribute,
    Card,
    cards_by_rank_descending,
    Rank,
    ScoredHand,
    group_cards_by_attribute_and_get_four_longest_sublists_if_present,
)
from src.constants import OBSERVABLE_HAND_SIZE, CARDS_IN_HAND
from src.observable_state import ObservableState


class Strategy(ABC):
    """
    Strategy effectively represents the class of functions that take an ObservableState
    and return an Action, with each implementation semantically corresponding to some
    instance of a strategy for playing Balatro.
    """

    @abstractmethod
    def strategize(self, state: ObservableState) -> Action:
        pass

    def __repr__(self):
        return self.__class__.__name__


class RandomStrategy(Strategy):
    """Plays a Hand action with five cards randomly chosen from the observable hand"""
    def strategize(self, state: ObservableState) -> Action:
        random_hand = sample(state.observable_hand, k=5)
        return Action(ActionType.HAND, random_hand)


class PartRandomStrategy(Strategy):
    """Defers to RandomStrategy (epsilon * 100)% of the time, defers to other_strategy the rest of the time.
       Epsilon is the frequency with which to defer to the RandomStrategy as a float in: [0,1]"""
    def __init__(self, epsilon: float, other_strategy: Strategy):
        self.epsilon = epsilon
        self.other_strategy = other_strategy

    def strategize(self, state: ObservableState) -> Action:
        if random() < self.epsilon:
            return RandomStrategy().strategize(state)

        return self.other_strategy.strategize(state)


class FirstCardStrategy(Strategy):
    """Always plays just the first card"""
    def strategize(self, state: ObservableState) -> Action:
        return Action(ActionType.HAND, [state.observable_hand[0]])


class FirstFiveCardsStrategy(Strategy):
    """Always plays the first five cards in the observable_hand""" 
    def strategize(self, state: ObservableState) -> Action:
        return Action(ActionType.HAND, state.observable_hand[0:5])


class FirstFiveCardsSortedBySuitStrategy(Strategy):
    """
    Always plays the first five cards in the observable_hand, but pre-sorts the cards
    by suit first
    """ 
    def strategize(self, state: ObservableState) -> Action:
        observable_hand_by_suit = sorted(
            state.observable_hand, key=lambda card: card.suit.value
        )
        return Action(ActionType.HAND, observable_hand_by_suit[0:5])


class PrioritizeFlushSimple(Strategy):
    """
    Prioritizes flushes by doing the following:
        1. If there are enough cards for a Flush, sort the cards of that suit by rank 
           (descending) and play the highest 5.
        2. If there are not enough cards for a Flush right now, "discard" up to 5 cards
           that are not in the "most populous" suit. This "discarding" should use
           discard actions until they are exhausted, and then switch to using hand
           actions.
    """
    def strategize(self, state: ObservableState) -> Action:
        observable_hand = state.observable_hand
        we_can_play_a_flush, most_populated_suit_cards = _we_can_play_a_flush(observable_hand)

        if we_can_play_a_flush:
            most_populated_suit_cards_descending = sorted(
                most_populated_suit_cards,
                key=lambda card: card.suit.value,
                reverse=True,
            )
            return Action(ActionType.HAND, most_populated_suit_cards_descending[0:5])

        cards_in_hand_but_not_most_populated_suit = list(
            filter(
                lambda card: card not in most_populated_suit_cards,
                observable_hand,
            )
        )
        most_cards_we_can_take = min(5, len(cards_in_hand_but_not_most_populated_suit))

        if state.discard_actions > 0:
            return Action(
                ActionType.DISCARD,
                cards_in_hand_but_not_most_populated_suit[0:most_cards_we_can_take],
            )
        else:
            return Action(
                ActionType.HAND,
                cards_in_hand_but_not_most_populated_suit[0:most_cards_we_can_take],
            )


class BestHandNow(Strategy):
    """
    Plays the best hand available from the current observable hand. Does not discard.
    """
    def strategize(self, state: ObservableState) -> Action:
        observable_hand = state.observable_hand

        we_can_play_a_flush, most_populated_suit_cards = _we_can_play_a_flush(observable_hand)

        we_can_play_a_straight_flush, straight_flush_cards_desc = (_we_can_play_a_straight(most_populated_suit_cards)
                                                                   if we_can_play_a_flush else (False, []))

        if we_can_play_a_straight_flush:
            return Action(ActionType.HAND, straight_flush_cards_desc)

        (
            most_populated_rank,
            second_most_populated_rank,
            third_most_populated_rank,
            fourth_most_populated_rank,
        ) = group_cards_by_attribute_and_get_four_longest_sublists_if_present(
            observable_hand, CardAttribute.RANK
        )

        # Four of a kind
        if len(most_populated_rank) >= 4:
            return Action(ActionType.HAND, most_populated_rank[:4])

        # Full House
        if len(most_populated_rank) >= 3 and len(second_most_populated_rank) >= 2:
            return Action(ActionType.HAND, most_populated_rank[:3] + second_most_populated_rank[:2])

        if we_can_play_a_flush:
            most_populated_suit_cards_descending = sorted(
                most_populated_suit_cards,
                key=lambda card: card.suit.value,
                reverse=True,
            )
            return Action(ActionType.HAND, most_populated_suit_cards_descending[0:5])

        we_can_play_a_straight, straight_cards_desc = _we_can_play_a_straight(observable_hand)

        if we_can_play_a_straight:
            return Action(ActionType.HAND, straight_cards_desc)

        # Three of a kind
        if len(most_populated_rank) >= 3:
            return Action(ActionType.HAND, most_populated_rank[:3])

        # Two pair
        if len(most_populated_rank) >= 2 and len(second_most_populated_rank) >= 2:
            return Action(ActionType.HAND, most_populated_rank[:2] + second_most_populated_rank[:2])

        # Pair
        if len(most_populated_rank) >= 2:
            return Action(ActionType.HAND, most_populated_rank[:2])

        # High card
        return Action(ActionType.HAND,
                          [list(sorted(observable_hand, key=lambda card: card.rank.value, reverse=True))[0]])


class PreferHigherPlays(Strategy):
    """
    Plays the highest-scoring hand currently available if it is at least a two-pair or there are no discards left.
    Otherwise, it discards up to five cards not in the most populated suit.
    """
    def strategize(self, state: ObservableState) -> Action:
        hands_left, discards_left = state.hand_actions, state.discard_actions
        observable_hand = state.observable_hand

        we_can_play_a_flush, most_populated_suit_cards = _we_can_play_a_flush(observable_hand)

        if we_can_play_a_flush:
            we_can_play_a_straight_flush, straight_flush_cards_desc = _we_can_play_a_straight(most_populated_suit_cards)

            if we_can_play_a_straight_flush:
                return Action(ActionType.HAND, straight_flush_cards_desc)

            most_populated_suit_cards_descending = sorted(
                most_populated_suit_cards,
                key=lambda card: card.suit.value,
                reverse=True,
            )
            return Action(ActionType.HAND, most_populated_suit_cards_descending[0:5])

        we_can_play_a_straight, straight_cards_desc = _we_can_play_a_straight(observable_hand)

        if we_can_play_a_straight:
            return Action(ActionType.HAND, straight_cards_desc)

        (
            most_populated_rank,
            second_most_populated_rank,
            third_most_populated_rank,
            fourth_most_populated_rank,
        ) = group_cards_by_attribute_and_get_four_longest_sublists_if_present(
            observable_hand, CardAttribute.RANK
        )

        if len(most_populated_rank) >= 4:
            return Action(ActionType.HAND, most_populated_rank[:4])

        if len(most_populated_rank) >= 3 and len(second_most_populated_rank) >= 2:
            return Action(ActionType.HAND, most_populated_rank[:3] + second_most_populated_rank[:2])

        cards_not_in_most_populated_suit = [card for card in observable_hand if card not in most_populated_suit_cards]

        if len(most_populated_rank) >= 3:
            return Action(ActionType.HAND, most_populated_rank[:3])

        if len(most_populated_rank) >= 2 and len(second_most_populated_rank) >= 2:
            return Action(ActionType.HAND, most_populated_rank[:2] + second_most_populated_rank[:2])

        if discards_left > 0:
            return Action(ActionType.DISCARD, cards_not_in_most_populated_suit[:5])

        if len(most_populated_rank) >= 2:
            return Action(ActionType.HAND, most_populated_rank[:2])

        if discards_left == 0 and hands_left == 1:
            return Action(ActionType.HAND,
                          [list(sorted(observable_hand, key=lambda card: card.rank.value, reverse=True))[0]])
        else:
            return Action(ActionType.HAND, cards_not_in_most_populated_suit[:5])


def _we_can_play_a_flush(observable_hand: list[Card]) -> Tuple[bool, list[Card]]:
    """
    Returns (true if we can play a flush, list of Cards of most populous suite)
    """
    card_sublists_by_suit = group_cards_by_attribute(
        observable_hand, CardAttribute.SUIT
    )
    most_populated_suit_cards = card_sublists_by_suit[0]
    return len(most_populated_suit_cards) >= 5, most_populated_suit_cards


def _we_can_play_a_straight(hand: list[Card]) -> Tuple[bool, list[Card]]:
    """
    Returns (true if we can play a straight, list of Cards of in straight if true (otherwise empty))
    """
    rank_descending_hand = cards_by_rank_descending(hand)
    ranks: list[Rank] = list(map(lambda card: card.rank, rank_descending_hand))

    for i in range((len(hand) - CARDS_IN_HAND) + 1):
        c1v, c2v, c3v, c4v, c5v = ranks[i].value, ranks[i+1].value, ranks[i+2].value, ranks[i+3].value, ranks[i+4].value
        if c1v - c2v == c2v - c3v == c3v - c4v == c4v - c5v == 1:
            return True, rank_descending_hand[i:i+5]

    return False, []
