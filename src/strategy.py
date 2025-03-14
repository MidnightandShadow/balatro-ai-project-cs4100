from src.common import Action, ActionType, CardAttribute, group_cards_by_attribute
from src.observable_state import ObservableState


class Strategy:
    """
    Strategy effectively represents the class of functions that take an ObservableState
    and return an Action, which each implementation semantically corresponding to some
    instance of a strategy for playing Balatro.
    """

    def strategize(self, state: ObservableState) -> Action:
        pass

    def __repr__(self):
        return self.__class__.__name__


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
        card_sublists_by_suit = group_cards_by_attribute(
            state.observable_hand, CardAttribute.SUIT
        )
        most_populated_suit_cards = card_sublists_by_suit[0]
        we_can_play_a_flush = len(most_populated_suit_cards) >= 5

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
                state.observable_hand,
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
