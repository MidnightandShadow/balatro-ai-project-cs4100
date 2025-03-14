from typing import Set
from src.common import *


# Represents the state of Balatro that is observable by the player agent (see: description.md)
class ObservableState:
    def __init__(
        self,
        blind_chips: int,
        scored_chips: int,
        hand_actions: int,
        discard_actions: int,
        cards_left_in_deck: Set[Card],
        observable_hand: list[Card],
    ):
        self.blind_chips = blind_chips
        self.scored_chips = scored_chips
        self.hand_actions = hand_actions
        self.discard_actions = discard_actions
        self.cards_left_in_deck = cards_left_in_deck
        self.observable_hand = observable_hand
