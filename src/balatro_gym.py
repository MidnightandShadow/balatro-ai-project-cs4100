from __future__ import annotations

import itertools
from types import MappingProxyType
from typing import Any

import gymnasium as gym

from src.common import Action, ActionType, Card, Suit, Rank
from src.constants import OBSERVABLE_HAND_SIZE, HAND_ACTIONS, DISCARD_ACTIONS
from src.game_state import GameState, INITIAL_GAME_STATE
from src.observer_manager import ObserverManager
from src.simulator import simulate_turn


class BalatroEnv(gym.Env):
    """
    Represents our Balatro GameState as an AI environment following the gymnasium Env interface.
    Assumes src.constants.OBSERVABLE_HAND_SIZE = 8.
    """

    def __init__(self, game_state: GameState, observer_manager: ObserverManager):
        self.game_state = game_state
        self.observation_state = {}
        self.observer_manager = observer_manager
        self.observation_space = gym.spaces.Dict(
            {
                "blind_chips": gym.spaces.Box(0, float("inf"), dtype=int),
                "scored_chips": gym.spaces.Box(0, float("inf"), dtype=int),
                "hand_actions": gym.spaces.Discrete(HAND_ACTIONS),
                "discard_actions": gym.spaces.Discrete(DISCARD_ACTIONS),
                "cards_left_in_deck": gym.spaces.Sequence(gym.spaces.Discrete(52)),
                "observable_hand": gym.spaces.Sequence(gym.spaces.Discrete(52)),
            }
        )

        # See description.md:
        # We have 436 actions corresponding to playing a Hand or Discard of 1 - 5 card slots of the observable hand
        self.action_space = gym.spaces.Discrete(436)
        # Immutable Dictionary mapping action space values to corresponding SlotActions (to create corresponding
        # actions for the simulator)
        self.env_actions_to_slot_actions = self._generate_action_mapping()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset to start of episode.
        :return: obs, reward, terminated, truncated, info
        """
        super().reset(seed=seed)
        self.game_state = INITIAL_GAME_STATE
        self.observation_state = self._env_observation_state_from_current_game_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, 0, False, False, info

    def step(self, action: int):
        """
        Apply an action to take one step, and return the resulting information for the new state prime.
        :return: obs, reward, terminated, truncated, info
        """
        initial_agent_score = self.game_state.scored_chips

        self.game_state = simulate_turn(
            self.game_state,
            self._env_action_to_game_action(action),
            self.observer_manager,
        )

        self.observation_state = self._env_observation_state_from_current_game_state()

        terminated = self.game_state.is_game_over()
        truncated = False
        agent_score_difference = self.game_state.scored_chips - initial_agent_score
        reward = (
            self._win_reward()
            if terminated and self.game_state.did_player_win()
            else self._lose_reward() if terminated else agent_score_difference
        )
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """
        The obs component returned by env.step() and env.reset().
        """
        return self.observation_state.copy()

    @staticmethod
    def _get_info():
        """
        Auxiliary info returned by env.step() and env.reset(). Currently, we don't have anything that needs to be here.
        """
        return {}

    def _win_reward(self) -> int:
        return self.game_state.blind_chips * 5

    def _lose_reward(self) -> int:
        return self.game_state.blind_chips * -5

    @staticmethod
    def _generate_action_mapping() -> MappingProxyType[int, SlotAction]:
        """
        Returns a mapping from the action space's int representation of actions to the corresponding
        meaning of a SlotAction.
        """
        int_to_slot_action: dict[int, SlotAction] = {}
        all_possible_slots = (0, 1, 2, 3, 4, 5, 6, 7)
        slot_combinations = (
            list(itertools.combinations(all_possible_slots, 1))
            + list(itertools.combinations(all_possible_slots, 2))
            + list(itertools.combinations(all_possible_slots, 3))
            + list(itertools.combinations(all_possible_slots, 4))
            + list(itertools.combinations(all_possible_slots, 5))
        )

        for i, slot_combination in enumerate(slot_combinations):
            int_to_slot_action[i] = SlotAction(ActionType.HAND, slot_combination)
            int_to_slot_action[i + 1] = SlotAction(ActionType.DISCARD, slot_combination)

        return MappingProxyType(int_to_slot_action)

    @staticmethod
    def _generate_card_mapping() -> MappingProxyType[int, Card]:
        """
        Returns a mapping from the action space's int representation of cards to the corresponding
        meaning of a Card.
        """
        int_to_card: dict[int, Card] = {}

        for i, suit in enumerate(Suit):
            for j, rank in enumerate(Rank):
                int_to_card[i + j] = Card(rank, suit)

        return MappingProxyType(int_to_card)

    def _env_action_to_game_action(self, action: int) -> Action:
        slot_action = self.env_actions_to_slot_actions[action]
        played_cards = list(
            map(
                lambda idx: self.game_state.observable_hand[idx],
                slot_action.played_observable_hand_slots,
            )
        )
        return Action(slot_action.action_type, played_cards)

    def _env_observation_state_from_current_game_state(self) -> dict[str, Any]:
        observable_state = self.game_state.game_state_to_observable_state()
        env_cards_to_cards = self._generate_card_mapping()
        cards_to_env_cards: MappingProxyType[Card, int] = MappingProxyType(
            {value: key for key, value in env_cards_to_cards.items()}
        )

        int_cards_left_in_deck = set(
            map(
                lambda card: cards_to_env_cards[card],
                observable_state.cards_left_in_deck,
            )
        )
        int_cards_in_observable_hand = list(
            map(lambda card: cards_to_env_cards[card], observable_state.observable_hand)
        )

        return {
            "blind_chips": observable_state.blind_chips,
            "scored_chips": observable_state.scored_chips,
            "hand_actions": observable_state.hand_actions,
            "discard_actions": observable_state.discard_actions,
            "cards_left_in_deck": int_cards_left_in_deck,
            "observable_hand": int_cards_in_observable_hand,
        }


class IllegalSlotActionException(Exception):
    def __init__(self):
        super().__init__("ILLEGAL SLOT ACTION")


class SlotAction:
    """
    Represents an action of an ActionType and a list of integers corresponding to which cards from the observable
    hand have been selected for the action.
    Example: action_type == DISCARD, payed_observable_hand_slots == [0, 1, 7] represents a discard action
             with the first, second, and last card slots of the observable hand.
    """

    def __init__(
        self, action_type: ActionType, played_observable_hand_slots: list[int]
    ):
        self.action_type = action_type
        self.played_observable_hand_slots = played_observable_hand_slots
        self._validate_action()

    def __repr__(self):
        return f"Action: {self.action_type.name}\nCard Slots Chosen: {self.played_observable_hand_slots}"

    def _validate_action(self):
        num_action_slots_out_of_valid_range = not (
            1 <= len(self.played_observable_hand_slots) <= 5
        )

        slots_in_range_of_observable_hand_size = list(
            map(
                lambda slot: 0 <= slot <= OBSERVABLE_HAND_SIZE - 1,
                self.played_observable_hand_slots,
            )
        )
        action_has_slot_not_in_range_of_observable_hand_size = any(
            slots_in_range_of_observable_hand_size
        )

        if (
            num_action_slots_out_of_valid_range
            or action_has_slot_not_in_range_of_observable_hand_size
        ):
            raise IllegalSlotActionException
