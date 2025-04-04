from __future__ import annotations

from math import comb
from typing import Any

import gymnasium as gym
import numpy as np

from src.common import Action, ActionType, Card
from src.constants import HAND_ACTIONS, DISCARD_ACTIONS, NUM_CARDS
from src.constants import (
    SMALL_BLIND_CHIPS
)
from src.game_state import GameState
from src.game_state import generate_deck
from src.observer_manager import ObserverManager
from src.player import Player
from src.simulator import simulate_turn, simulate_game

MAX_CHIPS = 100_000

"""
TODO:
    - Our BalatroEnv constructor should probably take in a `RewardStrategy` class that 
      represents a strategy for rewarding an agent given a (state, action, state') triple.

"""

class BalatroEnv(gym.Env):
    """
    Represents our Balatro GameState as an AI environment following the gymnasium Env
    interface. Assumes src.constants.OBSERVABLE_HAND_SIZE = 8.
    """

    # TODO: pass in a GameStateFactory, then call factory.create to create a new 
    #       GameState.
    def __init__(self, game_state: GameState, observer_manager: ObserverManager):
        self.game_state = game_state
        self.observer_manager = observer_manager

        self.observation_space = gym.spaces.Dict(
            {
                "chips_left": gym.spaces.Box(low=0, high=MAX_CHIPS, dtype=np.int64),
                "hand_actions": gym.spaces.Discrete(HAND_ACTIONS+1),
                "discard_actions": gym.spaces.Discrete(DISCARD_ACTIONS+1),
                "deck": gym.spaces.MultiBinary(NUM_CARDS),
                "observable_hand": gym.spaces.OneOf(
                    [
                        # Generally we will have 8 cards in our observable hand, 
                        # but we could have fewer...
                        # 0 observable cards indicates a game over...
                        gym.spaces.Discrete(comb(NUM_CARDS, hand_size))
                        for hand_size in range(0, 9)
                    ]
                )
            }
        )

        # See description.md:
        # We have 436 actions corresponding to playing a Hand or Discard of 1 - 5 card
        # slots of the observable hand
        self.action_space = gym.spaces.Discrete(436)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """
        Reset to start of episode.

        :return: obs, reward, terminated, truncated, info
        """
        super().reset(seed=seed)
        self.game_state = GameState(
            blind_chips=SMALL_BLIND_CHIPS,
            scored_chips=0,
            hand_actions=HAND_ACTIONS,
            discard_actions=DISCARD_ACTIONS,
            deck=generate_deck(),
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        """
        Apply an action to take one step, and return the resulting information for the
        new state prime.

        :return: obs, reward, terminated, truncated, info
        """

        initial_scored_chips = self.game_state.scored_chips

        act = self.action_index_to_action(action)
        self.game_state = simulate_turn(
            self.game_state,
            act,
            self.observer_manager,
        )

        terminated = self.game_state.is_game_over()
        truncated = False
        agent_score_difference = self.game_state.scored_chips - initial_scored_chips
        reward = (
            self._win_reward() if terminated and self.game_state.did_player_win()
            # We want the agent to play the best it can, even if it loses
            else self._lose_reward() + agent_score_difference if terminated
            else agent_score_difference
        )
        observation = self._get_obs()
        info = {"previous_action": act}

        return observation, reward, terminated, truncated, info

    def simulate_single_turn(self, action: int):
        """
        Simulates a game from the current state for one turn based on the given action_space action,
        relying solely on the simulator. That is, this does not affect the current step of this env.
        This method is exposed for the sake of MCTS, which will need to simulate games at each step in
        the environment because MCTS plans online and relies on a simulator.

        :return: the resulting successor GameState
        """
        return simulate_turn(self.game_state, self.action_index_to_action(action), self.observer_manager)

    def simulate_till_finished(self, game_state: GameState, player: Player):
        """
        Simulates a game from the given game_state until the end of the game, relying solely on the simulator.
        That is, this does not affect the current step of this env.
        This method is exposed for the sake of MCTS, which will need to simulate games at each step in
        the environment because MCTS plans online and relies on a simulator.

        :return: true if the game resulted in a win, false otherwise
        """
        return simulate_game(game_state, player, self.observer_manager)

    def get_action_space_possibly_without_discards(self):
        """
        Balatro does not let you even try to take a discard action if you have no discards left.
        So, if this game state has no discards left, this returns an action space corresponding to only hand actions.
        Otherwise, it returns the full action space.
        This method is exposed for the sake of MCTS, which will need to simulate games at each step in
        the environment because MCTS plans online and relies on a simulator.
        INVARIANT: 218 is the right size space because of the invariant established in action_index_to_action().
        """
        if self.game_state.discard_actions == 0:
            return gym.spaces.Discrete(218)

        return self.action_space

    # https://wkerl.me/papers/algorithms2021.pdf
    @staticmethod
    def unrank_combination(n : int, k : int, index: int) -> list[int]:
        combination = []
        start = 0  # Smallest number to consider
        for i in range(k, 0, -1):
            for s in range(start, n):
                count = comb(n - s - 1, i - 1)
                if index < count:
                    combination.append(s)
                    start = s + 1
                    break
                index -= count
        return combination

    @staticmethod
    def rank_combination(n : int, k: int, combination : list[int]) -> int:
        combination = combination.copy()
        combination.sort() # remember to sort the combination before ranking
        index = 0
        start = 0  # Smallest number to consider
        for i, c in enumerate(combination):
            for j in range(start, c):
                index += comb(n - j - 1, k - (i + 1))
            start = c + 1  # Move to the next number in combination
        return index

    def _get_obs(self):
        """
        The obs component returned by env.step() and env.reset().
        """
        deck = np.array([0] * NUM_CARDS, dtype=np.int8)
        card_indices = [c.to_int() for c in self.game_state.deck]
        deck[np.array(card_indices)] = 1
        
        observable_hand = [c.to_int() for c in self.game_state.observable_hand]
        observable_hand_index = self.rank_combination(
            NUM_CARDS, len(observable_hand), observable_hand
        )

        return {
            "chips_left": np.array(
                [self.game_state.blind_chips - self.game_state.scored_chips]
            ),
            "hand_actions": np.int64(self.game_state.hand_actions),
            "discard_actions": np.int64(self.game_state.discard_actions),
            "deck": deck,
            "observable_hand": (
                np.int64(len(observable_hand)),
                np.int64(observable_hand_index)
            )
        }

    def get_observable_state(self):
        return self.game_state.game_state_to_observable_state()

    @staticmethod
    def _get_info():
        """
        Auxiliary info returned by env.step() and env.reset(). Currently, we don't have
        anything that needs to be here.
        """
        return {}

    def _win_reward(self) -> int:
        return self.game_state.blind_chips * 5

    def _lose_reward(self) -> int:
        # NOTE: A lose reward that is really negative punishes the agent too harshly for 
        #       being in a state where it might not even be possible to win from!
        #       
        #       Let's instead rely on rewards.
        return 0

    def action_index_to_action(self, action: int) -> Action:
        assert(0 <= action % 218 < 436)
        """
        Step 1. Order game_state.observable_hand
        Step 2. Determine hand action / discard action (i < 218 => hand action)
        Step 3. Determine `k` in `kC5`.
        Step 4. Index into lexicographically ordered observable hand and construct a 
                new action.

        Mapping:
            [0,8)       --> 1C8 hand action
            [8,36)      --> 2C8 hand action
            ...
            [218,226)   --> 1C8 discard action
            ...
        """ 
        # TODO: how to actions translate when the observable_hand has fewer than 
        #       8 cards??
        
        ordered_cards = self.game_state.observable_hand
        assert(8 == len(ordered_cards)) # we hard-code that there are 8 cards
        ordered_cards.sort(key=Card.to_int)
        action_type = ActionType.HAND if action < 218 else ActionType.DISCARD
        action %= 218

        k = 1
        while action >= comb(8,k):
            action -= comb(8,k)
            k += 1

        c = self.unrank_combination(8,k,action)
        cards = [ordered_cards[i] for i in c]
        return Action(action_type, cards)

    def action_to_action_index(self, action: Action) -> int:
        action_type_constant = 218 if action.action_type == ActionType.DISCARD else 0

        cards = action.played_hand
        cards.sort(key=Card.to_int)
        ordered_cards_as_int = [c.to_int() for c in cards]
        k = len(cards)

        total = action_type_constant
        for i in range(1, k):
            total += comb(8, i)

        observable_hand_cards_as_int = [c.to_int() for c in self.get_observable_state().observable_hand]
        observable_hand_cards_as_int.sort()
        ordered_cards_as_int_wrt_obs_hand_index = [observable_hand_cards_as_int.index(c) for c in ordered_cards_as_int]
        total += self.rank_combination(8, k, ordered_cards_as_int_wrt_obs_hand_index)
        return total

