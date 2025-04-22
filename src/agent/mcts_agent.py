import random
import time
from math import sqrt, log2, log
from typing import Optional


from src.agent.agent import Agent
from src.env import BalatroEnv
from src.game_state import GameState
from src.observer_manager import ObserverManager
from src.player import Player
from src.simulator import simulate_turn, simulate_game_with_reward
from src.strategy import (
    PrioritizeFlushSimple,
    FirstFiveCardsStrategy,
    RandomStrategy,
    PartRandomStrategy,
    Strategy,
)


class MCTSNode:
    """
    Represents a non-leaf node in an MCTS search tree. All fields are mutable.
    Leafs will be represented as None. Optional parent added as convenience for backprop.
    The action represents the action (in terms of a BalatroEnv's action_space) that leads
    to the current node's state.
    INVARIANT: total_reward and visited_count values are updated during backprop.
    INVARIANT: When creating a child node from a parent node n, parent field is initialized with n.
    """

    def __init__(
        self,
        game_state_from_agent_pov: GameState,
        action: int,
        total_reward: int = 0,
        visited_count: int = 0,
        parent: Optional["MCTSNode"] = None,
        reward_range: int = 0
    ):
        self.game_state_from_agent_pov = game_state_from_agent_pov
        self.action_that_led_here = action
        self.total_reward = total_reward
        self.visited_count = visited_count
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.reward_range = reward_range

    def add_child(self, child: "MCTSNode"):
        self.children.append(child)


class MctsAgent(Agent):
    def __init__(self, env: BalatroEnv, num_iters: int, exploration_constant: float, epsilon: float,
                 playout_non_random_component: Strategy, observer_manager: ObserverManager = ObserverManager()):
        super().__init__(env)
        self.NUM_ITERATIONS = num_iters
        self.EXPLORATION_CONSTANT = exploration_constant
        self.epsilon = epsilon
        self.playout_player = Player(PartRandomStrategy(self.epsilon, playout_non_random_component))
        self.observer_manager = observer_manager

    def get_action(self, obs) -> int:
        return self.mcts()

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        MCTS uses online planning, so there is no need to "update" the agent: self.env will already have
        all the updated information each time the agent takes a step in the environment.
        """
        pass

    def mcts(self) -> int:
        """
        Implements the MCTS algorithm as discussed in Russell & Norvig's AIMA 4e, pp. 207 - 210.
        The overall structure mimics their pseudocode in Figure 6.11 on page 209, but a few modifications
        were made to implement their pseudocode as real code adapted to fit our codebase.
        """
        game_state_from_agent_pov = GameState.game_state_from_observable_state(self.env.get_observable_state(),
                                                                               shuffle_deck=True)
        tree = MCTSNode(game_state_from_agent_pov, -1)  # The action that resulted in the initial state is meaningless
        num_iterations = 0

        while num_iterations < self.NUM_ITERATIONS:
            selection = self._select(tree)
            new_child = self._expand(selection)
            reward = self._simulate(new_child)
            self._backprop(reward, new_child)
            num_iterations += 1

        children_playouts = [child.visited_count for child in tree.children]
        greatest_playout_index = children_playouts.index(max(children_playouts))

        return tree.children[greatest_playout_index].action_that_led_here

    def _select(
        self,
        initial_node: MCTSNode,
    ) -> MCTSNode:
        """
        INVARIANT: initial_node is not None.
        """
        num_children = len(initial_node.children)
        if num_children == 0:
            return initial_node

        children_with_greatest_ucbs = []
        greatest_ucb_so_far = self._ucb1_for_node(initial_node)

        for child in initial_node.children:
            ucb = self._ucb1_for_node(child)

            if ucb > greatest_ucb_so_far:
                children_with_greatest_ucbs = [child]
            elif ucb == greatest_ucb_so_far:
                children_with_greatest_ucbs.append(child)

        selected = random.sample(children_with_greatest_ucbs + [initial_node], 1).pop()

        # We want to explore a new leaf from the current node instead of continuing to search down
        if selected == initial_node:
            return initial_node

        return self._select(selected)

    # def _select_help(
    #     self,
    #     initial_node: MCTSNode,
    #     mcts_already_sampled_actions_for_depth_1: list[int],
    # ):
    #     """
    #     INVARIANT: Assumes called from _select() and that the node with the greatest UCB was the initial_node.
    #     """
    #     # We are in the root note, and we want to ensure we don't make multiple leaves for the same actions
    #     if not self.NO_MORE_NEW_STATES_AT_DEPTH_1 or initial_node.parent is None:
    #         if self.env.get_action_space_possibly_without_discards().n == 218:
    #             discard_actions_used = len(
    #                 list(
    #                     filter(
    #                         lambda a: a >= 218, mcts_already_sampled_actions_for_depth_1
    #                     )
    #                 )
    #             )
    #             if discard_actions_used == 218 or discard_actions_used == 0:
    #                 self.NO_MORE_NEW_STATES_AT_DEPTH_1 = True
    #                 return random.sample(initial_node.children, 1).pop()
    #
    #         else:  # we still have discards left, so 436 total possible actions
    #             if len(initial_node.children) == 436:
    #                 self.NO_MORE_NEW_STATES_AT_DEPTH_1 = True
    #                 return random.sample(initial_node.children, 1).pop()
    #
    #     return initial_node

    # def _select_iterative(
    #     self,
    #     initial_node: MCTSNode,
    #     mcts_already_sampled_actions_for_depth_1: list[int],
    # ) -> MCTSNode:
    #     """
    #     INVARIANT: initial_node is not None.
    #     """
    #     while True:
    #         num_children = len(initial_node.children)
    #         if num_children == 0:
    #             return initial_node
    #
    #         children_with_greatest_ucbs = []
    #         greatest_ucb_so_far = self._ucb1_for_node(initial_node)
    #
    #         for child in initial_node.children:
    #             ucb = self._ucb1_for_node(child)
    #
    #             if ucb > greatest_ucb_so_far:
    #                 children_with_greatest_ucbs = [child]
    #             elif ucb == greatest_ucb_so_far:
    #                 children_with_greatest_ucbs.append(child)
    #
    #         children_with_greatest_ucbs.append(initial_node)
    #         selected = random.sample(children_with_greatest_ucbs, 1).pop()
    #
    #         # We want to explore a new leaf from the current node instead of continuing to search down
    #         if selected == initial_node:
    #             return self._select_help(initial_node, mcts_already_sampled_actions_for_depth_1)
    #
    #         initial_node = selected

    def _ucb1_for_node(
        self, node: MCTSNode) -> float:
        parent_playouts = (
            node.visited_count if node.parent is None else node.parent.visited_count
        )
        return self._ucb1(node.total_reward, node.visited_count, parent_playouts, node.reward_range)

    def _ucb1(
        self,
        util: int,
        playouts: int,
        parent_playouts: int,
        reward_range: int,
    ) -> float:
        return (util / playouts) + ((self.EXPLORATION_CONSTANT + (self.EXPLORATION_CONSTANT * reward_range))/10) * sqrt(log(parent_playouts, 10) / playouts)
        # return (util / playouts) + self.EXPLORATION_CONSTANT * sqrt(log(parent_playouts, 10) / playouts)
        # return (util / playouts) + sqrt(log(parent_playouts, 10) / playouts)


    def _expand(self, parent: MCTSNode) -> MCTSNode:

        # If we've hit a terminal state node, we should not expand further and should simulate again on that node
        if parent.game_state_from_agent_pov.hand_actions == 0:
            return parent

        parent_game_state = parent.game_state_from_agent_pov
        action_int = self.env.get_action_space_possibly_without_discards(parent_game_state).sample()
        true_successor_state = (
            simulate_turn(parent_game_state.copy(),
                          self.env.action_index_to_action(parent_game_state, action_int),
                          self.observer_manager))

        successor_state_from_agent_pov = (
            GameState.game_state_from_observable_state(true_successor_state.game_state_to_observable_state()))

        successor = MCTSNode(
            successor_state_from_agent_pov,
            action_int,
            parent=parent,
        )
        parent.add_child(successor)
        return successor

    def _simulate(self, child: MCTSNode):
        reward = simulate_game_with_reward(child.game_state_from_agent_pov.copy(), self.playout_player, self.observer_manager)

        # If we want to ignore the simulation reward system and just reward for "game won":
        # return 1 if reward > 0 else 0

        return reward

    def _backprop(self, reward: int, node: MCTSNode) -> None:
        if node is None:
            return

        node.total_reward += reward
        node.visited_count += 1
        node.reward_range = 0
        self._backprop_helper(reward, node.parent, reward, reward)

    def _backprop_helper(self, reward: int, node: MCTSNode, max_subtree_reward: int, min_subtree_reward: int) -> None:
        if node is None:
            return

        node.total_reward += reward
        node.visited_count += 1
        max_subtree_reward = max(max_subtree_reward, node.total_reward)
        min_subtree_reward = min(min_subtree_reward, node.total_reward)
        node.reward_range = max_subtree_reward - min_subtree_reward
        self._backprop_helper(reward, node.parent, max_subtree_reward, min_subtree_reward)
