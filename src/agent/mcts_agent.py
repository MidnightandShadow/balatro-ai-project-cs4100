import random
import time
from math import sqrt, log2, log
from typing import Optional

from src.agent.agent import Agent
from src.env import BalatroEnv
from src.game_state import GameState
from src.player import Player
from src.strategy import (
    PrioritizeFlushSimple,
    FirstFiveCardsStrategy,
    RandomStrategy,
    PartRandomStrategy,
)


class MCTSNode:
    """
    Represents a non-leaf node in an MCTS search tree. All fields are mutable.
    Leafs will be represented as None. Optional parent added as convenience for backprop.
    The action represents the action (in terms of a BalatroEnv's action_space) that leads
    to the current node's state.
    INVARIANT: Counts for win_count and visited_count are updated during backprop.
    INVARIANT: When creating a child node from a parent node n, parent field is initialized with n.
    """

    def __init__(
        self,
        action: int,
        win_count: int = 0,
        visited_count: int = 0,
        parent: Optional["MCTSNode"] = None,
    ):
        self.action = action
        self.win_count = win_count
        self.visited_count = visited_count
        self.parent = parent
        self.children: list["MCTSNode"] = []

    def add_child(self, child: "MCTSNode"):
        self.children.append(child)


def _is_time_remaining(
        start_time_in_seconds: float, time_limit_in_seconds: int
) -> bool:
    return (time.time() - start_time_in_seconds) < time_limit_in_seconds


class MctsAgent(Agent):
    def __init__(self, env: BalatroEnv):
        super().__init__(env)
        self.NUM_ITERATIONS = 2000
        self.EXPLORATION_CONSTANT = 1.52
        self.epsilon = 0
        self.playout_player = Player(PartRandomStrategy(epsilon=self.epsilon, other_strategy=PrioritizeFlushSimple()))
        self.NO_MORE_NEW_STATES_AT_DEPTH_1 = False

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

    def mcts(self, time_limit_in_seconds: int = 3) -> int:
        """
        Implements the MCTS algorithm as discussed in Russell & Norvig's AIMA 4e, pp. 207 - 210.
        The overall structure mimics their pseudocode in Figure 6.11 on page 209, but a few modifications
        were made to implement their pseudocode as real code adapted to fit our codebase.
        """
        self.NO_MORE_NEW_STATES_AT_DEPTH_1 = False
        mcts_already_sampled_actions_for_depth_1: list[int] = []
        tree = MCTSNode(-1)  # The action that resulted in the initial state is ignored
        start_time_in_seconds = time.time()
        num_iterations = 0

        # while _is_time_remaining(start_time_in_seconds, time_limit_in_seconds):
        while num_iterations < self.NUM_ITERATIONS:

            selection = self._select(tree, mcts_already_sampled_actions_for_depth_1)
            new_child = self._expand(selection, mcts_already_sampled_actions_for_depth_1)
            reward = self._simulate(new_child.action)
            self._backprop(reward, new_child)
            num_iterations += 1

        children_playouts = [child.visited_count for child in tree.children]
        greatest_playout_index = children_playouts.index(max(children_playouts))

        return tree.children[greatest_playout_index].action

    def _select(
        self,
        initial_node: MCTSNode,
        mcts_already_sampled_actions_for_depth_1: list[int],
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

        children_with_greatest_ucbs.append(initial_node)
        selected = random.sample(children_with_greatest_ucbs, 1).pop()

        # We want to explore a new leaf from the current node instead of continuing to search down
        if selected == initial_node:

            return self._select_help(
                initial_node, mcts_already_sampled_actions_for_depth_1
            )

        return self._select(selected, mcts_already_sampled_actions_for_depth_1)

    def _select_help(
        self,
        initial_node: MCTSNode,
        mcts_already_sampled_actions_for_depth_1: list[int],
    ):
        """
        INVARIANT: Assumes called from _select() and that the node with the greatest UCB was the initial_node.
        """
        # We are in the root note, and we want to ensure we don't make multiple leaves for the same actions
        if not self.NO_MORE_NEW_STATES_AT_DEPTH_1 or initial_node.parent is None:
            if self.env.get_action_space_possibly_without_discards().n == 218:
                discard_actions_used = len(
                    list(
                        filter(
                            lambda a: a >= 218, mcts_already_sampled_actions_for_depth_1
                        )
                    )
                )
                if discard_actions_used == 218 or discard_actions_used == 0:
                    self.NO_MORE_NEW_STATES_AT_DEPTH_1 = True
                    return random.sample(initial_node.children, 1).pop()

            else:  # we still have discards left, so 436 total possible actions
                if len(initial_node.children) == 436:
                    self.NO_MORE_NEW_STATES_AT_DEPTH_1 = True
                    return random.sample(initial_node.children, 1).pop()

        return initial_node

    def _ucb1_for_node(
        self, node: MCTSNode) -> float:
        parent_playouts = (
            node.visited_count if node.parent is None else node.parent.visited_count
        )
        return self._ucb1(node.win_count, node.visited_count, parent_playouts)

    def _ucb1(
        self,
        util: int,
        playouts: int,
        parent_playouts: int,
    ) -> float:
        return (util / playouts) + self.EXPLORATION_CONSTANT * sqrt(log(parent_playouts) / playouts)

    def _expand(
        self, parent: MCTSNode, mcts_already_sampled_actions: list[int]
    ) -> MCTSNode:
        successor = MCTSNode(
            self._get_new_random_action(mcts_already_sampled_actions, parent),
            parent=parent,
        )
        parent.add_child(successor)
        return successor

    def _simulate(self, action: int):
        if self.env.game_state.hand_actions == 1 and action < 218:
            return 1 if self.env.simulate_single_turn(action).did_player_win() else 0

        game_state = self.env.simulate_single_turn(action)
        return (
            1 if self.env.simulate_till_finished(game_state, self.playout_player) else 0
        )

    def _backprop(self, reward: int, node: MCTSNode) -> None:
        """
        INVARIANT: reward is 1 if simulation resulted in win, 0 otherwise.
        """
        if node is None:
            return

        node.win_count += reward
        node.visited_count += 1
        self._backprop(reward, node.parent)

    def _get_new_random_action(
        self,
        mcts_already_sampled_actions: list[int],
        parent: Optional["MCTSNode"],
    ):
        """
        Gets a "new" random action. Returns any random action from the environment's sample space if the current
        node is not at the first layer, otherwise returns a random action that has not already been chosen.
        INVARIANT: mcts_already_sampled_actions always refers to the same list, and mutation here affects the
                   corresponding list in the caller.
        """
        action_space = self.env.get_action_space_possibly_without_discards()

        if parent.parent is not None:
            return action_space.sample()

        # action_space_values = range(action_space.n)
        depth_1_actions_remaining = list(
            filter(
                lambda a: a not in mcts_already_sampled_actions,
                (list(range(action_space.n))),
            )
        )
        if len(depth_1_actions_remaining) == 0:
            new_action = random.sample(depth_1_actions_remaining, 1).pop()

        new_action = random.sample(depth_1_actions_remaining, 1).pop()

        mcts_already_sampled_actions.append(new_action)
        return new_action
