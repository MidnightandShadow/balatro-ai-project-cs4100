import random
import time
from math import sqrt, log2
from typing import Optional

from src.agent.agent import Agent
from src.env import BalatroEnv
from src.game_state import GameState
from src.player import Player


class MctsAgent(Agent):
    def __init__(self, env: BalatroEnv):
        super().__init__(env)

    def get_action(self, obs) -> int:
        pass

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        pass

    def _get_random_action(self) -> int:
        return self.env.action_space.sample()


class MCTSNode:
    """
    Represents a non-leaf node in an MCTS search tree. All fields are mutable.
    Leafs will be represented as None. Optional parent added as convenience for backprop.
    The action represents the action (in terms of a BalatroEnv's action_space) that leads
    to the current node's state.
    INVARIANT: Counts for win_count and visited_count are updated during backprop.
    INVARIANT: When creating a child node from a parent node n, parent field is initialized with n.
    """
    def __init__(self, action: int, win_count: int = 0, visited_count: int = 0, parent: Optional['MCTSNode'] = None):
        self.action = action
        self.win_count = win_count
        self.visited_count = visited_count
        self.parent = parent
        self.children: list['MCTSNode'] = []

    def add_child(self, child: 'MCTSNode'):
        self.children.append(child)


def mcts(env: BalatroEnv, player: Player, time_limit_in_seconds: int = 5):
    """
    Implements the MCTS algorithm as discussed in Russell & Norvig's AIMA 4e, pp. 207 - 210.
    The overall structure mimics their pseudocode in Figure 6.11 on page 209, but a few modifications
    were made to implement their pseudocode as real code adapted to fit our codebase.
    """
    mcts_already_sampled_actions: list[int] = []
    tree = MCTSNode(-1)  # The action that resulted in the initial state is ignored
    start_time_in_seconds = time.time()

    while _is_time_remaining(start_time_in_seconds, time_limit_in_seconds):
        selection = _select(tree)
        new_child = _expand(selection, env, mcts_already_sampled_actions)
        reward = _simulate(env, env.simulate_single_turn(new_child.action), player)
        _backprop(reward, new_child)

    children_playouts = [child.visited_count for child in tree.children]
    greatest_playout_index = children_playouts.index(max(children_playouts))

    return tree.children[greatest_playout_index].action


def _is_time_remaining(start_time_in_seconds: float, time_limit_in_seconds: int) -> bool:
    return (time.time() - start_time_in_seconds) < time_limit_in_seconds


def _select(initial_node: MCTSNode) -> MCTSNode:
    """
    INVARIANT: initial_node is not null.
    INVARIANT: since the result of UCB1 is always non-zero, will always be able to return one node.
    """
    if len(initial_node.children) == 0:
        return initial_node

    children_with_greatest_ucbs = []
    greatest_ucb_so_far = 0

    for child in initial_node.children:
        ucb = _ucb1_for_node(child)

        if ucb > greatest_ucb_so_far:
            children_with_greatest_ucbs = [child]
        elif ucb == greatest_ucb_so_far:
            children_with_greatest_ucbs.append(child)

    return _select(random.sample(children_with_greatest_ucbs, 1))


_EXPLORATION_CONSTANT = sqrt(2)


def _ucb1_for_node(node: MCTSNode, exploration_factor: float = _EXPLORATION_CONSTANT) -> float:
    parent_playouts = 0 if not node.parent else node.parent.visited_count
    return _ucb1(node.win_count, node.visited_count, parent_playouts, exploration_factor)


def _ucb1(util: int, playouts: int, parent_playouts: int, exploration_factor: float = _EXPLORATION_CONSTANT) -> float:
    return (util / playouts) + exploration_factor * sqrt(log2(parent_playouts) / playouts)


def _expand(node: MCTSNode, env: BalatroEnv, mcts_already_sampled_actions: list[int]) -> MCTSNode:
    new_node = MCTSNode(_get_new_random_action(env, mcts_already_sampled_actions), parent=node)
    node.add_child(new_node)
    return new_node


def _simulate(env: BalatroEnv, game_state: GameState, player: Player):
    return 1 if env.simulate_till_finished(game_state, player) else 0


def _backprop(reward: int, node: MCTSNode) -> None:
    """
    INVARIANT: reward is 1 if simulation resulted in win, 0 otherwise.
    """
    if node is None:
        return

    node.visited_count += reward
    node.visited_count += 1
    _backprop(reward, node.parent)


def _get_new_random_action(env: BalatroEnv, mcts_already_sampled_actions: list[int]):
    """
    INVARIANT: mcts_already_sampled_actions always refers to the same list, and mutation here affects the
               corresponding list in the caller.
    """
    new_action = env.action_space.sample()
    while new_action in mcts_already_sampled_actions:
        new_action = env.action_space.sample()

    mcts_already_sampled_actions.append(new_action)
    return new_action

