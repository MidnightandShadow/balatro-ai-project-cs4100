import multiprocessing
import sys
import time

import matplotlib.pyplot as plt

from src.agent.mcts_agent import MctsAgent

sys.path.extend([".", "./src"])

from src.referee import *
from src.env import BalatroEnv

# Note: NUM_GAMES gets multiplied by NUM_PROCESSES (e.g. 10 parallel processes * 100 NUM_GAMES = 1000 total games)
NUM_GAMES = 100
NUM_PROCESSES = 10
NUM_ITERS = 10000
EPSILON = 0.3
EXP_CONSTANT = 1.414
NON_RANDOM_STRATEGY_COMPONENT = BestHandNow()

hand_to_color = {PokerHand.STRAIGHT_FLUSH: "lightgreen", PokerHand.FOUR_OF_A_KIND: "yellow", PokerHand.FULL_HOUSE: "purple",
                           PokerHand.FLUSH: "pink", PokerHand.STRAIGHT: "orange", PokerHand.THREE_OF_A_KIND: "red",
                           PokerHand.TWO_PAIR: "blue", PokerHand.PAIR: "green", PokerHand.HIGH_CARD: "brown"}

default_hand_action_map = {PokerHand.STRAIGHT_FLUSH: 0, PokerHand.FOUR_OF_A_KIND: 0, PokerHand.FULL_HOUSE: 0,
                           PokerHand.FLUSH: 0, PokerHand.STRAIGHT: 0, PokerHand.THREE_OF_A_KIND: 0,
                           PokerHand.TWO_PAIR: 0, PokerHand.PAIR: 0, PokerHand.HIGH_CARD: 0}

def play_games_with_MCTS(args: (str, int)):
    process_name, num_games, num_iters, exploration_constant, epsilon, non_random_strategy_component = args

    wins = 0
    running_total_scored_chips = 0
    avg_score_chips = 0

    mutable_hand_action_map = default_hand_action_map.copy()
    mutable_discard_action_map = default_hand_action_map.copy()

    manager = ObserverManager()
    manager.add_observer(CollectActionsTakenObserver(mutable_hand_action_map, mutable_discard_action_map))
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    mcts_agent = MctsAgent(env, num_iters, exploration_constant, epsilon, non_random_strategy_component)

    for game_num in range(1, num_games + 1):
        print(f"\n\n{process_name}\nSTARTING GAME #{game_num}, {wins = },  {avg_score_chips = :.3f}")
        cur_state, _ = env.reset()
        done = False
        while not done:
            action = mcts_agent.get_action(cur_state)
            nxt_state, reward, terminated, truncated, _ = env.step(action)
            cur_state = nxt_state
            done = terminated

        if env.game_state.did_player_win():
            wins += 1
        running_total_scored_chips += env.game_state.scored_chips
        avg_score_chips = running_total_scored_chips / game_num

    print(f"\n\n{process_name}\nENDED WITH #{num_games = }, {wins = },  {avg_score_chips = :.3f}")
    return (wins, avg_score_chips, mutable_hand_action_map, mutable_discard_action_map,
            sum(mutable_hand_action_map.values()), sum(mutable_discard_action_map.values()))


if __name__ == "__main__":
    processes = [f"PROCESS_{x}" for x in range(1, NUM_PROCESSES + 1)]
    num_games = [NUM_GAMES for x in range(1, NUM_PROCESSES + 1)]
    num_iters_list = [NUM_ITERS for x in range(1, NUM_PROCESSES + 1)]
    exp_constant_list = [EXP_CONSTANT for x in range(1, NUM_PROCESSES + 1)]
    epsilon_list = [EPSILON for x in range(1, NUM_PROCESSES + 1)]
    non_rand_strat_component = [
        NON_RANDOM_STRATEGY_COMPONENT for x in range(1, NUM_PROCESSES + 1)
    ]

    total_hand_action_map = default_hand_action_map.copy()
    total_discard_action_map = default_hand_action_map.copy()
    total_hand_actions = 0
    total_discard_actions = 0
    set_results: list[list[tuple[int, float, dict, dict, int, int]]] = []

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        args = zip(processes, num_games, num_iters_list, exp_constant_list, epsilon_list, non_rand_strat_component)

        set_results.append(pool.map(play_games_with_MCTS, args))

        set_results_flattened: list[tuple[int, float, dict, dict, int, int]] = [x for xs in set_results for x in xs]

    wins = [x[0] for x in set_results_flattened]
    avg_score_chips = [x[1] for x in set_results_flattened]
    hand_action_maps = [x[2] for x in set_results_flattened]
    discard_action_maps = [x[3] for x in set_results_flattened]
    hand_action_count = sum([x[4] for x in set_results_flattened])
    discard_action_count = sum([x[5] for x in set_results_flattened])

    for m in hand_action_maps:
        for k in total_hand_action_map.keys():
            total_hand_action_map[k] += m[k]

    for m in discard_action_maps:
        for k in total_discard_action_map.keys():
            total_discard_action_map[k] += m[k]

    total_hand_actions += hand_action_count
    total_discard_actions += discard_action_count

    total_wins = sum(wins)
    overall_avg_chip_score = sum(avg_score_chips) / len(avg_score_chips)

    print(f"WINS/TOTAL GAMES: {total_wins}/{NUM_GAMES * NUM_PROCESSES}")
    print(f"OVERALL AVERAGE CHIP SCORE PER GAME: {overall_avg_chip_score}")

    hand_actions_probabilities = []

    for k, v in total_hand_action_map.items():
        freq = (v / total_hand_actions) * 100
        print(f"HAND {k} frequency: {freq:.3f}%")

    for k, v in total_discard_action_map.items():
        if total_discard_actions > 0:
            freq = (v / total_discard_actions) * 100
            print(f"DISCARD {k} frequency: {freq:.3f}%")
        else:
            print(f"DISCARD {k} frequency: {0}%")


    timestamp = time.strftime("%Y%m%d-%H%M%S")

    mcts_hand_freq_plt, mcts_hand_freq_ax = plt.subplots()
    mcts_hand_freq_ax.set_title(f"MCTS Average Played Hand Frequencies Over {NUM_GAMES*NUM_PROCESSES} Games")
    mcts_hand_freq_plt.suptitle(f"ITERS: {NUM_ITERS} -- EXP C: {EXP_CONSTANT} -- ε: {EPSILON} "
                                f"-- NON-RAND-STRAT: {NON_RANDOM_STRATEGY_COMPONENT}", fontsize=8)

    for k, v in total_hand_action_map.items():
        freq = (v / total_hand_actions) * 100
        mcts_hand_freq_ax.axhline(y=freq, linestyle="--", color=hand_to_color[k], label=k.name)

    mcts_hand_freq_ax.get_xaxis().set_visible(False)
    mcts_hand_freq_ax.set_ylabel("Hand Frequency (%)")
    mcts_hand_freq_ax.set_yticks([0, 20, 40, 60, 80, 100])
    mcts_hand_freq_plt.legend(bbox_to_anchor=(1, .9))

    mcts_hand_freq_plt.savefig(f"../temp_ib/staging/mcts_hand_freq_{timestamp}.png")


    mcts_discard_freq_plt, mcts_discard_freq_ax = plt.subplots()
    mcts_discard_freq_ax.set_title(f"MCTS Average Discard Frequencies Over {NUM_GAMES*NUM_PROCESSES} Games")
    mcts_discard_freq_plt.suptitle(f"ITERS: {NUM_ITERS} -- EXP C: {EXP_CONSTANT} -- ε: {EPSILON} "
                                f"-- NON-RAND-STRAT: {NON_RANDOM_STRATEGY_COMPONENT}", fontsize=8)

    for k, v in total_discard_action_map.items():
        freq = (v / total_discard_actions) * 100 if total_discard_actions > 0 else 0
        mcts_discard_freq_ax.axhline(y=freq, linestyle="--", color=hand_to_color[k], label=k.name)

    mcts_discard_freq_ax.get_xaxis().set_visible(False)
    mcts_discard_freq_ax.set_ylabel("Hand Frequency (%)")
    mcts_discard_freq_ax.set_yticks([0, 20, 40, 60, 80, 100])
    mcts_discard_freq_plt.legend(bbox_to_anchor=(1, .9))

    mcts_discard_freq_plt.savefig(f"../temp_ib/staging/mcts_discard_freq_{timestamp}.png")
