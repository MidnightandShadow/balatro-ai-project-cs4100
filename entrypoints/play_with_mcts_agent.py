# Not sure how to get rid of this... lmk if someone finds a workaround
import sys

from src.agent.mcts_agent import MctsAgent

sys.path.extend([".", "./src"])

from src.referee import *
from src.env import BalatroEnv

if __name__ == "__main__":
    #play_a_single_default_game_with_a_single_strategy_and_observe_it(
    #    PrioritizeFlushSimple(), [PlayerObserver()]
    #)
    #print(f"\n{TEXT_HASH_SEPARATOR}\n{TEXT_HASH_SEPARATOR}\n")
    #play_games_with_a_single_strategy(
    #    PrioritizeFlushSimple(), games_to_play=1000, blind_chips=SMALL_BLIND_CHIPS
    #)

    NUM_GAMES = 100
    wins = 0
    running_total_scored_chips = 0
    avg_score_chips = 0

    manager = ObserverManager()
    # manager.add_observer(PlayerObserver())
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    mcts_agent = MctsAgent(env)
    
    for game_num in range(1, NUM_GAMES + 1):
        print(f"\n\nSTARTING GAME #{game_num} {wins = }  {avg_score_chips = :.3f}")
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
        avg_score_chips = running_total_scored_chips/game_num

