import sys

from src.agent.hardcoded_agent import HardcodedAgent

sys.path.extend([".", "./src"])

from src.referee import *
from src.env import BalatroEnv

if __name__ == "__main__":
    NUM_GAMES = 10000
    wins = 0
    running_total_scored_chips = 0
    avg_score_chips = 0

    manager = ObserverManager()
    manager.add_observer(PlayerObserver())
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    hardcoded_agent = HardcodedAgent(env, PrioritizeFlushSimple())
    
    for game_num in range(1, NUM_GAMES + 1):
        print(f"\n\nSTARTING GAME #{game_num}, {wins = }, {avg_score_chips = :.3f}")
        cur_state, _ = env.reset()
        done = False
        while not done:
            action = hardcoded_agent.get_action(cur_state)
            nxt_state, reward, terminated, truncated, _ = env.step(action)
            cur_state = nxt_state
            done = terminated

        if env.game_state.did_player_win():
            wins += 1
        running_total_scored_chips += env.game_state.scored_chips
        avg_score_chips = running_total_scored_chips/game_num
    print(f"\n\nENDED WITH #{NUM_GAMES = }, {wins = },  {avg_score_chips = :.3f}")
