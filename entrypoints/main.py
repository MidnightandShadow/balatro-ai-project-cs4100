# Not sure how to get rid of this... lmk if someone finds a workaround
import sys
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
    manager = ObserverManager()
    manager.add_observer(PlayerObserver())
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    print(env.observation_space.sample())
