# Not sure how to get rid of this... lmk if someone finds a workaround
import sys
sys.path.extend([".", "./src"])

from gymnasium.spaces.utils import flatten

from src.referee import *
from src.common import Card
from src.simulator import IllegalActionException
from src.env import BalatroEnv
from src.agent.policy_nn import DQNAgent

NUM_GAMES = 100_000
INVALID_ACTION_PUNISHMENT = -1

if __name__ == "__main__":
    manager = ObserverManager()
    #manager.add_observer(PlayerObserver())
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    agent = DQNAgent(env)
    wins = 0
    avg_score_chips = 0
    ALPHA = 0.001

    for game_num in range(1,NUM_GAMES+1):
        print()
        print(f"STARTING GAME #{game_num} {wins = } {agent.eps_threshold=:.3f} \
{ALPHA=:.3f} {avg_score_chips = :.3f}")
        cur_state, _ = env.reset()
        done = False
        rewards = []
        while not done:
            k,rank = cur_state["observable_hand"]
            #print(list(Card.from_int(c) for c in env.unrank_combination(52,k,rank)))
            action = agent.get_action(cur_state)
            try:
                nxt_state, reward, terminated, truncated, info = env.step(action)
                rewards.append((str(info["previous_action"]), reward))
                agent.update(cur_state, action, reward, terminated, nxt_state)
                cur_state = nxt_state
                done = terminated
            except IllegalActionException:
                #print("Agent attempted invalid action -- WONT be punished...")
                agent.update(cur_state, action, INVALID_ACTION_PUNISHMENT, True, cur_state)
                pass
        if env.game_state.did_player_win():
            wins += 1
        print(f"{rewards = }")
        avg_score_chips = (1-ALPHA)*avg_score_chips + ALPHA*(env.game_state.scored_chips)
        ALPHA = max(
            min((env.game_state.scored_chips - avg_score_chips)/avg_score_chips, 0.05), 
            0.001
        )
