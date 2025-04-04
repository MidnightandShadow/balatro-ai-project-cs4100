# Not sure how to get rid of this... lmk if someone finds a workaround
import sys
sys.path.extend([".", "./src"])

from src.referee import *
from src.simulator import IllegalActionException
from src.env import BalatroEnv
from src.agent.policy_nn import DQNAgent

NUM_GAMES = 100_000
INVALID_ACTION_PUNISHMENT = -1000

if __name__ == "__main__":
    manager = ObserverManager()
    #manager.add_observer(PlayerObserver())
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    agent = DQNAgent(env)
    win_counter = 0
    avg_score_chips = 0
    ALPHA = 0.001

    
    for game_num in range(1,NUM_GAMES+1):
        print(f"=== STARTING GAME #{game_num} {win_counter=} {agent.eps_threshold=:.3f} "
              f"{ALPHA=:.3f} {avg_score_chips=:.3f} ===")
        cur_state, _ = env.reset()
        done = False
        rewards = []
        while not done:
            k,rank = cur_state["observable_hand"]
            #print(list(Card.from_int(c) for c in env.unrank_combination(52,k,rank)))
            action = agent.get_action(cur_state)
            s = "" if agent.was_last_action_nn else "@"
            try:
                nxt_state, reward, terminated, truncated, info = env.step(action)
                rewards.append((s + str(info["previous_action"]), reward))
                agent.update(cur_state, action, reward, terminated, nxt_state)
                cur_state = nxt_state
                done = terminated
            except IllegalActionException:
                agent.update(cur_state, action, INVALID_ACTION_PUNISHMENT, True, cur_state)
                rewards.append((s + "INVALID ACTION!", INVALID_ACTION_PUNISHMENT))
                pass

        if env.game_state.did_player_win():
            win_counter += 1

        print(f"{rewards = }\n")

        if avg_score_chips == 0: 
            avg_score_chips = env.game_state.scored_chips
        avg_score_chips = (1-ALPHA)*avg_score_chips + ALPHA*(env.game_state.scored_chips)
        ALPHA = max(
            min((env.game_state.scored_chips - avg_score_chips)/avg_score_chips, 0.05), 
            0.001
        )
