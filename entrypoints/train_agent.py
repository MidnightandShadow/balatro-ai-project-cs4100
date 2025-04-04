# Not sure how to get rid of this... lmk if someone finds a workaround
import sys
sys.path.extend([".", "./src"])

from src.referee import *
from src.common import *
from src.simulator import IllegalActionException
from src.env import BalatroEnv
from src.agent.policy_nn import DQNAgent

NUM_GAMES = 1_000_000
INVALID_ACTION_PUNISHMENT = -10
PRECISION = 3
DISCARD = "DISCARD"

def clamp(n, l, h):
    return max(min(n, h), l)

def freq_to_prob(freq_map):
    tot = 0
    for k in freq_map:
        tot += freq_map[k]
    return {k: round(100*freq_map[k]/tot, PRECISION) for k in freq_map}

if __name__ == "__main__":
    manager = ObserverManager()
    #manager.add_observer(PlayerObserver())
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    agent = DQNAgent(env)
    win_counter = 0
    avg_score_chips = 70        # estimate
    ALPHA = 0.001
    LOW = 0.0001
    HIGH = 0.001
    avg_reward_per_hand = 10    # estimate

    nn_discards, nn_actions = 0, 0
    nn_hand_freq = {}
    
    rand_discards, rand_actions = 0, 0
    rand_hand_freq = {}

    PRINT_FREQ = 50

    def print_if(game_num, *args, **kwargs):
        if game_num % PRINT_FREQ == 0:
            print(*args, **kwargs)
    
    for game_num in range(1,NUM_GAMES+1):
        print_if(game_num,  f"=== STARTING GAME #{game_num} {win_counter=} {agent.eps_threshold=:.3f} "
                            f"{avg_score_chips=:.3f} {avg_reward_per_hand=:.3f} ===")
        cur_state, _ = env.reset()
        done = False
        rewards = []
        nn_rewards = []
        hand_types = []
        while not done:
            k,rank = cur_state["observable_hand"]
            action = agent.get_action(cur_state)
            s = "" if agent.was_last_action_nn else "@"
            try:
                nxt_state, reward, terminated, truncated, info = env.step(action)
                rewards.append((s + str(info["previous_action"]), reward))

                # Collect "Random Agent" Statistics
                agent_was_random = not agent.was_last_action_nn
                if agent_was_random and info["previous_action"].action_type == ActionType.HAND:
                    scored_hand = hand_to_scored_hand(info["previous_action"].played_hand).poker_hand.name
                    if scored_hand not in rand_hand_freq:
                        rand_hand_freq[scored_hand] = 0
                    rand_hand_freq[scored_hand] += 1
                if agent_was_random:
                    rand_actions += 1
                if agent_was_random and info["previous_action"].action_type == ActionType.DISCARD:
                    rand_discards += 1
                """ 
                elif agent_was_random and info["previous_action"].action_type == ActionType.DISCARD:
                    hand_types.append(DISCARD)
                    if DISCARD not in rand_hand_freq:
                        rand_hand_freq[DISCARD] = 0
                    rand_hand_freq[DISCARD] += 1
                """


                # Collect "NN Agent" Statistics
                if agent.was_last_action_nn and info["previous_action"].action_type == ActionType.HAND:
                    scored_hand = hand_to_scored_hand(info["previous_action"].played_hand).poker_hand.name
                    if scored_hand not in nn_hand_freq:
                        nn_hand_freq[scored_hand] = 0
                    nn_hand_freq[scored_hand] += 1
                    hand_types.append(scored_hand)
                    avg_reward_per_hand = (1-HIGH) * avg_reward_per_hand + HIGH * reward
                """
                elif agent.was_last_action_nn and info["previous_action"].action_type == ActionType.DISCARD:
                    hand_types.append(DISCARD)
                    if DISCARD not in nn_hand_freq:
                        nn_hand_freq[DISCARD] = 0
                    nn_hand_freq[DISCARD] += 1
                """
                if agent.was_last_action_nn:
                    nn_actions += 1
                if agent.was_last_action_nn and info["previous_action"].action_type == ActionType.DISCARD:
                    nn_discards += 1
                if agent.was_last_action_nn:
                    nn_rewards.append((s + str(info["previous_action"]), reward))

                # Notify Agent of Step
                agent.update(cur_state, action, reward, terminated, nxt_state)
                cur_state = nxt_state
                done = terminated

            except IllegalActionException:
                agent.update(cur_state, action, INVALID_ACTION_PUNISHMENT, True, cur_state)
                rewards.append((s + "INVALID ACTION!", INVALID_ACTION_PUNISHMENT))
                pass

        if env.game_state.did_player_win():
            win_counter += 1

        print_if(game_num, f"{rewards = }")
        print_if(game_num, f"{nn_rewards = }")
        print_if(game_num, f"{hand_types = }")
        print_if(game_num, f"{freq_to_prob(nn_hand_freq) = }")
        print_if(game_num, f"{freq_to_prob(rand_hand_freq) = }")
        print_if(game_num, f"nn_discard_prob = {round(100*nn_discards/(nn_actions+1), PRECISION)}%")
        print_if(game_num, f"rand_discard_prob = {round(100*rand_discards/(rand_actions+1), PRECISION)}%")
        print_if(game_num)

        if avg_score_chips == 0: 
            avg_score_chips = env.game_state.scored_chips
        avg_score_chips = (1-ALPHA)*avg_score_chips + ALPHA*(env.game_state.scored_chips)
        ALPHA = clamp((env.game_state.scored_chips - avg_score_chips)/avg_score_chips, LOW, HIGH)
