import sys

sys.path.extend([".", "./src"])

import torch.nn as nn
from torchinfo import summary

from src.referee import *
from src.common import *
from src.simulator import IllegalActionException
from src.env import BalatroEnv
from src.agent.ppo import PPOAgent

NUM_GAMES = 1_000_000
INVALID_ACTION_PUNISHMENT = -100
FREQ_MAP_MAX = 20_000
PRECISION = 3
DISCARD = "DISCARD"
POLICY_NET = None

def clamp(n, l, h):
    return max(min(n, h), l)

def freq_len(freq_map):
    tot = 0
    for k in freq_map:
        tot += freq_map[k]
    return tot

def freq_half(freq_map):
    tot = 0
    for k in freq_map:
        tot += freq_map[k]
    if FREQ_MAP_MAX < tot:
        for k in freq_map:
            freq_map[k] //= 2

def freq_to_prob(freq_map):
    tot = 0
    for k in freq_map:
        tot += freq_map[k]
    return {k: round(100*freq_map[k]/tot, PRECISION) for k in freq_map}

def main():
    global POLICY_NET

    manager = ObserverManager()
    env = BalatroEnv(INITIAL_GAME_STATE, manager)
    # in dimension is 63, out dimension is 436
    """
    LINEAR:
    agent = DQNAgent(env, lambda: nn.Sequential(
        nn.Linear(63, 436),
    ), EPS_DECAY=10**5)
    """
    agent = PPOAgent(
        env, 
        lambda: nn.Linear(63, 436),
        lambda: nn.Linear(63, 1)
    )
    summary(agent.actor, (1,63))
    summary(agent.critic, (1,1,63))
    print("\n")

    PRINT_FREQ = 1

    def print_if(game_num, *args, **kwargs):
        if game_num % PRINT_FREQ == 0:
            print(*args, **kwargs)

    nn_discards, nn_actions = 0, 0
    hand_freq = {}
    avg_reward_per_hand = 10
    avg_score_chips = 70

    TIME_STEPS = 1000
    ALPHA = 0.01
    win_counter = 0

    for game_num in range(1,TIME_STEPS+1):
        print_if(game_num,  f"=== STARTING GAME #{game_num} {win_counter=} "
                        f"{avg_score_chips=:.3f} {avg_reward_per_hand=:.3f} ===")
        done = False
        state, _ = env.reset()

        rewards = []
        nn_rewards = []
        hand_types = []

        while not done:
            (action, logprob), _ = agent.get_action_and_value(state)
            try:
                nxt_state, reward, terminated, _, info = env.step(action)
                agent.update(state, action, logprob, reward, terminated, nxt_state)

                # Collect "NN Agent" Statistics
                if info["previous_action"].action_type == ActionType.HAND:
                    scored_hand = hand_to_scored_hand(
                        info["previous_action"].played_hand
                    ).poker_hand.name
                    if scored_hand not in hand_freq:
                        hand_freq[scored_hand] = 0
                    hand_freq[scored_hand] += 1
                    hand_types.append(scored_hand)
                    avg_reward_per_hand = clamp(
                        (1-ALPHA) * avg_reward_per_hand + ALPHA * reward,
                        avg_reward_per_hand - 1,
                        avg_reward_per_hand + 1
                    )

                nn_actions += 1
                if info["previous_action"].action_type == ActionType.DISCARD:
                    nn_discards += 1
                nn_rewards.append((str(info["previous_action"]), reward))

                state = nxt_state
                done = terminated
            except IllegalActionException:
                agent.update(state, action, logprob, INVALID_ACTION_PUNISHMENT, False, state)
                break

        print_if(game_num, f"{nn_rewards = }")
        print_if(game_num, f"{hand_types = }")
        print_if(game_num, f"{freq_to_prob(hand_freq) = }")
        print_if(game_num, f"{freq_len(hand_freq) = }")
        print_if(game_num, f"nn_discard_prob = {round(100*nn_discards/(nn_actions+1), PRECISION)}%")
        print_if(game_num)


        if env.game_state.did_player_win():
            win_counter += 1

        if avg_score_chips == 0: 
            avg_score_chips = env.game_state.scored_chips
        # A maximum increase of 1 is allowed
        avg_score_chips = clamp(
            (1-ALPHA)*avg_score_chips + ALPHA*(env.game_state.scored_chips),
            avg_score_chips - 1,
            avg_score_chips + 1
        )

        agent.optimize_model()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("TODO SAVE...")

