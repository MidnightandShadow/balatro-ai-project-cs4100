# Not sure how to get rid of this... lmk if someone finds a workaround
import sys
sys.path.extend([".", "./src"])

import torch
import torch.nn as nn
from torchinfo import summary

from src.referee import *
from src.agent.device import device
from src.common import *
from src.agent.nn.card_embedding import CardEmbedding
from src.agent.nn.range import Range
from src.agent.nn.select import Select
from src.agent.nn.positional_encoding import PositionalEncoding
from src.simulator import IllegalActionException, simulate_turn
from src.env import BalatroEnv
from src.agent.dqn import DQNAgent
from src.rewards.chip_diff import ChipDiff

NUM_GAMES = 50_000
INVALID_ACTION_PUNISHMENT = -1_000_000
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
    env = BalatroEnv(INITIAL_GAME_STATE, manager, reward=ChipDiff())

    # source: autoencoder/decoder.py
    decoder = torch.load("models/decoder.pth", weights_only=False)
    # Since we have already trained the decoder's parameters, we don't need to retrain
    for c in decoder.parameters():
        c.requires_grad = False

    EMB_DIM = 512
    DIM_FF = 512
    agent = DQNAgent(env, lambda: nn.Sequential(
        Range(0,8),                             # (1,1,8)
        CardEmbedding(0,8,8,emb_dim=EMB_DIM),   # (1,8,EMB_DIM)

        # Transformer layers
        nn.TransformerEncoderLayer(EMB_DIM,32,dim_feedforward=DIM_FF),
        nn.TransformerEncoderLayer(EMB_DIM,32,dim_feedforward=DIM_FF),
        nn.TransformerEncoderLayer(EMB_DIM,32,dim_feedforward=DIM_FF),
        nn.TransformerEncoderLayer(EMB_DIM,32,dim_feedforward=DIM_FF),

        # (1,8,EMB_DIM)
        Select(1),                          # (1,1,EMB_DIM)
        nn.Flatten(),                       # (1,EMB_DIM)

        # At least 3 linear layers are needed to learn.
        Range(0, 436)
    ), EPS_DECAY=10**4)
    summary(agent.policy_net, (1,1,63))
    print("\n")
    win_counter = 0
    avg_score_chips = 0        # estimate
    ALPHA = 0.01
    avg_reward_per_hand = 0    # estimate

    nn_discards, nn_actions = 0, 0
    nn_hand_freq = {}
    
    rand_discards, rand_actions = 0, 0
    rand_hand_freq = {}

    hand_actions_played = 0
    best_hand_action_played = 0

    PRINT_FREQ = 1

    def print_if(game_num, *args, **kwargs):
        if game_num % PRINT_FREQ == 0:
            print(*args, **kwargs)
    
    for game_num in range(1,NUM_GAMES+1):
        print_if(game_num,  f"=== STARTING GAME #{game_num} {win_counter=} {agent.eps_threshold=:.3f} {avg_score_chips=:.3f} {avg_reward_per_hand=:.3f} ===")
        print_if(game_num, f"loss={list(round(f.item(), 3) for f in agent.loss_buffer)}")
        agent.loss_buffer = []
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


                # Collect "NN Agent" Statistics
                if agent.was_last_action_nn and info["previous_action"].action_type == ActionType.HAND:
                    scored_hand = hand_to_scored_hand(info["previous_action"].played_hand).poker_hand.name
                    if scored_hand not in nn_hand_freq:
                        nn_hand_freq[scored_hand] = 0
                    nn_hand_freq[scored_hand] += 1
                    hand_types.append(scored_hand)
                    score_delta = (cur_state["chips_left"] - nxt_state["chips_left"])[0]

                    if avg_reward_per_hand == 0:
                        avg_reward_per_hand = score_delta

                    avg_reward_per_hand = (1-ALPHA) * avg_reward_per_hand + ALPHA * score_delta

                    # Calculate the best hand action
                    c = env.unrank_combination(52,8,cur_state["observable_hand"][1])
                    best_chip_diff = 0
                    for i in range(162, 218): # 56 total actions simulated
                        cards = [Card.from_int(c[j]) for j in env.unrank_combination(8,5,i-162)]
                        best_chip_diff = max(best_chip_diff, hand_to_scored_hand(cards).score())

                    if cur_state["chips_left"] - nxt_state["chips_left"] == best_chip_diff:
                        best_hand_action_played += 1
                        s = "BEST! " + s
                    hand_actions_played += 1

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
                if agent.was_last_action_nn:
                    nn_rewards.append((s + "INVALID ACTION!", INVALID_ACTION_PUNISHMENT))
                break

        if env.game_state.did_player_win():
            win_counter += 1

        print_if(game_num, f"{rewards = }")
        print_if(game_num, f"{nn_rewards = }")
        print_if(game_num, f"{hand_types = }")
        print_if(game_num, f"{freq_to_prob(nn_hand_freq) = }")
        print_if(game_num, f"{freq_to_prob(rand_hand_freq) = }")
        print_if(game_num, f"{freq_len(nn_hand_freq) = }")
        print_if(game_num, f"{freq_len(rand_hand_freq) = }")
        print_if(game_num, f"nn_discard_prob = {round(100*nn_discards/(nn_actions+1), PRECISION)}%")
        print_if(game_num, f"rand_discard_prob = {round(100*rand_discards/(rand_actions+1), PRECISION)}%")
        print_if(game_num, f"best_hand_action_played_prob = {(100*best_hand_action_played / hand_actions_played) if hand_actions_played != 0 else 100:.2f}%")
        print_if(game_num)

        if avg_score_chips == 0: 
            avg_score_chips = env.game_state.scored_chips
        avg_score_chips = (1-ALPHA)*avg_score_chips + ALPHA*(env.game_state.scored_chips)

        freq_half(nn_hand_freq)
        freq_half(rand_hand_freq)

        POLICY_NET = agent.policy_net

        if game_num % 10 == 0:
            torch.save(POLICY_NET, "model.pth")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Saving...")
        torch.save(POLICY_NET, "model.pth")

