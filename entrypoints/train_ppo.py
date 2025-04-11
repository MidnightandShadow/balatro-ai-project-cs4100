import sys

sys.path.extend([".", "./src"])

import torch.nn as nn
from torchinfo import summary

from src.referee import *
from src.agent.nn.card_embedding import CardEmbedding
from src.agent.nn.positional_encoding import PositionalEncoding
from src.agent.nn.select import Select
from src.agent.nn.range import Range
from src.agent.nn.unsqueeze import Unsqueeze
from src.common import *
from src.simulator import IllegalActionException
from src.env import BalatroEnv
from src.agent.dqn import DQNAgent
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
    summary(agent.policy, (1,1,63))
    summary(agent.value, (1,1,63))
    print("\n")

    PRINT_FREQ = 1

    for game_num in range(1,NUM_GAMES+1):
        done = False
        state, _ = env.reset()
        while not done:
            (action, logprob), state_val = agent.get_action_and_value(state)
            try:
                nxt_state, reward, terminated, _, _ = env.step(action)
                agent.update(state, action, logprob, reward, terminated, nxt_state, state_val)
                state = nxt_state
                done = terminated

            except IllegalActionException:
                agent.update(state, action, logprob, INVALID_ACTION_PUNISHMENT, False, state, state_val)
                pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Saving...")
        #if POLICY_NET != None:
        #    if input("Would you like to save the NN? [Y/n] ") in ["yes", "y", "Y", ""]:
        #       torch.save(POLICY_NET.state_dict(), input("path: "))
        torch.save(POLICY_NET.state_dict(), "model.chk")

