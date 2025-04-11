import numpy as np

from src.agent.agent import Agent
from src.env import BalatroEnv

class NNAgent(Agent):
    @staticmethod
    def convert_state_to_input(state):
        """ state is 8 + 52 + 1 + 1 + 1 """
        chips_left = state["chips_left"]
        hand_actions = state["hand_actions"]
        discard_actions = state["discard_actions"]
        deck = state["deck"]
        hand_size, rank = state["observable_hand"]
        observable_hand = BalatroEnv.unrank_combination(52, hand_size, rank)
        return np.concatenate(
            (observable_hand, deck, [hand_actions], [discard_actions], chips_left),
            dtype=np.float32
        ).reshape((1,-1))

