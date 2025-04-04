from src.common import Card
from src.env import BalatroEnv
from src.agent.agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import numpy as np
from collections import namedtuple, deque, OrderedDict


DEBUG = False

def dprint(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        #print(pe.shape)
        #print(div_term.shape)
        #print(position.shape)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0,:x.size(1)]
        return self.dropout(x)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        """

        Each node in the first layer should correspond to the various "types of hands"
        that can be played. 
        
        We know there are 436 different kinds of hands that can be played, i.e. 
        8C1 + 8C2 + 8C3 + 8C4 ... + 8C5, but there are 

        52C1 + 52C2 + 52C3 + 52C4 + 52C5 = 2,893,163 POSSIBLE hands that can be played.

        Then, from those 2 million combinations * 2 for hand / discard action, only 
        436 of those combinations are legal to play. Furthermore, there's a specific 
        algorithm that maps the hand from the 4 million possibilities x the current 
        observable hand (there are 52C8 possible observable hands) down to the specific 
        action.

        Expecting a model-free NN (ran locally) to fit the weights for the 4M x 700M 
        mapping to the 436 dimension output action is absurd.

        """
        obsd = 8 + 52 + 1 + 1 + 1
        actd = 436
        N = 1000 # for the 52 cards in the deck
        self.layer1 = nn.Linear(obsd, N)
        self.layer2 = nn.Linear(N, N)
        self.layer3 = nn.Linear(N, actd)
        self.pe =  PositionalEncoding(18)
        self.model = nn.Sequential(OrderedDict([
            ("lay1", self.layer1),
            ("rel1", nn.ReLU()),
            ("lay2", self.layer2),
            ("rel2", nn.ReLU()),
            ("lay3", self.layer3),
            ("sof3", nn.Softmax(dim=-1)),
        ]))
        NHEADS = 18
        self.mha1 = nn.MultiheadAttention(18, NHEADS, batch_first=True)
        self.key1 = torch.randn((1,NHEADS,18))
        self.val1 = torch.randn((1,NHEADS,18))

        self.mha2 = nn.MultiheadAttention(18, NHEADS, batch_first=True)
        self.key2 = torch.randn((1,NHEADS,18))
        self.val2 = torch.randn((1,NHEADS,18))

        self.mha3 = nn.MultiheadAttention(18, NHEADS, batch_first=True)
        self.key3 = torch.randn((1,NHEADS,18))
        self.val3 = torch.randn((1,NHEADS,18))

        self.lin = nn.Linear(144, 436)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x :: (??, 107)
        # obs_hand, deck, ...
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # x :: (1,1,107)

        obs_hands = x[:,:,0:8].squeeze(1)
        rst       = x[:,:,8:]

        #print(obs_hands.shape)
        card_embeddings = torch.zeros((x.shape[0],8,18))
        #print(card_embeddings.shape)
        for i, obs_hand in enumerate(obs_hands):
            for j, card in enumerate(obs_hand):
                arr = Card.int_to_emb(int(card.item()))
                #print(arr, len(arr))
                card_embeddings[i,j] = torch.tensor(arr)
        #print("pe(ce):: ", self.pe(card_embeddings).shape)
        #print("pe :", self.pe(card_embeddings).shape)
        batch_size = x.shape[0]
        K = torch.ones((batch_size, self.key1.shape[1], self.key1.shape[2]))
        V = torch.ones((batch_size, self.val1.shape[1], self.val1.shape[2]))
        attn_output = self.mha1.forward(
            self.pe(card_embeddings), 
            K@self.key1,
            V@self.val1,
            need_weights=False
        )[0]
        attn_output = self.mha2.forward(
            attn_output, 
            K@self.key2,
            V@self.val2,
            need_weights=False
        )[0]
        attn_output = self.mha3.forward(
            attn_output,
            K@self.key3,
            V@self.val3,
            need_weights=False
        )[0]
        flat_output = nn.Flatten()(attn_output).unsqueeze(1)
        #print(flat_output.shape)
        return self.lin(flat_output)

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10**3
TAU = 0.005
LR = 1e-4

class DQNAgent(Agent):
    def __init__(
        self,
        env: BalatroEnv
    ):
        super().__init__(env)

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n
        # Get the number of state observations
        # state, info = self.env.reset()
        # n_observations = len(state)

        self.policy_net = DQN()
        self.target_net = DQN()

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.eps_threshold = 1

        self.steps_done = 0

    def convert_state_to_input(self, state):
        """ state is 8 + 52 + 1 + 1 + 1 """
        chips_left = state["chips_left"]
        hand_actions = state["hand_actions"]
        discard_actions = state["discard_actions"]
        deck = state["deck"]
        hand_size, rank = state["observable_hand"]
        observable_hand = self.env.unrank_combination(52, hand_size, rank)
        return np.concatenate(
            (observable_hand, deck, [hand_actions], [discard_actions], chips_left),
            dtype=np.float32
        ).reshape((1,-1))

    def update(
        self,
        cur_state, # obs type
        action: int,
        reward_f: float,
        terminated: bool,
        next_state, # obs type
    ):
        reward = torch.tensor([reward_f])

        # convert to input type
        cur_state = self.convert_state_to_input(cur_state)
        next_state = self.convert_state_to_input(next_state)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                cur_state, dtype=torch.float32
            ).unsqueeze(0)

        # Store the transition in memory
        self.memory.push(
            torch.tensor(cur_state, dtype=torch.float32).unsqueeze(0), 
            torch.tensor([[[action]]], dtype=torch.long),
            next_state,
            reward
        )

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)


    def get_action(self, state) -> int:
        sample = random.random()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        input = self.convert_state_to_input(state)
        if sample > self.eps_threshold:
            dprint(f"get_action: Sampling action from policy, {self.eps_threshold=:.3f}")
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(pn := self.policy_net(torch.from_numpy(input)))
                return self.policy_net(torch.from_numpy(input)).max(2).indices.view(1, 1).item()
        else:
            dprint(f"get_action: Sampling action randomly, {self.eps_threshold=:.3f}")
            return self.env.action_space.sample()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(2, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1).values.max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

