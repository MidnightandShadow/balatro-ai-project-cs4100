from src.common import Card
from src.env import BalatroEnv
from src.agent.agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim

import math
import random
import itertools
import gc
import numpy as np
from collections import namedtuple, deque, OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        """

        DQNv1 was a success! Although, it's troubling me that it discards so infrequently.
        I think the reward function is foolproof -- I think maybe the bigger issue here is that the 
        hand, discard, and deck information need to be passed into the beginning of the NN in order to make 
        an informed decision about which of the 436 actions to pick.

        """
        super(DQN, self).__init__()

        self.NHEADS = 18
        self.EMBDIM = 450
        self.NLAYERS = 4

        self.pe =  PositionalEncoding(self.EMBDIM).to(device)
        self.mha = nn.ParameterList(
            nn.MultiheadAttention(self.EMBDIM, self.NHEADS, batch_first=True).to(device)
            for _ in range(self.NLAYERS)
        )
        self.key = nn.ParameterList(
            torch.zeros((1,self.NHEADS,self.EMBDIM)).to(device) for _ in range(self.NLAYERS)
        )
        self.val = nn.ParameterList(
            torch.zeros((1,self.NHEADS,self.EMBDIM)).to(device) for _ in range(self.NLAYERS)
        )
        self.lin = nn.ParameterList(
            nn.Linear(self.EMBDIM,self.EMBDIM).to(device) for _ in range(self.NLAYERS)
        )
        self.batch_norm = nn.ParameterList(
            nn.BatchNorm1d(num_features=63).to(device) for _ in range(self.NLAYERS)
        )
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU().to(device)

        self.flatten = nn.Flatten().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        x.to(device)
        batch_size = x.shape[0]

        # x        :: (B,1,63)
        # obs_hand :: (B,8)
        # decks    :: (B,52)
        # rest     :: (B,3)
        obs_hands = x[:,:,0:8].squeeze(1).to(device)
        decks     = x[:,:,8:60].squeeze(1).to(device)
        rest      = x[:,:,60:].squeeze(1).to(device)

        """ EMBEDDING """
        ob_hand_embeddings = torch.zeros((batch_size,8,self.EMBDIM)).to(device)
        deck_embeddings = torch.zeros((batch_size,52,self.EMBDIM)).to(device)
        rest_embeddings = torch.zeros((batch_size,3,self.EMBDIM)).to(device)
        for i, obs_hand in enumerate(obs_hands):
            for j, card in enumerate(obs_hand):
                emb = Card.int_to_emb(int(card.item()))
                emb = emb + [1] * (self.EMBDIM - len(emb)) # one extend emb
                ob_hand_embeddings[i,j] = torch.tensor(emb)
        for i, deck in enumerate(decks):
            for j, present_bit in enumerate(deck):
                if present_bit:
                    emb = Card.int_to_emb(j)
                    emb = emb + [1] * (self.EMBDIM - len(emb)) # one extend emb
                    deck_embeddings[i,j] = torch.tensor(emb)
        for i, r in enumerate(rest):
            for j, val in enumerate(r):
                if present_bit:
                    emb = [val] * (self.EMBDIM)
                    rest_embeddings[i,j] = torch.tensor(emb)


        """ POSITION ENCODE THE OBSERVABLE HAND """
        ob_hand_embeddings = self.pe(ob_hand_embeddings)

        """ CONCATENATE ALL EMBEDDINGS """
        x = torch.concatenate((ob_hand_embeddings, deck_embeddings, rest_embeddings), dim=1)

        """ MULTI-ATTENTION BLOCKS """
        K = torch.ones((batch_size, self.EMBDIM, self.NHEADS)).to(device)
        V = torch.ones((batch_size, self.EMBDIM, self.NHEADS)).to(device)

        for i in range(self.NLAYERS):
            x = self.mha[i].forward(
                x, 
                K@self.key[i],
                V@self.val[i],
                need_weights=False
            )[0]
            x = self.lin[i].forward(x)
            x = self.relu(x)
            x = self.batch_norm[i](x)
            x = self.dropout(x)
        x = x[:,0,:436]
        x = self.softmax(x)

        return x.unsqueeze(1)



class DQNv1(nn.Module):
    def __init__(self):
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
        super(DQNv1, self).__init__()

        self.NHEADS = 54
        self.EMBDIM = 54
        self.NLAYERS = 8

        self.pe =  PositionalEncoding(self.EMBDIM).to(device)
        self.mha = [nn.MultiheadAttention(self.EMBDIM, self.NHEADS, batch_first=True).to(device)
            for _ in range(self.NLAYERS)
        ]
        self.key = [torch.randn((1,self.NHEADS,self.EMBDIM)).to(device) for _ in range(self.NLAYERS)]
        self.val = [torch.randn((1,self.NHEADS,self.EMBDIM)).to(device) for _ in range(self.NLAYERS)]
        self.lin = [nn.Linear(self.EMBDIM,self.EMBDIM).to(device) for _ in range(self.NLAYERS)]
        self.batch_norm = [nn.BatchNorm1d(num_features=8).to(device) for _ in range(self.NLAYERS)]

        LINDIM = self.EMBDIM + 52 + 3

        self.lin1 = nn.Linear(LINDIM,LINDIM).to(device)
        self.ban1 = nn.BatchNorm1d(num_features=1).to(device)
        self.lin2 = nn.Linear(LINDIM,LINDIM).to(device)
        self.ban2 = nn.BatchNorm1d(num_features=1).to(device)

        #assert(LINDIM >= 436)
        self.final_linear = nn.Linear(LINDIM, 436).to(device)

        self.relu = nn.ReLU().to(device)
        self.flatten = nn.Flatten().to(device)
        self.softmax = nn.Softmax(dim=2).to(device)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # obs_hand, deck, ...
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        x.to(device)

        # x        :: (B,1,63)
        # obs_hand :: (B,8)
        # rest     :: (B,55)
        obs_hands = x[:,:,0:8].squeeze(1).to(device)
        rest      = x[:,:,8:].squeeze(1).to(device)

        """ EMBEDDING """
        card_embeddings = torch.zeros((x.shape[0],8,self.EMBDIM)).to(device)
        for i, obs_hand in enumerate(obs_hands):
            for j, card in enumerate(obs_hand):
                emb = Card.int_to_emb(int(card.item()))
                emb = emb + [1] * (self.EMBDIM - len(emb)) # one extend emb
                card_embeddings[i,j] = torch.tensor(emb)

        batch_size = x.shape[0]
        K = torch.ones((batch_size, self.EMBDIM, self.EMBDIM)).to(device)
        V = torch.ones((batch_size, self.EMBDIM, self.EMBDIM)).to(device)

        """ MULTI-ATTENTION BLOCKS """
        attn_output = self.pe(card_embeddings)
        for i in range(self.NLAYERS):
            attn_output = self.mha[i].forward(
                attn_output, 
                K@self.key[i],
                V@self.val[i],
                need_weights=False
            )[0]
            attn_output = self.lin[i].forward(attn_output)
            attn_output = self.relu(attn_output)
            attn_output = self.batch_norm[i](attn_output)
        flat_output = self.flatten(attn_output[:,0,:])
        flat_output = torch.concatenate((flat_output, rest), dim=1)

        """ FFN """
        #print(f"{x=}")
        x = flat_output.unsqueeze(1)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.ban1(x)

        x = self.lin2(x)
        x = self.relu(x)
        x = self.ban2(x)

        x = self.final_linear(x)
        x = self.relu(x)
        x = self.softmax(x)

        return x

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
SKIP = 2000
MEM_SIZE = 2000
BATCH_SIZE = 128
TRAIN_FREQ = 1
GAMMA = 0.99  # no discount since the # of moves it takes to win doesn't matter
EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY = 10**4
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

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEM_SIZE)
        self.eps_threshold = EPS_START
        self.was_last_action_nn = False

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

        # Perform one step of the optimization (on the policy network) IF we exceed SKIP
        if self.steps_done >= SKIP:
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
        if self.steps_done >= SKIP: # only decrease epsilon when we go over SKIP
            self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * (self.steps_done - SKIP) / EPS_DECAY)
        self.steps_done += 1
        input = self.convert_state_to_input(state)
        self.was_last_action_nn = sample > self.eps_threshold
        if sample > self.eps_threshold:
            #print(f"get_action: Sampling action from policy, {self.eps_threshold=:.3f}")
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print(pn := self.policy_net(torch.from_numpy(input)))
                p = self.policy_net(torch.from_numpy(input))[0,0,:].cpu().numpy()
                #return np.random.choice(436, 1, p=p)[0]
                return np.argmax(p)
        else:
            dprint(f"get_action: Sampling action randomly, {self.eps_threshold=:.3f}")
            return self.env.action_space.sample()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE or (self.steps_done % TRAIN_FREQ) != 0:
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
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(2, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE).to(device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states.to(device)
            ).max(1).values.max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, 
            expected_state_action_values.unsqueeze(1).unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # garbage collect
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()



