from src.env import BalatroEnv
from src.agent.agent import Agent
from src.agent.device import device
from src.agent.replay_memory import ReplayMemory, Transition

import torch
import torch.nn as nn
import torch.optim as optim

import math
import random
import gc
from typing import Callable
import numpy as np


DEBUG = False

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
SKIP = 2000
MEM_SIZE = 2000
BATCH_SIZE = 256
TRAIN_FREQ = 1
GAMMA = 0.99  # no discount since the # of moves it takes to win doesn't matter
EPS_START = 0.90
EPS_END = 0.05
TAU = 0.005
LR = 1e-4

class DQNAgent(Agent):
    def __init__(
        self,
        env: BalatroEnv,
        nn_factory: Callable[[], nn.Module],
        EPS_DECAY=10**4
    ):
        super().__init__(env)

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n # type:ignore
        # Get the number of state observations
        # state, info = self.env.reset()
        # n_observations = len(state)

        self.policy_net = nn_factory().to(device)
        self.target_net = nn_factory().to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEM_SIZE)
        self.eps_threshold = EPS_START
        self.was_last_action_nn = False
        self.EPS_DECAY = EPS_DECAY

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
            target_net_state_dict[key] = (
                policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            )
        self.target_net.load_state_dict(target_net_state_dict)


    def get_action(self, state) -> int:
        sample = random.random()
        if self.steps_done >= SKIP: # only decrease epsilon when we go over SKIP
            self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * (self.steps_done - SKIP) / self.EPS_DECAY)
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
                output = self.policy_net(torch.from_numpy(input))
                if output.shape[-1] != 436:
                    raise Exception("OUTPUT has invalid shape:", output.shape)
                if len(output.shape) == 3:
                    p = output[0,0,:].cpu().numpy()
                else:
                    p = output[0,:].cpu().numpy()
                return np.argmax(p).item()
        else:
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
        pnout = self.policy_net(state_batch)
        if pnout.shape[-1] != 436:
            raise Exception("PNOUT has invalid shape:", pnout.shape)
        if len(pnout.shape) == 2:
            pnout = pnout.unsqueeze(1)
        state_action_values = pnout.gather(2, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE).to(device)
        with torch.no_grad():
            tnout = self.target_net(
                non_final_next_states.to(device)
            )
            if len(tnout.shape) == 3:
                tnout = tnout.squeeze(1)
            next_state_values[non_final_mask] = tnout.max(1).values
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



