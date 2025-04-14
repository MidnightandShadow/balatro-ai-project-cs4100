from src.agent.nn_agent import NNAgent
from src.agent.device import device
from src.agent.replay_memory import ReplayValueMemory, TransitionValue

import torch
import torch.nn as nn
import torch.optim as optim

import math
import random
import gc
from typing import Callable
import numpy as np

SKIP = 1000
CAPACITY = 1000
BATCH_SIZE = 128
TRAIN_FREQ = 1
GAMMA = 0.97
EPS_START = 0.90
EPS_END = 0.05
TAU = 0.005


class PPOAgent:
    def __init__(self, env, actor_factory, actor_critic_factory, EPS_CLIP=0.2):
        self.env = env
        self.actor = actor_factory().to(device)
        self.critic = actor_critic_factory().to(device)
        self.memory = ReplayValueMemory(CAPACITY)
        self.gamma = GAMMA
        self.training_sessions = 2000
        self.mse_loss = torch.nn.MSELoss()
        self.steps_done = 0
        self.eps_clip = EPS_CLIP
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters())
        )
        self.value_loss_coef = 0.6
        self.entropy_coef = 0.5

    def update(
        self,
        cur_state,  # obs
        action: int,
        log_prob: float,
        reward_f: float,
        terminated: bool,
        next_state,  # obs
    ):
        reward = torch.tensor([reward_f])

        # convert to input type
        cur_state = torch.from_numpy(NNAgent.convert_state_to_input(cur_state))
        next_state = torch.from_numpy(NNAgent.convert_state_to_input(next_state))

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(cur_state, dtype=torch.float32)

        state_val = self.critic(cur_state)

        assert reward is not None
        # Store the transition in memory
        self.memory.push(
            torch.tensor(cur_state, dtype=torch.float32),
            torch.tensor([action], dtype=torch.long),
            next_state,
            log_prob,
            reward,
            state_val,
        )

    def get_action_and_value(self, state) -> tuple[tuple[int, float], float]:
        """(action, log_prob), value"""
        input = NNAgent.convert_state_to_input(state)
        output = self.actor(torch.from_numpy(input))
        if output.shape[-1] != 436:
            raise Exception("OUTPUT has invalid shape:", output.shape)

        with torch.no_grad():
            if len(output.shape) == 3:
                p = output[0, 0, :].cpu().numpy()
            else:
                p = output[0, :].cpu().numpy()
            ind = np.argmax(p).item()
            self.steps_done += 1
            return (ind, np.log(p[ind])), self.critic(torch.from_numpy(input)).item()  # type: ignore

    def discounted_returns(self, batch):
        """Truncated version of GAE"""
        transitions = TransitionValue(*zip(*batch))
        returns = []
        discounted_reward = 0
        for reward in reversed(transitions.reward):
            assert reward is not None
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = np.array(returns)
        returns = torch.flatten(torch.from_numpy(returns).float()).to(device)
        return returns

    def optimize_model(self):
        """batching data"""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        transitions = TransitionValue(*zip(*batch))
        discounted_rewards = self.discounted_returns(batch)

        state_vals = torch.tensor(transitions.state_value)
        advantages = discounted_rewards - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        ITERATIONS = 20
        for i in range(ITERATIONS):
            # (B, 436)

            action_probabilities = self.actor(torch.cat(transitions.state, dim=0))
            action_probabilities = action_probabilities[:, transitions.action]
            actor_log_probs = torch.log(action_probabilities)

            old_log_probs = torch.tensor(transitions.logprobs)

            ratios = torch.exp(actor_log_probs - old_log_probs)

            # Finding L^{CLIP} inner min LHS and RHS
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Calculating -L^{CLIP} => it is negative because we want adam to optimize
            l_clip = -torch.min(surr1, surr2).mean()

            state_values = torch.tensor(transitions.state_value)
            critic_loss = self.mse_loss(state_values, discounted_rewards)

            # entropy.shape: (B,)
            entropy = torch.sum(
                action_probabilities * torch.log(action_probabilities), dim=-1
            )

            loss = (
                l_clip
                + self.value_loss_coef * critic_loss
                - self.entropy_coef * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
