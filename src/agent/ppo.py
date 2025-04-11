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
GAMMA = 0.99  # no discount since the # of moves it takes to win doesn't matter
EPS_START = 0.90
EPS_END = 0.05
TAU = 0.005
LR = 1e-4

class PPOAgent(NNAgent):
    def __init__(self, env, actor_factory, value_factory, EPS_DECAY=10**-4):
        self.env = env
        self.actor = actor_factory().to(device)
        self.value = value_factory().to(device)
        self.memory = ReplayValueMemory(CAPACITY)
        self.gamma = 0.99
        self.EPS_DECAY = EPS_DECAY
        self.training_sessions = 2000
        

    def update(self, cur_state, action: int, reward_f: float, terminated: bool, next_state):
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

    def get_action_and_value(self, state) -> tuple[int, int]:
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
                output = self.actor(torch.from_numpy(input))
                if output.shape[-1] != 436:
                    raise Exception("OUTPUT has invalid shape:", output.shape)
                if len(output.shape) == 3:
                    p = output[0,0,:].cpu().numpy()
                else:
                    p = output[0,:].cpu().numpy()
                return np.argmax(p).item(), self.value(input) # type: ignore
        else:
            return self.env.action_space.sample(), self.value(input) # type: ignore


    def get_action(self, state) -> int:
        return self.get_action_and_value(state)[0]

    def discounted_returns(self) -> torch.Tensor:
        transitions = self.memory.sample(BATCH_SIZE)
        transitions = TransitionValue(*zip(*transitions))
        returns = []
        discounted_reward = 0
        for reward, nxt_state in zip(reversed(transitions.reward), reversed(transitions.next_state)):
            done = nxt_state is None
            if done: 
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = np.array(returns)
        returns = torch.flatten(torch.from_numpy(returns).float()).to(device)
        return returns

    def optimize_model(self):
        # print(len(self.buffer.rewards))
        rewards_to_go = self.compute_returns()
        # print(len(rewards_to_go))

        states = torch.from_numpy(np.array(self.buffer.states)).float().to(device)
        actions = torch.from_numpy(np.array(self.buffer.actions)).float().to(device)
        old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs)).float().to(device)
        state_vals = torch.from_numpy(np.array(self.buffer.state_values)).float().to(device)

        # print('stage-0:', rewards_to_go.shape, state_vals.shape)
        # print('stage-1:', rewards_to_go.device, state_vals.device)
        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        

        for _ in range(self.training_sessions):
            # generate random indices for minibatch
            indices = np.random.permutation(len(self.buffer.states))

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]
                
                # evaluate old actions and values
                state_values, logprobs, dist_entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                # print(logprobs.shape, batch_old_logprobs.shape)

                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - batch_old_logprobs.squeeze(-1))

                # Finding Surrogate Loss
                # print(ratios.shape, batch_advantages.shape)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages

                # final loss of clipped objective PPO
                actor_loss = -torch.min(surr1, surr2).mean()
                # print(state_values.dtype, batch_rewards_to_go.dtype)
                critic_loss = 0.5 * self.mse_loss(state_values.squeeze(), batch_rewards_to_go)
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean()
                # print("Final loss:", actor_loss, critic_loss, dist_entropy, loss)

                # calculate gradients and backpropagate for actor network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.buffer.clear()
