import random
from collections import namedtuple, deque

"""

TODO: Make ReplayMemory take in the type of the transitiion

"""

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

TransitionValue = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'state_value')
)

class ReplayValueMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(TransitionValue(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

