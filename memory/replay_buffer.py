from collections import deque, namedtuple
import numpy as np
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'goal'))

class ReplayBuffer(object):
    def __init__(self, num_slot=10000, batch_size=32):
        self.num_slot = num_slot
        self.batch_size = batch_size
        self.memory = deque()
        
    def add(self, state, action, reward, next_state, goal=None):
        self.memory.append(Transition(*(state, action, reward, next_state, goal)))
        if len(self.memory) > self.num_slot:
            # remove oldest memory
            self.memory.popleft()
    
    def sample(self, batch_size=None):
        if (batch_size is None):
            batch_size = self.batch_size
        transitions = random.sample(self.memory, batch_size)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        return batch

    def __len__(self):
        return len(self.memory)