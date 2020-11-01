from collections import deque
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, num_slot, batch_size=32):
        self.num_slot = num_slot
        self.batch_size = batch_size
        self.memory = deque()
        
    def add(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.num_slot:
            # remove oldest memory
            self.memory.popleft()
        
    def sample(self, batch_size=None):
        if (batch_size is None):
            batch_size = self.batch_size
        samples = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in samples])
        actions = np.array([x[1] for x in samples])
        rewards = np.array([x[2] for x in samples])
        next_states = np.array([x[3] for x in samples])
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.memory)