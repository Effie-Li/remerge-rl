import math
import random
import torch
import torch.nn
import torch.nn.functional as F
from . import ConvNet

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class DQN:
    '''
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    
    def __init__(self, state_dim, action_dim, device, memory):
        self.state_dim = state_dim # [C, H, W]
        c, h, w = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.network = ConvNet(h=h, w=w, n_out=action_dim).to(device)
        self.target_network = self.network.clone().to(device)
        
        self.memory = memory
        self.optim = torch.optim.RMSprop(self.network.parameters())
        
        self.step_count = 0
    
    def step(self, state):
        qvals = self.network(state) # [1, action_dim]
        
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.step_count / EPS_DECAY)
        
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = qvals.max(1)[1].view(1, 1)
        else:
            action =  torch.tensor([[random.randrange(self.action_dim)]], 
                                   device=self.device, dtype=torch.long)
        
        self.step_count += 1 # maybe train() increments it?
        return action
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
        
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(batch_size=BATCH_SIZE)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optim.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        
        return loss
    
    def test(self, num_episodes=100):
        pass