import math
import random
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from itertools import count
from . import GoalCondGridSimpleNet, GoalCondGridNet, GridConvNet, GoalCondGridConvNet

class DQN():
    '''
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 device, 
                 memory,
                 goalcond=False):
        
        self.state_dim = state_dim # [C, H, W]
        # c, h, w = state_dim
        self.action_dim = action_dim
        self.device = device
        self.memory = memory
        self.goalcond = goalcond
        
        if self.goalcond:
            self.network = GoalCondGridSimpleNet(n_in=self.state_dim, n_out=action_dim).to(device)
        else:
            self.network = GridConvNet(h=h, w=w, n_out=action_dim).to(device)
        self.target_network = self.network.clone().to(device)

        self.optim = torch.optim.RMSprop(self.network.parameters(), lr=3e-4)
        
        self.step_count = 0
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 200
    
    def select_action(self, qvals, explore=True):
        
        # for a single step
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.step_count / self.EPS_DECAY)
        
        sample = random.random()
        
        if explore and (sample <= eps_threshold):
            action = torch.tensor([[random.randrange(self.action_dim)]], 
                                   device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = qvals.max(1)[1].view(1,1)
        
        return action #[[a]]
    
    def step(self,
             state,
             goal=None,
             explore=True):
        
        if self.goalcond:
            if goal is None:
                raise ValueError('Gotta have a goal!!')
            batch_qvals = self.network(state, goal)
        else:
            batch_qvals = self.network(state) # [batch, action_dim]
        
        actions = [self.select_action(qvals.unsqueeze(0), explore=explore) for qvals in batch_qvals]
        
        return actions
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def _train(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = self.memory.sample(batch_size=self.BATCH_SIZE)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        if self.goalcond:
            goal_batch = torch.cat(batch.goal)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        if self.goalcond:
            state_action_values = self.network(state_batch, goal_batch).gather(1, action_batch)
        else:
            state_action_values = self.network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or the reward in case the state was final.
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            if self.goalcond:
                # trim goal_batch to match with non_final_next_states
                non_final_goal_states = torch.cat([goal_batch[i:i+1] for i, mask in enumerate(non_final_mask) if mask], axis=0)
                next_state_values[non_final_mask] = self.target_network(non_final_next_states, 
                                                                        non_final_goal_states).max(1)[0].detach()
            else:
                next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        
        # final states will have next_state_values=0, making expected qval (0*gamma+reward)
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optim.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        
        return loss