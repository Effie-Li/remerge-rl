from .replay_buffer import ReplayBuffer
from .attractor_network import AttractorNetwork
from collections import deque, namedtuple
import numpy as np

Linked = namedtuple('Linked', ('state1', 'state2', 'step'))

class RemergeMemory(ReplayBuffer):
    
    '''
    remerge wrapper for replay buffer, 
    maintains memory content storages for state and hidden layers
    '''
    
    def __init__(self,
                 num_slot=10000,
                 batch_size=32,
                 hidden_size=1000, 
                 state_size=1000,
                 **kwargs):
        
        # self.memory is inherited from super, functions as the default batch sampling buffer
        # self.remerge is the memory network that does recurrent computation on onehot memory keys
        # self.states, next_states, and links are the corresponding content storage for self.remerge
        
        super().__init__(num_slot=num_slot, batch_size=batch_size)
        
        self.hidden_size = hidden_size
        self.state_size = state_size
        
        self.attractor_network = AttractorNetwork(hidden_size, state_size, **kwargs)
        self.all_states = deque() # unique states remembered
        self.links = deque()
    
    def add(self, state, action, reward, next_state, goal):
        super().add(state, action, reward, next_state, goal)
        
        if next_state is None:
            next_state = goal # arrived at goal
        
        state = state.detach().numpy()[0]
        next_state = next_state.detach().numpy()[0]
        
        # if state == next_state, non-meaninful transtiion, don't remember lol
        if np.array_equal(state, next_state):
            return
        
        step = 1 # new links are always 1-step transitions
        new_link = Linked(*(state, next_state, step))

        if self.find_link(self.links, new_link) != -1:
            # memory link exists, don't need to adjust memory network weights
            return

        # TODO: maintain num_slots in all of (links, states, next_states)
        # TODO: when memory is freed up need to update weights
        if len(self.links) > self.hidden_size:
            # assuming implementation is correct, effective length of
            # states and next_states will always be <= links, so
            # we only need to check link capacity and free things accordingly
            pass

        self.links.append(new_link)
        link_index = len(self.links)-1

        state_index = self.find_state(self.all_states, state)
        if state_index == -1:
            self.all_states.append(state)
            state_index = len(self.all_states)-1

        next_state_index = self.find_state(self.all_states, next_state)
        if next_state_index == -1:
            self.all_states.append(next_state)
            next_state_index = len(self.all_states)-1

        # construct new weights as memory comes in
        self.attractor_network.update_weights('s2h', pre_index=state_index, post_index=link_index, mode='add')
        self.attractor_network.update_weights('h2s', pre_index=link_index, post_index=state_index, mode='add')
        self.attractor_network.update_weights('ns2h', pre_index=next_state_index, post_index=link_index, mode='add')
        self.attractor_network.update_weights('h2ns', pre_index=link_index, post_index=next_state_index, mode='add')
    
    def plan(self, s_probe=None, ns_probe=None, plan_steps=4, T=100, mode='sample'):
        
        # plan_steps needs to be >=1
        # probes are in memory content space (e.g. [C,H,W])
        # ns_probe is goal state
        # self.attractor_network.forward can take None inputs
        # if plan_steps==0 one of s_probe and ns_probe better be None
        # T better be > 10, at least > 1 so that we won't run into all 
        #     zeros after softmax and random.choice complains
        # mode=='max': select state corresponding to max activation
        #       'sample': select state based on softmax activation as probabilities
        
        if s_probe is None and ns_probe is None:
            print("what do you want from meee??")
            return []
        
        # turn probes into keys
        s_index = self.find_state(self.all_states, s_probe) if s_probe is not None else -1
        s_in = self.index_to_onehot(s_index, self.state_size) if s_index != -1 else None
        ns_index = self.find_state(self.all_states, ns_probe) if ns_probe is not None else -1
        ns_in = self.index_to_onehot(ns_index, self.state_size) if ns_index != -1 else None
        
        if plan_steps==0:
            # just activate the linked memories, no multiple copies needed
            n = self.attractor_network.clone()
            
            for t in range(T):
                sact, nsact, hact = n.forward(s_in=s_in, ns_in=ns_in)
            
            if ns_probe is None:
                plan_indexes = self.activation_to_indexes(nsact, mode=mode)
                plan_keys = [self.index_to_onehot(ind, self.state_size) for ind in plan_indexes]
                plan = [self.retrieve_instance(self.all_states, key) for key in plan_keys]
            else: # hopefully s_probe is None:
                plan_indexes = self.activation_to_indexes(sact, mode=mode)
                plan_keys = [self.index_to_onehot(ind, self.state_size) for ind in plan_indexes]
                plan = [self.retrieve_instance(self.all_states, key) for key in plan_keys]
            return plan

        # make copies of the attractor network
        sub_networks = [self.attractor_network.clone() for _ in range(plan_steps+1)]
        
        activation_buffer = {'sact': np.zeros((len(sub_networks), self.state_size)),
                             'nsact':np.zeros((len(sub_networks), self.state_size)),
                             'hact':np.zeros((len(sub_networks), self.hidden_size))}
        
        # insert probes
        sact, nsact, hact = sub_networks[0].forward(s_in=s_in, ns_in=None)
        activation_buffer['sact'][0] = sact
        activation_buffer['nsact'][0] = nsact
        activation_buffer['hact'][0] = hact
        sact, nsact, hact = sub_networks[-1].forward(s_in=None, ns_in=ns_in)
        activation_buffer['sact'][-1] = sact
        activation_buffer['nsact'][-1] = nsact
        activation_buffer['hact'][-1] = hact
        
        # run recurrent computation and settle (?) on plan
        # TODO: check convergence?
        # keep injecting s_in and ns_in at both ends
        for t in range(T):
        
            for n in reversed(range(len(sub_networks)-1)): # skip last network
                ns = activation_buffer['sact'][n+1] # state from next sub network
                sact, nsact, hact = sub_networks[n].forward(ns_in=ns)
                activation_buffer['sact'][n] = sact
                activation_buffer['nsact'][n] = nsact
                activation_buffer['hact'][n] = hact
            
            for n in range(len(sub_networks)):
                if n==0: # skip first network
                    continue
                s = activation_buffer['nsact'][n-1] # state from next sub network
                sact, nsact, hact = sub_networks[n].forward(s_in=s)
                activation_buffer['sact'][n] = sact
                activation_buffer['nsact'][n] = nsact
                activation_buffer['hact'][n] = hact

#         print('sact: ', activation_buffer['sact'])
#         print('hact: ', activation_buffer['hact'])
#         print('nsact: ', activation_buffer['nsact'])
#         print()
        
        # form plan
        plan_indexes = self.activation_to_indexes(activation_buffer['nsact'][:-1], mode=mode)
        print('plan_indexes: ', plan_indexes)
        plan_keys = [self.index_to_onehot(ind, self.state_size) for ind in plan_indexes]
        # retrieve content from memory
        plan = [self.retrieve_instance(self.all_states, key) for key in plan_keys]

        return plan
    
    def activation_to_indexes(self, activation, mode='max'):
        # activations is shape [B, state_size]
        if mode=='max':
            plan_indexes = np.argmax(activation, axis=-1)
        elif mode=='sample':
            plan_indexes = list(map(self.sample_state, activation))
        return plan_indexes
    
    def sample_state(self, probs):
        # in case there is a tie
        elements = list(range(self.state_size))
        return np.random.choice(elements, 1, p=probs)[0]

    def find_link(self, links, link):
        # finds first instance of same link in links else -1
        for i, x in enumerate(links):
            checks = [np.array_equal(getattr(x, f), getattr(link, f)) for f in x._fields]
            if sum(checks) == len(checks):
                return i
        return -1

    def find_state(self, states, state):
        # finds first instance of state in states else -1
        checks = [np.array_equal(x, state) for x in states]
        if sum(checks)==0:
            return -1
        else:
            return checks.index(True)

    def retrieve_instance(self, memory, onehot_key):
        # works for both link and state storages
        if len(np.unique(onehot_key))!=2 or (sum(onehot_key) > 1):
            # okay this isn't the perfect check but...
            raise ValueError('invalid memory key!')
        return memory[self.onehot_to_index(onehot_key)]

    def onehot_to_index(self, onehot):
        return np.where(onehot==1)[0][0]

    def index_to_onehot(self, i, length):
        key = np.zeros(length)
        key[i] = 1.
        return key
    
    def forget(self):
        pass