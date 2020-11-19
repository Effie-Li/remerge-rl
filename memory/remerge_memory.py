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
        
        state = state.detach().cpu().numpy()[0]
        next_state = next_state.detach().cpu().numpy()[0]
        
        # if state == next_state, non-meaninful transtiion, don't remember lol
        if np.array_equal(state, next_state):
            return
        
        step = 1 # incoming links are always 1-step transitions
        new_link = Linked(*(state, next_state, step))

        if self.find_link(self.links, new_link) != -1:
            # memory link exists, add a step+1 link instead
            link = self.links[self.find_link(self.links, new_link)]
            all_nns = self.find_all_ns(next_state) # next next state
            can_add = [self.find_link(self.links, Linked(*(state, nns, link.step+1)))!=-1
                       for nns in all_nns]
            if np.sum(can_add) == 0: # all related step+1 transitions exist
                return
            else: # go on and add a step+1 transition
                new_link = l
            return

        # maintain size of self.links
        # TODO: maintain size of self.all_states
        if len(self.links) > self.hidden_size:
            self.links.popleft()
            # link indexes shifted left, shift weights accordingly
            # reset connections to and from the empty right most position in deque
            self.attractor_network.weights['s2h'] = np.concatenate((self.attractor_network.weights['s2h'][:, 1:], 
                                                                    np.zeros((self.state_size, 1))), axis=1)
            self.attractor_network.weights['ns2h'] = np.concatenate((self.attractor_network.weights['ns2h'][:, 1:], 
                                                                     np.zeros((self.state_size, 1))), axis=1)
            
            self.attractor_network.weights['h2s'] = np.concatenate((self.attractor_network.weights['h2s'][1:, :],
                                                                    np.zeros((1, self.state_size,))), axis=0)
            self.attractor_network.weights['h2ns'] = np.concatenate((self.attractor_network.weights['h2ns'][1:, :],
                                                                     np.zeros((1, self.state_size,))), axis=0)

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
        
        # probes are in memory content space (e.g. [C,H,W])
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
        activations = self.run_recurrence(s_in, ns_in, sub_networks, T)
        
        # form plan
        plan_indexes = self.activation_to_indexes(activations['nsact'][:-1], mode=mode)
        plan_keys = [self.index_to_onehot(ind, self.state_size) for ind in plan_indexes]
        # retrieve content from memory
        plan = [self.retrieve_instance(self.all_states, key) for key in plan_keys]

        return plan
    
    def run_recurrence(self, s_in, ns_in, sub_networks, T):
        
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
            
            # TODO: how best to implement recurrence on this system?
            
            # from side to center
#             for i in range(len(sub_networks)//2):
#                 n = i
#                 if i==0: # first network
#                     s = None # s_in
#                     ns = activation_buffer['sact'][n+1]
#                 else:
#                     s = activation_buffer['nsact'][n-1]
#                     ns = activation_buffer['sact'][n+1]
                
#                 sact, nsact, hact = sub_networks[n].forward(s_in=s, ns_in=ns)
#                 activation_buffer['sact'][n] = sact
#                 activation_buffer['nsact'][n] = nsact
#                 activation_buffer['hact'][n] = hact
                
#                 n = len(sub_networks)-1-i
#                 if i==0: # last network
#                     s = activation_buffer['nsact'][n-1]
#                     ns = None # ns_in
#                 else:
#                     s = activation_buffer['nsact'][n-1]
#                     ns = activation_buffer['sact'][n+1]
                
#                 sact, nsact, hact = sub_networks[n].forward(s_in=s, ns_in=ns)
#                 activation_buffer['sact'][n] = sact
#                 activation_buffer['nsact'][n] = nsact
#                 activation_buffer['hact'][n] = hact
                
#             if len(sub_networks) % 2 != 0:
#                 # deal with the middle one when there is an odd number of networks
#                 n = len(sub_networks)//2
#                 s = activation_buffer['nsact'][n-1]
#                 ns = activation_buffer['sact'][n+1]
#                 sact, nsact, hact = sub_networks[n].forward(s_in=s, ns_in=ns)
#                 activation_buffer['sact'][n] = sact
#                 activation_buffer['nsact'][n] = nsact
#                 activation_buffer['hact'][n] = hact
            
#             # from center to side
#             for i in reversed(range(len(sub_networks)//2)):
#                 n = i
#                 if i==0: # first network
#                     s = None # s_in
#                     ns = activation_buffer['sact'][n+1]
#                 else:
#                     s = activation_buffer['nsact'][n-1]
#                     ns = activation_buffer['sact'][n+1]
                
#                 sact, nsact, hact = sub_networks[n].forward(s_in=s, ns_in=ns)
#                 activation_buffer['sact'][n] = sact
#                 activation_buffer['nsact'][n] = nsact
#                 activation_buffer['hact'][n] = hact
                
#                 n = len(sub_networks)-1-i
#                 if i==0: # last network
#                     s = activation_buffer['nsact'][n-1]
#                     ns = None # ns_in
#                 else:
#                     s = activation_buffer['nsact'][n-1]
#                     ns = activation_buffer['sact'][n+1]
                
#                 sact, nsact, hact = sub_networks[n].forward(s_in=s, ns_in=ns)
#                 activation_buffer['sact'][n] = sact
#                 activation_buffer['nsact'][n] = nsact
#                 activation_buffer['hact'][n] = hact
            
            for n in range(len(sub_networks)):
                if n==0: # first network
                    sact, nsact, hact = sub_networks[n].forward()
                else:
                    s = activation_buffer['nsact'][n-1] # next_state from the previous sub network
                    sact, nsact, hact = sub_networks[n].forward(s_in=s)
                activation_buffer['sact'][n] = sact
                activation_buffer['nsact'][n] = nsact
                activation_buffer['hact'][n] = hact
        
            for n in reversed(range(len(sub_networks))): # skip last network
                if n==len(sub_networks)-1: # last network
                    sact, nsact, hact = sub_networks[n].forward()
                else:
                    ns = activation_buffer['sact'][n+1] # state from the next sub network
                    sact, nsact, hact = sub_networks[n].forward(ns_in=ns)
                activation_buffer['sact'][n] = sact
                activation_buffer['nsact'][n] = nsact
                activation_buffer['hact'][n] = hact
        
        return activation_buffer
    
    def activation_to_indexes(self, activation, mode='max'):
        # activations is shape [B, state_size]
        if mode=='max':
            plan_indexes = np.argmax(activation, axis=-1)
        elif mode=='sample':
            plan_indexes = list(map(self.sample_state, activation))
        return plan_indexes
    
    def sample_state(self, probs):
        elements = list(range(self.state_size))
        try:
            s = np.random.choice(elements, 1, p=probs)[0]
        except:
            # if no state was activated, probs don't sum to one and numpy complains
            s = np.random.choice(list(range(len(self.all_states))), 1)[0]
        return s

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
    
    def find_all_ns(self, s):
        x = []
        for l in self.links:
            if np.array_equal(s, l.state1):
                x.append(self.find_state(self.all_states, l.state2))
        return x