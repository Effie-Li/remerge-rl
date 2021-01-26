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
                 max_higher_order_transition=6,
                 **kwargs):
        
        # self.memory is inherited from super, functions as the default batch sampling buffer
        # self.attractor_network is the memory network that does recurrent computation on memory keys
        # self.states, next_states, and links are the corresponding content storage for self.remerge
        
        super().__init__(num_slot=num_slot, batch_size=batch_size)
        
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.max_higher_order_transition = max_higher_order_transition
        
        self.attractor_network = AttractorNetwork(hidden_size, state_size, **kwargs)
        self.all_states = deque() # unique states remembered
        self.links = deque()
    
    def add(self, state, action, reward, next_state, goal):
        super().add(state, action, reward, next_state, goal)
    
    def wire_memory(self):
        
        self.attractor_network.clean_weights()
        self.attractor_network.clean_buffer()
        self.all_states = deque()
        self.links = deque()
        
        for transition in self.memory:
            state = transition.state
            next_state = transition.next_state
        
            if next_state is None:
                next_state = transition.goal # arrived at goal

            state = state.detach().cpu().numpy()[0]
            next_state = next_state.detach().cpu().numpy()[0]

            # if state == next_state, non-meaninful transtiion, don't remember lol
            if np.array_equal(state, next_state):
                return

            # maintain size of self.links
            # TODO: maintain size of self.all_states
            if len(self.links) == self.hidden_size:
                # make room for the new link
                self.links.popleft()
                # link indexes shifted left, shift weights accordingly
                # reset connections to and from the empty right most position in deque
                self.attractor_network.weights['s2h'] = np.concatenate((self.attractor_network.weights['s2h'][:, 1:], 
                                                                        np.zeros((self.state_size, 1))), axis=1)
                self.attractor_network.weights['ns2h'] = np.concatenate((self.attractor_network.weights['ns2h'][:, 1:], 
                                                                         np.zeros((self.state_size, 1))), axis=1)

                self.attractor_network.weights['h2s'] = np.concatenate((self.attractor_network.weights['h2s'][1:, :],
                                                                        np.zeros((1, self.state_size))), axis=0)
                self.attractor_network.weights['h2ns'] = np.concatenate((self.attractor_network.weights['h2ns'][1:, :],
                                                                         np.zeros((1, self.state_size))), axis=0)

            step = 1 # incoming links are always 1-step transitions
            new_link = Linked(*(state, next_state, step))

            if self.find_link(self.links, new_link) != -1:
                # memory link exists, add a higher-order link instead
                all_nns, all_steps = self.find_all_ns(new_link.state2)
                if len(all_nns)==0: # no available further states
                    continue
                can_add = False
                for i, nns in enumerate(all_nns):
                    if np.array_equal(new_link.state1, nns) or all_steps[i]+1>self.max_higher_order_transition:
                        # the reversal transition or it's too high-order, ignore
                        continue
                    higher_link = Linked(*(new_link.state1, nns, all_steps[i]+1))
                    if self.find_link(self.links, higher_link) == -1:
                        can_add = True
                        new_link = higher_link
                        break
                if not can_add:
                    continue

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
    
    def plan(self, s_probe=None, ns_probe=None, n_level=1, T=4, mode='sample'):
        
        # probes are in memory content space (e.g. [C,H,W])
        # self.attractor_network.forward can take None inputs
        # if plan_steps==0 one of s_probe and ns_probe better be None
        # T better be > 10, at least > 1 so that we won't run into all 
        #     zeros after softmax and random.choice complains
        # mode=='max': select state corresponding to max activation
        #       'sample': select state based on softmax activation as probabilities
        
        if s_probe is None and ns_probe is None:
            print("what do you want from meee??")
            return None
        
        # turn probes into keys
        s_index = self.find_state(self.all_states, s_probe) if s_probe is not None else -1
        s_in = self.index_to_onehot(s_index, self.state_size) if s_index != -1 else None
        ns_index = self.find_state(self.all_states, ns_probe) if ns_probe is not None else -1
        ns_in = self.index_to_onehot(ns_index, self.state_size) if ns_index != -1 else None
        
        if s_index == -1 or ns_index == -1:
            # not in memory...
            return None
        
        n1 = self.attractor_network.clone()
        n2 = self.attractor_network.clone()
        
        for t in range(n_level):
            # propogate successor info in n1 and predecessor info in n2
            if t==0:
                n1sact, n1nsact, n1hact = n1.forward(s_in=s_in, binarize=True)
                n1sact, n1nsact, n1hact = n1.forward(s_in=s_in, binarize=True)
                n2sact, n2nsact, n2hact = n2.forward(ns_in=ns_in, binarize=True)
                n2sact, n2nsact, n2hact = n2.forward(ns_in=ns_in, binarize=True)
            else:
                n1sact, n1nsact, n1hact = n1.forward(s_in=n1nsact, binarize=True)
                n1sact, n1nsact, n1hact = n1.forward(s_in=n1nsact, binarize=True)
                n2sact, n2nsact, n2hact = n2.forward(ns_in=n2sact, binarize=True)
                n2sact, n2nsact, n2hact = n2.forward(ns_in=n2sact, binarize=True)
            n1.clean_buffer()
            n2.clean_buffer()
        
        # check overlap
        for t in range(T):
            sact, nsact, hact = n1.forward(s_in=n1nsact, ns_in=n2sact)
        
        s_index = self.activation_to_index(sact.flatten(), mode=mode)
        ns_index = self.activation_to_index(nsact.flatten(), mode=mode)
        if s_index is None or ns_index is None:
            return
        
        s_key = self.index_to_onehot(s_index, self.state_size)
        s = self.retrieve_instance(self.all_states, s_key)
        ns_key = self.index_to_onehot(ns_index, self.state_size)
        ns = self.retrieve_instance(self.all_states, ns_key)
        return s, ns
    
    def activation_to_index(self, activation, mode='max'):
        # activations is shape [B, state_size]
        if mode=='max':
            index = np.argmax(activation, axis=-1)
        elif mode=='sample':
            index = self.sample_state(activation) # list(map(self.sample_state, activation))
        return index
    
    def sample_state(self, probs):
        elements = list(range(self.state_size))
        try:
            s = np.random.choice(elements, 1, p=probs)[0]
        except:
            # if no state was activated, probs don't sum to one and numpy complains
            s = None
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
        nss = []
        steps = []
        for l in self.links:
            if np.array_equal(s, l.state1):
                nss.append(self.all_states[self.find_state(self.all_states, l.state2)])
                steps.append(l.step)
        return nss, steps