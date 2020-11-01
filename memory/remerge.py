from .replay_buffer import ReplayBuffer

class RemergeMemory(ReplayBuffer):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.states = []
        self.next_states = []
    
        # initialize hopfield net
        self.init_memory_network()
    
    def init_memory_network(self):
        
        self.hidden_size = self.num_slot
        self.state_size = self.num_slot
        self.next_state_size = self.num_slot
        
        self.s2h = np.zeros((self.state_size, self.hidden_size))
        self.h2s = np.zeros((self.state_size, self.hidden_size))
        self.ns2h = np.zeros((self.state_size, self.hidden_size))
        self.h2ns = np.zeros((self.state_size, self.hidden_size))
    
    def add(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
        # add to self.states and self.next_state
        self.states.append(state)
        self.next_states.append(next_state)
        
        if len(self.memory) > self.num_slot:
            # remove oldest memory
            self.free()
        
    def free(self):
        forget = self.memory.popleft()
        state = forget[0]
        next_state = forget[-1]
        # TODO: adjust memory network connections
        pass