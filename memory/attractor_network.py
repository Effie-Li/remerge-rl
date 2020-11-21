import numpy as np

class AttractorNetwork:
    
    '''
    instance-based memory with recurrent computations
    onehot memory code can be used as key to memory storage in parent class
    '''
    
    def __init__(self, 
                 hidden_size, 
                 state_size,
                 raw_excite_weight=1.0,
                 raw_inhibit_weight=0.0,
                 weight_scale=1.0,
                 i_tau=0.04,
                 h_tau=0.04, 
                 h_C=0.,
                 lmda=0.1, 
                 ext=0.2):
        
        self.hidden_size = hidden_size
        self.state_size = state_size
        
        self.weight_scale = weight_scale # class hyperparam?
        self.excite = raw_excite_weight * self.weight_scale
        self.inhibit = raw_inhibit_weight * self.weight_scale
        
        self.i_tau = i_tau
        self.h_tau = h_tau
        self.h_C = h_C
        self.lmda = lmda
        self.ext = ext
        
        self.weights = {
            's2h':np.zeros((self.state_size, self.hidden_size)),
            'h2s':np.zeros((self.hidden_size, self.state_size)),
            'ns2h':np.zeros((self.state_size, self.hidden_size)),
            'h2ns':np.zeros((self.hidden_size, self.state_size)),
        }
        
        self.netin_buffer = {
            's':np.zeros((1, self.state_size)),
            'ns':np.zeros((1, self.state_size)),
            'h':np.zeros((1, self.hidden_size))
        }
    
    def update_weights(self, weight_dict_key, pre_index, post_index, mode='add'):
        # add or remove a specific weight between state and link
        # pre_index indicates the node whose outgoing weights are being updated
        # post_index indicates the node whose incoming weights are being updated
        new_weights = self.weights[weight_dict_key][pre_index]

        if mode == 'add':
            new_weights[post_index] = self.excite
            # update zero weights to inhibitory
            new_weights[new_weights==0.0] = self.inhibit

        elif mode == 'del':
            new_weights[post_index] = self.inhibit
            # if no more excitatory connections change all weights to zero
            if sum(new_weights==self.excite) == 0:
                new_weights = np.zeros(len(new_weights))

        self.weights[weight_dict_key][pre_index] = new_weights

    def i_activation(self, netin):
        # logistic
        # return np.array([1/(1+np.exp(-x/self.i_tau)) for x in netin]) - 0.5 # i don't want zero memories to be activated
        # softmax
        # NOTE: on non-zero values only, to ensure empty memory slots don't get chosen
        non_zero_mask = netin!=0.0
        masked_netin = netin[non_zero_mask]
        denom = np.sum([np.exp(x/self.i_tau) for x in masked_netin])
        activation = np.array([np.exp(x/self.i_tau)/denom for x in masked_netin])
        netin[non_zero_mask] = activation
        return netin

    def h_activation(self, netin):
        # hedge softmax
        non_zero_mask = netin!=0.0
        masked_netin = netin[non_zero_mask]
        denom = self.h_C**(1/self.h_tau) + np.sum([np.exp(x/self.h_tau) for x in masked_netin])
        activation = np.array([np.exp(x/self.h_tau)/denom for x in masked_netin])
        netin[non_zero_mask] = activation
        return netin

    def forward(self, s_in=None, ns_in=None, binarize=False):
        
        s_in = s_in if s_in is not None else np.zeros((1,self.state_size))
        ns_in = ns_in if ns_in is not None else np.zeros((1,self.state_size))

        snetin = np.dot(self.h_activation(self.netin_buffer['h']),self.weights['h2s']) + self.ext*s_in
        snetin = self.lmda*(snetin) + (1-self.lmda)*self.netin_buffer['s']
        nsnetin = np.dot(self.h_activation(self.netin_buffer['h']),self.weights['h2ns']) + self.ext*ns_in
        nsnetin = self.lmda*(nsnetin) + (1-self.lmda)*self.netin_buffer['ns']
        hnetin = np.dot(self.i_activation(snetin),self.weights['s2h']) + np.dot(self.i_activation(nsnetin),self.weights['ns2h'])
        hnetin = self.lmda*(hnetin) + (1-self.lmda)*self.netin_buffer['h']
        
        if binarize:
            snetin[snetin!=0.0] = 1.0
            nsnetin[nsnetin!=0.0] = 1.0
            hnetin[hnetin!=0.0] = 1.0

        # update netin buffer
        self.netin_buffer['s'] = snetin
        self.netin_buffer['ns'] = nsnetin
        self.netin_buffer['h'] = hnetin

        # compute activation
        if binarize:
            sact = snetin
            nsact = nsnetin
            hact = hnetin
        else:
            sact = self.i_activation(snetin)
            nsact = self.i_activation(nsnetin)
            hact = self.h_activation(hnetin)

        return sact, nsact, hact
    
    def clean_buffer(self):
        self.netin_buffer = {
            's':np.zeros((1, self.state_size)),
            'ns':np.zeros((1, self.state_size)),
            'h':np.zeros((1, self.hidden_size))
        }
    
    def clone(self):
        x = AttractorNetwork(hidden_size=self.hidden_size, 
                             state_size=self.state_size, 
                             weight_scale=self.weight_scale,
                             i_tau=self.i_tau, 
                             h_tau=self.h_tau, 
                             h_C=self.h_C, 
                             lmda=self.lmda, 
                             ext=self.ext)
        x.weights = self.weights.copy()
        return x