''' Week 1: MP, MRP, MDP class definition
'''

import numpy as np
from collections import defaultdict

class MP:
    ''' Markov Process Class
    '''
    def __init__(self, states, P):
        '''Rk: The best datastructure for P would be a 
            sparse matrix, but we use np matrices for simplicity here
        ''' 
        self.states = states
        self.P = P
        self.nb_states = len(self.states)
    
        #Mapping of states and indexes
        self.state_to_ind = {}
        self.ind_to_state = {}
        for ind, s in enumerate(self.states):
            self.state_to_ind[s] = ind
            self.ind_to_state[ind] = s

    def comput_stationary_dist(self):
        op = (np.identity(self.nb_states) - self.P)
        aug_op = np.concatenate((op, np.zeros((1,self.nb_states))+1), axis = 0)
        
        b = np.concatenate((np.zeros((self.nb_states, 1)), np.zeros((1,1))+1))
        
        self.stationary_dist = np.linalg.solve((aug_op.T).dot(aug_op), (aug_op.T).dot(b))
        
    def get_stationary_dist(self):
        try:
            return self.stationary_dist
        except:
            self.comput_stationary_dist()
            return self.stationary_dist
            
        
class MRP(MP):
    def __init__(self, states, P, rewards, gamma = 0.99):
        super().__init__(states, P)
        self.rewards = rewards
        self.gamma = gamma
        
        try:
            assert(self.P.shape == self.rewards.shape)
        except:
            raise Exception('Transition and Rewards matrix should have same dimensions')

        self.precomputation()

    def precomputation(self):
        self.R = np.zeros(self.nb_states)
        for ind in self.ind_to_state.keys():
            self.R[ind] = self.P[ind].dot(self.rewards[ind])
        self.V = np.linalg.inv(np.identity(self.nb_states) - self.gamma*self.P).dot(self.R)
        
    def get_reward(self, state, next_state):
        return self.rewards[self.state_to_ind[state], self.state_to_ind[next_state]]
    
    def get_expected_reward(self, state):
        return self.R[self.state_to_ind[state]]
    
    def get_value_function(self, state):
        return self.V[self.state_to_ind[state]]
    

class MDP(MRP):
    def __init__(self, states, policies, rewards, actions, gamma = 0.99):
        self.states = states
        self.policies = policies
        self.rewards = rewards
        self.actions = actions
        self.gamma = gamma

    def get_related_MRP(self, policy):
        ''' Policy are mapping from space of states
            to a distribution of proba on states
            Policy rep = sto. matrix |S|*|S|
            Each lign is a proba of dist.
        '''
        if policy not in self.policies:
            raise Exception('Unkown policy')
            
        return MRP(self.states, self.policies[policy], self.rewards, self.gamma)

if __name__ == '__main__':
    #MRP example
    states = ['T','H', 'HH']
    P = np.array([[1/2, 1/2, 0], 
                  [1/2, 0, 1/2], 
                  [0, 0, 1]])
    
    rewards = np.array([[1, 1, 1], 
                        [1, 1, 1], 
                        [1, 1, 0]])
    
    my_MP = MP(states, P)
    my_MRP = MRP(states, P, rewards)
    
    actions = ['toss', 'stay']
    policies = {'toss': np.array([[1/2, 1/2, 0], [1/2, 0, 1/2], [0, 0, 1]]),
                'stay': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}
    
    my_MDP = MDP(states, policies, rewards, actions)
    toss_MRP = my_MDP.get_related_MRP('toss')
    
    print(toss_MRP.get_expected_reward('T'))
    print(toss_MRP.get_value_function('H'))
    print(toss_MRP.get_stationary_dist())
