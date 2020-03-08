''' 
Week 1: MP, MRP, MDP class definition
Author: Raphael Abbou
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

        self.R = np.zeros(self.nb_states)
        for state in self.states:
            self.R[state] = self.P[state].dot(self.rewards[state])
        
    def get_reward(self, state, next_state):
        return self.rewards[state, next_state]
    
    def get_R(self, state):
        return self.R[state]
 
    def get_V(self):
        try:
            return self.V
        except:
            self.V = np.linalg.inv(np.identity(self.nb_states) - self.gamma*self.P).dot(self.R)
            return self.V
    
    def get_state_V(self, state):
        try:
            return self.V[state]
        except:
            self.V = np.linalg.inv(np.identity(self.nb_states) - self.gamma*self.P).dot(self.R)
            return self.V[state]
    

class MDP():
    def __init__(self, states, action2P, rewards, actions, gamma = 0.99):
        self.states = states
        self.action2P = action2P
        self.rewards = rewards
        self.actions = actions
        self.gamma = gamma
        
        self.nb_states = len(self.states)
        
        self.mrps = dict() #maps policy ids to mrp
        
    def get_related_MRP(self, det_policy):
        try:
            return self.mrps[det_policy.id]
        except: 
            P = det_policy.get_P()
            self.mrps[det_policy.id] = MRP(self.states, P, self.rewards, self.gamma)
            return self.mrps[det_policy.id]

    def get_Qvalue(self, policy, state, action):
        mrp = self.get_related_MRP(policy)
        return mrp.get_R(state) + mrp.gamma*np.dot(mrp.P, mrp.V)[state]
            
    def get_Rvalue(self, state, action):
        return self.action2P[action][state].dot(self.rewards[state])
        
        
if __name__ == '__main__':
    #MRP example
    #states = ['T','H', 'HH']
    states = [0,1,2]
    P = np.array([[1/2, 1/2, 0], 
                  [1/2, 0, 1/2], 
                  [0, 0, 1]])
    
    rewards = np.array([[1, 1, 1], 
                        [1, 1, 1], 
                        [1, 1, 0]])
    
    my_MP = MP(states, P)
    my_MRP = MRP(states, P, rewards)
    
    my_MRP.get_V()
