'''
Author: Raphael Abbou
'''

import numpy as np
import os
os.chdir("../")

from my_utils.markov_process import MP, MRP, MDP
from my_utils.policy import RandPolicy, DetPolicy

class RWMDP(MDP):
    def __init__(self, n, states, action2P, action2rewards, actions):
        super().__init__(states, action2P, action2rewards, actions)
        self.n = n
        self.gamma = 1
        
    def IsEndState(self, state):
        ''' This method is supposed to be overriden
            For each particular instance of the MDP,
            in order to define End State
        '''
        if (state==self.n) or (state == 0):
            return True
        
def g(i, j, n):
    if (i==0):
        if (j==0):
            return 1
        else:
            return 0
    elif (i==n):
        if (j==n):
            return 1
        else:
            return 0
    elif i == j+1:
        return 1/2
    elif i == j-1:
        return 1/2
    else:
        return 0

def GetRWPolicyRewards(n):
    ''' Random Walk
    '''
    rewards = np.zeros((n+1, n+1))
    rewards[:,0] = 0
    rewards[:,n] = 1
    
    P = np.array([[g(i,j,n) for j in range(n+1)] for i in range(n+1)])
    
    action2P = {'walk': P}
    action2rewards = {'walk': rewards}
    
    mapping = {i: {'walk':1} for i in range(n)}
    
    
    RWPolicy = DetPolicy(action2P, action2rewards, mapping)
    
    return RWPolicy, rewards

def get_rw_mdp(n):
    policy, rewards = GetRWPolicyRewards(n)
    
    states = [i for i in range(n+1)]
    
    actions = ['walk']    
    mdp = RWMDP(n, states, policy.action2P, policy.action2rewards, actions)

    return mdp, policy

if __name__ == "__main__":
    n = 50
    mdp, policy = get_rw_mdp(n)
    print(mdp.action2rewards['walk'])
