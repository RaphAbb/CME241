'''
Author: Raphael Abbou
'''

import numpy as np
import os
os.chdir("../")

from my_utils.markov_process import MP, MRP, MDP
from my_utils.policy import RandPolicy, DetPolicy

class FrogMDP(MDP):
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

def f(i, j, n):
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
        return i/n
    elif i == j-1:
        return (n-i)/n
    else:
        return 0
    
def generate_transitions_rewards(n):
    P_A = np.array([[f(i,j,n) for j in range(n+1)] for i in range(n+1)])
    
    l1, l2, l = [0]*(n+1), [0]*(n+1), [1/n]*(n+1)
    l1[0] = 1
    l2[n] = 1
    
    P_B = np.array([[l1] + [l]*(n-2) + [l2]]).squeeze(0)
    for i in range(1, n, 1):
        P_B[i,i] = 0
    
    rewards = np.zeros((n+1, n+1))
    rewards[:,0] = 0
    rewards[:,n] = 1
    
    return P_A, P_B, rewards

def get_frog_mdp(n):
    P_A, P_B, rewards = generate_transitions_rewards(n)
    
    states = [i for i in range(n+1)]
    
    actions = ['A', 'B']
    
    actions = ['toss', 'stay']
    action2P = {'A': P_A,
                'B': P_B}
    action2rewards = {'A': rewards,
                      'B': rewards}
    
    mapping = {i: {'A':1} for i in range(n)}
    
    
    a_policy = DetPolicy(action2P, action2rewards, mapping)
    
    mdp = FrogMDP(n, states, action2P, action2rewards, actions)

    return mdp, a_policy

if __name__ == "__main__":
    n = 10
    mdp, a_policy = get_frog_mdp(n)