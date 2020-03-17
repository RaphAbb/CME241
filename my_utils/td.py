'''
Author: Raphael Abbou
'''

import numpy as np
from collections import defaultdict
import os
os.chdir("../")

from my_utils.rl import RLAlgorithm, FixedRLAlgorithm, simulate

class TD(FixedRLAlgorithm):
    def __init__(self, detpolicy, alpha, states, gamma):
        super().__init__(detpolicy)
        self.alpha = alpha
        self.states = states
        self.gamma = gamma
        self.V = defaultdict(int)
        self.Vhisto = defaultdict(list)
        
    def incorporateFeedback(self, state, action, reward, newState):
        self.V[state] += self.alpha*(reward + self.gamma*self.V[newState] - self.V[state])
        self.Vhisto[state].append(self.V[state])
    
        
def GetValueTD(mdp, rl: TD, nIter=20):
    ''' Every time step update
    '''
    N = 1
    V = defaultdict(int)
    N = defaultdict(int)
    
    for start_state in mdp.states:
        totalRewards = simulate(mdp, rl, start_state, numTrials=nIter)
        
    return rl.V, rl.Vhisto

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    from my_utils.markov_process import MP, MRP, MDP
    from my_utils.policy import RandPolicy, DetPolicy
    
    from my_utils.frog_mdp import FrogMDP, f, generate_transitions_rewards, get_frog_mdp
    
    n = 10
    mdp, a_policy = get_frog_mdp(n)
    
    alphas = [0.05, 0.15, 0.3, 0.5]
    for alpha in alphas:
        rl = TD(a_policy, alpha, mdp.states, mdp.gamma)
        
        V, Vhisto= GetValueTD(mdp, rl, nIter=50)
        print(V)
        
        state = n//2
        plt.plot([i for i in range(len(Vhisto[state]))], Vhisto[state], label="alpha: {0:.2}".format(alpha))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel('Number of Updates')
    plt.ylabel('Value Function')
    plt.title('Monte Carlo A-Policy Evaluation for state {0:.0}'.format(state))
    plt.show()