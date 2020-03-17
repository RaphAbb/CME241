'''
Author: Raphael Abbou
'''

import numpy as np
from collections import defaultdict
import os
os.chdir("../")

from my_utils.rl import RLAlgorithm, FixedRLAlgorithm, simulate

class MCRL(FixedRLAlgorithm):
    def __init__(self, detpolicy, states, gamma):
        super().__init__(detpolicy)
        self.states = states
        self.RewardSequence = []
        self.gamma = gamma
        
    def incorporateFeedback(self, state, action, reward, newState):
        self.RewardSequence.append((state, reward))
    
    def reset(self):
        self.RewardSequence = []
        
def GetValueMC(mdp, rl: MCRL, nIter=20):
    ''' Every time step update
    '''
    N = 1
    V = defaultdict(int)
    N = defaultdict(int)
    
    #for MC convergence Plot
    #Maps states to histo of V values
    Vhisto = defaultdict(list)
    
    for i in range(nIter):
        for start_state in mdp.states:
            rl.reset()
            totalRewards = simulate(mdp, rl, start_state, numTrials=1)
            G_t = 0
            totalDiscount = 1
            #We go backward in the sequence of rewards
            rl.RewardSequence.reverse()
            for state, R_t in rl.RewardSequence:
                G_t += R_t*totalDiscount
                totalDiscount *= rl.gamma
                N[state] += 1
                V[state] += G_t
                Vhisto[state].append(1/N[state]*V[state])

    for state in N.keys():
        V[state] *= 1/N[state]
        
    return V, Vhisto


if __name__ == "__main__":
    from my_utils.markov_process import MP, MRP, MDP
    from my_utils.policy import RandPolicy, DetPolicy
    
    from my_utils.frog_mdp import FrogMDP, f, generate_transitions_rewards, get_frog_mdp
    
    n = 10
    mdp, a_policy = get_frog_mdp(n)
    rl = MCRL(a_policy, mdp.states, mdp.gamma)
    
    V, Vhisto= GetValueMC(mdp, rl, nIter=50)
    print(V)
    
    import matplotlib.pyplot as plt
    state = n//2
    plt.plot([i for i in range(len(Vhisto[state]))], Vhisto[state])
    plt.xlabel('Number of Updates')
    plt.ylabel('Value Function')
    plt.title('Monte Carlo A-Policy Evaluation for state {0:.0}'.format(state))
    plt.show()
