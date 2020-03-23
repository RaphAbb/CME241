'''
Author: Raphael Abbou
'''

import numpy as np
from collections import defaultdict
import os
os.chdir("../")

from my_utils.rl import RLAlgorithm, FixedRLAlgorithm, simulate
from my_utils.montecarlo import MCRL, GetValueMC

class TDLambda(MCRL):
    ''' Td Lambda stopped at time T
        T = float('inf') --> Full episode
    '''
    def incorporateFeedback(self, state, action, reward, newState):
        self.RewardSequence.append((state, reward, newState))
    
def GetValueTDLambda(mdp, rl: MCRL, lbda = 0.95, online = True, alpha = 0.01, max_T = 10000, nIter=20):
    V = defaultdict(int)
    
    #for convergence Plot
    #Maps states to histo of V values
    Vhisto = defaultdict(list)
    
    for i in range(nIter):
        for start_state in mdp.states:
            rl.reset()
            totalRewards = simulate(mdp, rl, start_state, numTrials=1, maxIterations = max_T)
            if not(online):
                episodeUpdate = defaultdict(int)
            
            T = len(rl.RewardSequence)
            for t in range(T-1):
                curSumReward = 0
                G_t_lbda = 0
                totalDiscount = 1
                scaling = (1-lbda)/lbda
                state_t = rl.RewardSequence[t][0]
                n = 0
                for state, R_t, nextState in rl.RewardSequence[t+1:]:
                    n += 1
                    if n < len(rl.RewardSequence)-1:
                        scaling *= lbda
                    else:
                        scaling = lbda**(T-t-1) #T = n for the last step
                    curSumReward += R_t*totalDiscount
                    G_t_n = curSumReward + totalDiscount*rl.gamma*V[nextState]
                    totalDiscount *= rl.gamma
                    G_t_lbda += scaling*G_t_n
                
                if online:
                    V[state_t] += + alpha*(G_t_lbda - V[state_t])
                    Vhisto[state_t].append(V[state_t])
                else:
                    episodeUpdate[state_t] += alpha*(G_t_lbda - V[state_t])
            
            if not(online):
                for state, val in episodeUpdate.items():
                    V[state] += val
                    Vhisto[state].append(V[state])
                
    
    return V, Vhisto

def decay(E_t, lbda, gamma):
    for s in E_t.keys():
        E_t[s] *= gamma*lbda

def GetValueBackwardTDLambda(mdp, rl: MCRL, lbda = 0.97, online = True, alpha = 0.01, max_T = 10000, nIter=20):
    V = defaultdict(int)
    
    #for convergence Plot
    #Maps states to histo of V values
    Vhisto = defaultdict(list)
    
    for i in range(nIter):
        for start_state in mdp.states:
            rl.reset()
            totalRewards = simulate(mdp, rl, start_state, numTrials=1, maxIterations = max_T)
            #Eligibility Trace
            E_t = defaultdict(float)
            
            if not(online):
                episodeUpdate = defaultdict(int)
            
            T = len(rl.RewardSequence)
            totalDiscount = 1
            for t in range(T):
                decay(E_t, lbda, rl.gamma)
                
                state_t, r_t, next_state = rl.RewardSequence[t]
                E_t[state_t] += 1
                delta_t = r_t + rl.gamma*V[next_state] - V[state_t]
                
                if online:
                    V[state_t] += alpha*delta_t*E_t[state_t] 
                    Vhisto[state_t].append(V[state_t])
                else:
                    episodeUpdate[state_t] += alpha*delta_t*E_t[state_t]
            
            if not(online):
                for state, val in episodeUpdate.items():
                    V[state] += val
                    Vhisto[state].append(V[state])
                
    
    return V, Vhisto

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    from my_utils.markov_process import MP, MRP, MDP
    from my_utils.policy import RandPolicy, DetPolicy
    
    #from my_utils.frog_mdp import FrogMDP, f, generate_transitions_rewards, get_frog_mdp, g, GetRWPolicy
    from my_utils.randomwalk import RWMDP, g, GetRWPolicyRewards, get_rw_mdp
    
    n = 10
    mdp, policy = get_rw_mdp(n)

    rl = TDLambda(policy, mdp.states, mdp.gamma)
    
    V, Vhisto= GetValueTDLambda(mdp, rl, online = True, nIter=500)
    print(V)

#    V, Vhisto = GetValueBackwardTDLambda(mdp, rl, lbda = 0.99, online = True, alpha = 0.005, nIter=500)
#    print(V)
    
    state = n//2
    plt.plot([i for i in range(len(Vhisto[state]))], Vhisto[state])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel('Number of Updates')
    plt.ylabel('Value Function')
    plt.title('TD(lambda) A-Policy Evaluation for state {0}'.format(state))
    plt.show()
    
    