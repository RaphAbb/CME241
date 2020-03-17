'''
Author: Raphael Abbou
'''

import numpy as np
import os
os.chdir("../")

from my_utils.markov_process import MP, MRP, MDP
from my_utils.policy import RandPolicy, DetPolicy

from my_utils.frog_mdp import FrogMDP, f, generate_transitions_rewards, get_frog_mdp

from my_utils.rl import RLAlgorithm, FixedRLAlgorithm, simulate

class FrogRL(RLAlgorithm):
    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score
    
    def getAction(self, state):
        pass
    #When simulating an MDP, update parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState):
        pass
        
if __name__ == "__main__":
    n = 10
    mdp, a_policy = get_frog_mdp(n)
    rl = FixedRLAlgorithm(a_policy)
    start_state = n//2
    totalRewards = simulate(mdp, rl, start_state, numTrials=10, maxIterations=1000)
    print(totalRewards)
