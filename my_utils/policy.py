'''
Author: Raphael Abbou
'''

import numpy as np
import uuid

class RandPolicy():
    
    def __init__(self, action2P, mapping, id_ = uuid.uuid4()):
        self.action2P = action2P
        self.mapping = mapping
        self.id = id_

    def get_action(self, state):
        actions = []
        probas = []
        for key, val in self.mapping[state].items():
            actions.append(key)
            probas.append(val)
        return np.random.choice(actions, 1, probas)[0]

    def update(self, state, state_policy):
        #state_policy must be a mapping action -> proba
        self.mapping[state] = state_policy
        
    def deepcopy(self):
        return RandPolicy(self.action2P, self.mapping, uuid.uuid4())
    
class DetPolicy(RandPolicy):
    
    def get_P(self):
        temp = []
        for state in self.mapping.keys():
            a = self.get_action(state)
            temp.append(self.action2P[a][state])
        return np.array(temp)

    def deepcopy(self):
        return DetPolicy(self.action2P, self.mapping, uuid.uuid4())
    
if __name__ == '__main__':
    from markov_process import MP, MRP, MDP
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
    
    
    
    actions = ['toss', 'stay']
    action2P = {'toss': np.array([[1/2, 1/2, 0], [1/2, 0, 1/2], [0, 0, 1]]),
                'stay': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}
    
    Rmapping = {0: {'toss': 1/3, 'stay': 2/3},
               1: {'toss': 2/3, 'stay': 1/3},
               2: {'toss': 1/3, 'stay': 2/3}
              }

    mapping = {0: {'toss':1},
               1: {'stay':1},
               2: {'toss':1}
              }
    
    policy = DetPolicy(action2P, mapping)
    policy.get_P()
    
    my_MDP = MDP(states, action2P, rewards, actions)
    toss_MRP = my_MDP.get_related_MRP(policy)
    
    print(toss_MRP.get_R(0))