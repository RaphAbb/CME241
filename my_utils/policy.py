'''
Author: Raphael Abbou
'''

import numpy as np
import uuid

class RandPolicy():
    
    def __init__(self, action2P, action2rewards, mapping, id_ = uuid.uuid4()):
        self.action2P = action2P
        self.action2rewards = action2rewards
        self.mapping = mapping
        self.id = id_

    def get_action(self, state):
        actions = []
        probas = []
        for key, val in self.mapping[state].items():
            actions.append(key)
            probas.append(val)
        idx = np.random.choice(len(actions), 1, probas)[0]
        return actions[idx]

    def update(self, state, state_policy):
        #state_policy must be a mapping action -> proba
        self.mapping[state] = state_policy
        
    def deepcopy(self):
        return RandPolicy(self.action2P, self.action2rewards, self.mapping, uuid.uuid4())
    
class DetPolicy(RandPolicy):
    
    def get_P(self):
        temp = []
        for state in self.mapping.keys():
            a = self.get_action(state)
            temp.append(self.action2P[a][state])
        return np.array(temp)

    def get_rewards(self):
        temp = []
        for state in self.mapping.keys():
            a = self.get_action(state)
            temp.append(self.action2rewards[a][state])
        return np.array(temp)

    def __getitem__(self, state):
        #same as get_action in this case, but more handy
        return list(self.mapping[state].keys())[0]
    
    def deepcopy(self):
        return DetPolicy(self.action2P, self.action2rewards, self.mapping, uuid.uuid4())
    
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
    action2rewards = {a: rewards for a in actions}
    
    Rmapping = {0: {'toss': 1/3, 'stay': 2/3},
               1: {'toss': 2/3, 'stay': 1/3},
               2: {'toss': 1/3, 'stay': 2/3}
              }

    mapping = {0: {'toss':1},
               1: {'stay':1},
               2: {'toss':1}
              }
    
    policy = DetPolicy(action2P, action2rewards, mapping)
    policy.get_P()
    
    my_MDP = MDP(states, action2P, action2rewards, actions)
    toss_MRP = my_MDP.get_related_MRP(policy)
    
    print(toss_MRP.get_R(0))