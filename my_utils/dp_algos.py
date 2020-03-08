'''
Author: Raphael Abbou
'''

import numpy as np

def value_step(mrp, V):
    V_next = mrp.R + mrp.gamma*np.dot(mrp.P, V).squeeze()
    return V_next
    
def store_value_function(mrp, tol=10**-7, max_iter=10**5):
    V_prev = np.zeros((mrp.nb_states,1))
    V = value_step(mrp, V_prev)
    
    my_iter = 0
    while np.sum(np.square(V - V_prev)) > tol:
        V_prev = V
        V = value_step(mrp, V)
        my_iter += 1
        if my_iter > max_iter:
            print("Maximum iteration exceed: convergence failed")
            mrp.V
    
    print("Policy Evaluation converged in {0} steps".format(my_iter))
    mrp.V = V

def policy_evaluation(mdp, policy, tol=10**-7, max_iter=10**5):
    mrp = mdp.get_related_MRP(policy)
    return mrp.V

def policy_iteration(mdp, policy, tol=10**-2):
    new_policy = policy.deepcopy()
    mrp = mdp.get_related_MRP(new_policy)
    store_value_function(mrp)
    
    is_improved = False
    
    for state in mdp.states:
        curr_Vvalue = mrp.get_state_V(state)
        for action in mdp.actions:
            qvalue = mdp.get_Qvalue(new_policy, state, action)
            #if qvalue > curr_Vvalue:
            if qvalue - curr_Vvalue > tol:
                curr_Vvalue = qvalue
                
                #update the policy matrix with related transition prba
                new_policy.update(state, {action: 1.})
                
                is_improved = True
            
    return is_improved, new_policy


def value_iteration(mdp, policy):
    new_policy = policy.deepcopy()
    mrp = mdp.get_related_MRP(new_policy)
    
    V = np.zeros((mdp.nb_states,1))
    mrp.V = V
    is_improved = False
    
    for state in mdp.states:
        curr_Vvalue = mrp.get_state_V(state)
        for action in mdp.actions:
            temp_estimate = mdp.get_Rvalue(state, action) + mdp.gamma*(mdp.action2P[action].dot(V))[state]
            if temp_estimate > curr_Vvalue:
                V[state] = temp_estimate
                new_policy.update(state, {action: 1.})
                is_improved = True
    
    return is_improved, new_policy

    
if __name__ == "__main__":
    from markov_process import MP, MRP, MDP
    from policy import RandPolicy, DetPolicy
    
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
    
    mdp = MDP(states, action2P, rewards, actions)
    toss_MRP = mdp.get_related_MRP(policy)
    
    is_improved, P = policy_iteration(mdp, policy)
    is_improved, P = value_iteration(mdp, policy)