'''
Author: Raphael Abbou
'''

import numpy as np
from scipy.stats import norm

class Merton():
    def __init__(self, W, T, mu, vol, r, Util = (lambda x: -np.exp(-x)), eps = 0.1, delt = 0.01):
        self.W = W
        self.T = T
        self.mu = mu
        self.vol = vol
        self.r = r
        self.Util = Util
        self.eps = eps
        self.delt = delt
        
        #sum Xi has variance Tvol^2
        self.states = [(0, self.W)]
        self.s_ind = [0]
        self.s2ind = {0: (0, self.W)}
        self.ind2s = {(0, self.W): 0}
        s = 1
        self.L = max(0, W - 3*np.sqrt(self.T)*self.vol)
        self.U = W + 3*np.sqrt(self.T)*self.vol
        for t in range(1, T+1, 1):
            for i in np.arange(self.L, self.U, self.eps):
                self.states.append((t, i))
                self.s_ind.append(s)
                self.ind2s[s] = (t, i)
                self.s2ind[(t, i)] = s
                s += 1
                
        self.n = len(self.states)
    
        self.action2P = dict()
        self.action2rewards = dict()
        
        self.actions = []
        for pi in np.arange(0., 1 + self.delt, self.delt):
            for c in np.arange(0., 1 + self.delt, self.delt):
                self.actions.append((pi, c))
          
    def generate_transition(self, pi, c):
        P = np.zeros((self.n, self.n))
        for t in range(1, self.T, 1):
            for i in np.arange(self.L, self.U, self.eps):
                s = self.s2ind[(t, i)]
                for i_next in np.arange(self.L, self.U, self.eps):
                    s_next = self.s2ind[(t+1, i_next)]
                    if i_next + self.eps < self.U:
                        i_next_plus = i_next + self.eps
                    else:
                        i_next_plus = float('inf')
                    if pi > 0:
                        up = (i_next_plus - i*(1+self.r-c))/pi + self.r
                        low = (i_next - i*(1+self.r-c))/pi + self.r
                        P[s, s_next] = norm.cdf((up-self.mu)/self.vol) - norm.cdf((low-self.mu)/self.vol)
                    else:
                        if (i_next <= i*(1+self.r)) and (i*(1+self.r) < i_next_plus):
                            P[s, s_next] = 1
        
        #position at time T are absorbing states
        for i in np.arange(self.L, self.U, self.eps):
            s = self.s2ind[(self.T, i)]
            P[s, s] = 1
            
        self.action2P[(pi, c)] = P

    def generate_rewards(self, pi, c):
        rewards = np.zeros((self.n, self.n))
        for t in range(1, self.T, 1):
            for i in np.arange(self.L, self.U, self.eps):
                s = self.s2ind[(t, i)]
                for i_next in np.arange(self.L, self.U, self.eps):
                    s_next = self.s2ind[(t+1, i_next)]
                    rewards[s, s_next] = self.Util(c*i)
                    if t+1 == self.T:
                        rewards[s, s_next] += self.Util(i_next)
        
        self.action2rewards[(pi, c)] = rewards

    def generate(self):
        for pi in np.arange(0., 1 + self.delt, self.delt):
            for c in np.arange(0., 1 + self.delt, self.delt):
                self.generate_transition(pi, c)
                self.generate_rewards(pi, c)


    def random_step(self):
        np.random.normal(self.mu, self.vol)
    
    def discrete_rand_step(self):
        pass
        
    

if __name__ == "__main__":
    W = 100
    T = 10
    mu = 0.05
    vol = 0.2
    r = 0.03
    
    mer = Merton(W, T, mu, vol, r, eps = W/100)
    mer.generate()
    
    from markov_process import MP, MRP, MDP
    from policy import RandPolicy, DetPolicy
    from dp_algos import value_step, store_value_function, policy_evaluation, policy_iteration, value_iteration
    
    mdp = MDP(mer.s_ind, action2P = mer.action2P, action2rewards = mer.action2rewards, actions = mer.actions)
    
    mapping = {s: {(0.2, 0.1):1} for s in mdp.states}
    policy = DetPolicy(mer.action2P, mer.action2rewards, mapping)
    
    is_improved, opt_policy = value_iteration(mdp, policy)
