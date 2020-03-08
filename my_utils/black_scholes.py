import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats

import os
os.chdir("../")

from my_utils.newton_method import newton

def get_d1(S, K, T, vol, r, D):
    return (np.log(S/K) + (r + vol**2/2)*T)/(vol*np.sqrt(T))
 
def get_d2(S, K, T, vol, r, D):
    return (np.log(S/K) + (r - vol**2/2)*T)/(vol*np.sqrt(T))


def price_call(S, K, T, vol, r, D = 0):
    d1 = get_d1(S, K, T, vol, r, D)
    d2 = get_d2(S, K, T, vol, r, D)
    
    return S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)

def price_put(S, K, T, vol, r, D = 0):
    d1 = get_d1(S, K, T, vol, r, D)
    d2 = get_d2(S, K, T, vol, r, D)
    
    return K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)

def put_call_parity(S, K, T, vol, r):
    assert(price_put(S, K, T, vol, r) + S == price_call(S, K, T, vol, r) + K*np.exp(-r*T))

def hedge_ratio(S, K, T, vol, r, D = 0):
    d1 = get_d1(S, K, T, vol, r, D)
    d2 = get_d2(S, K, T, vol, r, D)
    
    return stats.norm.cdf(d1) + 1/np.sqrt(2*np.pi)*1/(S*vol*np.sqrt(T))*\
                (S*np.exp(-d1**2/2) - K*np.exp(-r*T)*np.exp(-d2**2/2))

def vega(S, K, T, vol, r, D = 0):
    d1 = get_d1(S, K, T, vol, r, D)
    return S*np.sqrt(T)*stats.norm.cdf(d1)
    
if __name__ == "__main__":
    ### Problem 2
    #############
    
    ## Simple Computation
    S = 3200
    K = 3300
    T = 0.5
    vol = 0.13
    r = 0.02
    
    print("Call price: ${0:.2f}".format(price_call(S, K, T, vol, r)))
    print("Put price: ${0:.2f}".format(price_put(S, K, T, vol, r)))
    
    put_call_parity(S, K, T, vol, r)
    
    ## Price variation with respect to the volatility
    vols = np.arange(0.1, 0.31, 0.01)
    c_prices = [price_call(S,K,T,vol,r) for vol in vols]
    plt.plot(vols, c_prices)
    plt.xlabel('Volatility')
    plt.ylabel('Call Value')
    plt.title('Call Price: Volatility Dependency')
    plt.show()

    ## Hedge Ratio as function of S
    S_values = np.arange(3300*0.8, 3300*1.2, 10)
    delta_values = [hedge_ratio(S_val,K,1/12,vol,r) for S_val in S_values]
    plt.plot(S_values, delta_values)
    plt.xlabel('Stock Value S')
    plt.ylabel('Hedge Ratio Delta')
    plt.title('Hedge Ratio for a Call: Spot Dependency')
    plt.show()

    strikes = np.arange(2500, 4000)
    p_prices = [price_put(S,K,T,vol,r)-87.97 for K in strikes]
    plt.plot(strikes, p_prices)
    plt.xlabel('Strikes')
    plt.ylabel('Put Value')
    plt.title('Put Price: Strike Dependency')
    plt.show()
    
    
    
    