# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:45:52 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter07/random_walk.py
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm


# all states
N_STATES = 19

# discount
GAMMA = 1

# all states but terminal states
STATES = np.arange(1, N_STATES + 1)

# start from the middle state
START_STATE = 10

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state value from bellman equation
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0



def n_step_temporal_difference(value, n, alpha):
    '''
    
    Parameters
    ----------
    value : array
        value function
    n : int
        n-step for returns
    alpha : float
        learning rate

    Returns
    -------
    updated value function

    '''
    state = START_STATE
    
    
    # arrays to store states and rewards for an episode
    # space isn't a major consideration, so I didn't use the mod trick ???
    states = [state]
    rewards = [0]
    
    # track time step
    t = 0
    
    # refer Page 144 of "Reinforcement Learning--An Introduction"
    T = float('inf')
    while True:
        if t < T:
            # random walk
            if np.random.binomial(1, 0.5) == 1:
                next_state = state + 1
            else:
                next_state = state - 1
            
            # get the reward
            if next_state == 0:
                reward = -1
            elif next_state == 20:
                reward = 1
            else:
                reward = 0
                
            # get the time of the state to update
            states.append(next_state)
            rewards.append(reward)
            
            if next_state in END_STATES:
                T = t + 1
                
        
        # get the time of the state to update
        tau = t + 1 - n
        if tau >= 0:
            returns = 0.0
            # calculate the n-step return
            for i in range(tau + 1, min(T, tau + n) + 1):
                returns += pow(GAMMA, i - tau - 1) * rewards[i]
            if tau + n <= T:
                returns += pow(GAMMA, n) * value[states[tau + n]]
                
            state_to_update = states[tau]
            
            if not state_to_update in END_STATES:
                value[state_to_update] += alpha * (returns - value[state_to_update])
                
        if tau == T - 1:
            break
        
        state = next_state
        t += 1
        
        
    # return value    
    
                



def figure_7_2():
    
    # all possible steps
    steps = np.power(2, np.arange(0, 10))

    # all possible alphas
    alphas = np.arange(0, 1.1, 0.1)

    # perform 100 independent runs
    runs = 100
    
    # each run has 10 episodes
    episodes = 10
    
    
    
    # track the error for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(runs)):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                value = np.zeros(N_STATES + 2)
                for ep in range(episodes):
                    n_step_temporal_difference(value, step, alpha)
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(value - TRUE_VALUE, 2)) / N_STATES)
                    
    # take average
    errors /= runs * episodes
    
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label=f'n = {steps[i]}')
    plt.xlabel('alpha')
    plt.ylabel('Average RMS error over 19 states and first 10 episodes')
    plt.ylim([0.25, 0.8])
    plt.legend()
                
            
    
figure_7_2()























