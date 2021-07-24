# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:03:43 2021

@author: leyuan

reference: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/maximization_bias.py
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm


STATE_A = 0
STATE_B = 1

STATE_TERMINAL = 2 # use one terminal state

STATE_START = STATE_A

# possible action in A
RIGHT = 0
LEFT = 1


# possible action in B (assume "many" to be 10)
ACTIONS_B = range(10)


# all possible actions    Note: different state has different action set 
STATE_ACTIONS = [[RIGHT, LEFT], ACTIONS_B]

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.1

# discount for max value
GAMMA = 1.0


# state action pair values, if a state is a terminal state, then the value is always 0
INITIAL_Q = [np.zeros(2), np.zeros(len(ACTIONS_B)), np.zeros(1)]

# set up destination for each state and each action
TRANSITION = [[STATE_TERMINAL, STATE_B], [STATE_TERMINAL] * len(ACTIONS_B)]



def epsilon_greedy(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(STATE_ACTIONS[state])
    else:
        q_val = q_value[state]
        action = np.random.choice(np.where(q_val == np.max(q_val))[0])
        
    return action



def reward(state, action):
    if state == STATE_A:
        return 0
    return np.random.normal(-0.1, 1)



def q_learning(q1, q2=None):
    '''
    if there are two state action pair value array, use double Q-Learning
    otherwise use normal Q-Learning
    '''
    state = STATE_START
    
    # track the # of action LEFT in state A
    left_count = 0
    while state != STATE_TERMINAL:
        if q2 is None:
            action = epsilon_greedy(state, q1)
        else:
            action = epsilon_greedy(state, [item1 + item2 for item1, item2 in zip(q1, q2)])
            
        if state == STATE_A and action == LEFT:
            left_count += 1
        
        r = reward(state, action)
        next_state = TRANSITION[state][action]
        
        if q2 is None:
            active_q = q1
            target = r + GAMMA * np.max(active_q[next_state])
        else:
            if np.random.binomial(1, 0.5) == 1:
                active_q = q1
                target_q = q2
            else:
                active_q = q2
                target_q = q1
        
            best_action = np.random.choice(np.where(active_q[next_state] == np.max(active_q[next_state]))[0])
            target = r + GAMMA * target_q[next_state][best_action]
            
        
        # Q-learning update
        active_q[state][action] += ALPHA * (target - active_q[state][action])
        
        state = next_state
        
    return left_count
    
    

# Figure 6.5, 1,000 runs may be enough, # of actions in state B will also affect the curves
def figure_6_5(runs=1000, episodes=300):
    
    left_counts_q = np.zeros(episodes)
    left_counts_double_q = np.zeros(episodes)
    
    for run in tqdm(range(runs)):
        q = copy.deepcopy(INITIAL_Q)
        q1 = copy.deepcopy(INITIAL_Q)
        q2 = copy.deepcopy(INITIAL_Q)
        
        for ep in range(episodes):
            left_counts_q[ep] += q_learning(q)
            left_counts_double_q[ep] += q_learning(q1, q2)
            
    
    left_counts_q /= runs
    left_counts_double_q /= runs
    
    plt.figure()
    plt.plot(left_counts_q, label='Q-Learning')
    plt.plot(left_counts_double_q, label='Double Q-Learning')
    plt.plot(np.ones(episodes) * 0.05, label='Optimal')
    plt.xlabel('episodes')
    plt.ylabel('% left actions from A')
    plt.legend()


figure_6_5(runs=1000, episodes=300)





































