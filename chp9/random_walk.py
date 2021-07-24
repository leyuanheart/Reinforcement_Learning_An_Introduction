# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:39:19 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter09/random_walk.py


考虑的都是线性函数近似

"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm


N_STATES = 1000

STATES = np.arange(1, N_STATES + 1)

START_STATE = 500

END_STATES = [0, N_STATES + 1]


LEFT = -1
RIGHT = 1
ACTIONS = [LEFT, RIGHT]


# maxium stride for an action
STEP_RANGE = 100


def compute_true_value():
    # a promising guess
    true_value = np.arange(-1001, 1003, 2) / 1001.0
    
    # Dynamic programming to find the true state values, based on the promising guess above
    # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
    while True:
        old_value = np.copy(true_value)
        for state in STATES:   # STATES中只包括non-terminal state
            true_value[state] = 0
            for action in ACTIONS:
                for step in range(1, STEP_RANGE + 1):
                    step *= action
                    next_state = state + step
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # asynchronous update for faster convergence
                    true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
                    
        error = np.sum(np.abs(old_value - true_value))
        
        if error < 1e-2:
            break
        
    # correct the state value for terminal states to 0
    true_value[0] = true_value[-1] = 0
    
    return true_value


# true_value = compute_true_value()
# np.save('true_value.npy', true_value)
true_value = np.load('true_value.npy')    
        
        
def step(state, action):
    '''
    take an @action at @state, return new state and reward for this transition
    '''
    
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
        
    return state, reward



def get_action():
    '''
    get an action, following random policy
    '''
    
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1


# ========= Different kinds of features ============================
class StateAggregation(object):
    '''
    a wrapper class for aggregation value function
    '''
    def __init__(self, num_groups):
        self.num_groups = num_groups
        self.group_size = N_STATES // num_groups
        
        # parameters w
        self.w = np.zeros(num_groups)
        
        
    def value(self, state):
        '''
        get the value of @state
        '''
        if state in END_STATES:
            return 0
        group_index = (state - 1) // self.group_size
        return self.w[group_index]

    
    def update(self, delta, state):
        '''
        update parameters
        @delta: step size * (target - old estimation)
        @state: state of current sample
        '''
        group_index = (state - 1) // self.group_size
        self.w[group_index] += delta
        


POLYNOMIAL_BASES = 0
FOURIER_BASES = 1
class BasesValueFunction(object):
    '''
    a wrapper class for polynomial / Fourier -based value function
    assume the state is a one-dim scalar
    '''
    def __init__(self, order, type):
        '''
        @order: # of bases, each function also has one more constant parameter (called bias in machine learning)
        @type: polynomial bases or Fourier bases
        '''
        self.order = order
        self.w = np.zeros(order + 1)
        
        # set up the bases function
        self.bases = []
        if type == POLYNOMIAL_BASES:
            for i in range(order + 1):
                self.bases.append(lambda s, i=i: pow(s, i))
        elif type == FOURIER_BASES:
            for i in range(order + 1):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))
                
    def value(self, state):
        # map the state space to [0, 1]
        state /= float(N_STATES)
        # get the feature vector
        feature = np.asarray([func(state) for func in self.bases])
        
        return np.dot(self.w, feature)
    
    
    def update(self, delta, state):
        # map the state space to [0, 1]
        state /= float(N_STATES)
        derivative = np.asarray([func(state) for func in self.bases])
        self.w += delta * derivative
    
    

class TileCoding(object):
    '''
    a wrapper class for tile coding value function
    '''
    def __init__(self, num_tilings, tile_width, tiling_offset):
        '''
        # @num_tilings: # of tilings
        # @tile_width: each tiling has several tiles, this parameter specifies the width of each tile
        # @tile_offset: specifies how tilings are put together
        '''
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.tiling_offset = tiling_offset
        
        # to make sure that each state is covered by same num of tiles
        # we need one more tile for each tiling
        self.tiling_size = N_STATES // tile_width + 1
        
        self.w = np.zeros((self.num_tilings, self.tiling_size))
        
        # For performance, only track the starting position for each tiling
        # As we have one more tile for each tiling, the starting position will be negative
        self.tilings = np.arange(-tile_width + 1, 0, tiling_offset)
        
    
    def value(self, state):
        state_value = 0.0
        # go through all the tilings
        for tiling_index in range(len(self.tilings)):
            # find the active tile in current tiling
            tile_index = (state - self.tilings[tiling_index]) // self.tile_width
            state_value += self.w[tiling_index, tile_index]
            
        return state_value
    
    def update(self, delta, state):
        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling(tile)
        delta /= self.num_tilings
        
        for tiling_index in range(len(self.tilings)):
            tile_index = (state - self.tilings[tiling_index]) // self.tile_width
            self.w[tiling_index, tile_index] += delta
    



# ================ Monte-Carlo or Temporal Difference =========================
def gradient_monte_carlo(value_function, alpha, distribution=None):
    '''
    gradient Monte Carlo algorithm
    @value_function: an instance of class ValueFunction
    @alpha: step size
    @distribution: array to store the distribution statistics
    '''
    state = START_STATE
    trajectory = [state]
    
    # we assume gamma = 1, so return is just the same as the latest reward
    reward = 0.0
    while state not in END_STATES:
        action = get_action()
        next_state, reward = step(state, action)
        trajectory.append(next_state)
        state = next_state
        
    # gradient update for each state in trajectory
    for state in trajectory[:-1]:  # 不更新terminal state
       delta = alpha * (reward - value_function.value(state))
       value_function.update(delta, state)
       if distribution is not None:
           distribution[state] += 1
        
    

def semi_gradient_temporal_difference(value_function, n, alpha):
    '''
    semi-gradient n-step TD algorithm
    @valueFunction: an instance of class ValueFunction
    @n: # of steps
    @alpha: step size
    '''
    state = START_STATE
    
    states = [state]
    rewards = [0]
    
    
    # track the time 
    t = 0
    
    # the length of the episode
    T = float('inf')
    while True:
        
        if t < T:
            action = get_action()
            next_state, reward = step(state, action)
            
            states.append(next_state)
            rewards.append(reward)
            
            if next_state in END_STATES:
                T = t + 1
                
        # get the time of the state to update
        tau = t + 1 - n
        if tau >= 0:
            returns = 0.0
            for i in range(tau + 1, min(T, tau + n) + 1):
                returns += rewards[i]
            if tau + n <= T:
                returns += value_function.value(states[tau + n])
            
            state_to_update = states[tau]
            
            if not state_to_update in END_STATES:
                delta = alpha * (returns - value_function.value(state_to_update))
                value_function.update(delta, state_to_update)
                
        if tau == T - 1:
            break
        state = next_state
        t += 1
    
    
    
    

# ===================== Plot ===================================================
def figure9_1(true_value):
    episodes = int(1e5)
    alpha = 2e-5
    
    # we have 10 aggregations in this example, each has 100 states
    value_function = StateAggregation(10)
    distribution = np.zeros(N_STATES + 2)
    for ep in tqdm(range(episodes)):
        gradient_monte_carlo(value_function, alpha, distribution)
        
    distribution /= np.sum(distribution)
    state_values = [value_function.value(i) for i in STATES]
    
    plt.figure(figsize=(10, 20))   # (width, height)
    plt.subplot(2, 1, 1)
    plt.plot(STATES, state_values, label='Approximate MC value')
    plt.plot(STATES, true_value[1:-1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(STATES, distribution[1:-1], label='State distribution')
    plt.xlabel('State')
    plt.ylabel('Distribution')
    plt.legend()
    
    # return state_values, distribution

# figure9_1(true_value)
        
        

def figure_9_2_left(true_value):
    episodes = int(1e5)
    alpha = 2e-4
    value_function = StateAggregation(10)
    for ep in tqdm(range(episodes)):
        semi_gradient_temporal_difference(value_function, 1, alpha)
        
    state_values = [value_function.value(i) for i in STATES]
    plt.plot(STATES, state_values, label='Approximate TD value')
    plt.plot(STATES, true_value[1:-1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()
    
    
# figure_9_2_left(true_value)


def figure_9_2_right(true_value):
    
    steps = np.power(2, range(10))
    
    alphas = np.arange(0, 1.1, 0.1)
    
    # each run has 10 episodes
    episodes = 10
    
    # perform 100 independent runs
    runs = 100
    
    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(runs)):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                # we have 20 aggregations in this example
                value_function = StateAggregation(20)
                for ep in range(episodes):
                    semi_gradient_temporal_difference(value_function, step, alpha)
                    # calculate the RMSE
                    state_value = np.asarray([value_function.value(i) for i in STATES])
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(state_value - true_value[1:-1], 2)) / N_STATES)
    
    # take average
    errors /= episodes * runs
    
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])  # truncate the error
    plt.legend()
    

# figure_9_2_right(true_value)       
                
def figure_9_2(true_value):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    figure_9_2_left(true_value)
    plt.subplot(1, 2, 2)
    figure_9_2_right(true_value)                    
                
# figure_9_2(true_value)    
 


   
# Figure 9.5, Fourier basis and polynomials    
def figure_9_5(true_value):
    runs = 10
    
    episodes = 5000
    
    
    orders = [5, 10, 20]
    alphas = [1e-4, 5e-5]
    labels = [['polynomial basis'] * 3, ['fourier basis'] * 3]
    
    
    errors = np.zeros((len(alphas), len(orders), episodes))
    for run in range(runs):
        for i in range(len(orders)):
            value_functions = [BasesValueFunction(orders[i], POLYNOMIAL_BASES), BasesValueFunction(orders[i], FOURIER_BASES)]
            for j in range(len(value_functions)):
                for episode in tqdm(range(episodes)):
                    gradient_monte_carlo(value_functions[j], alphas[j])
                    
                    state_values = [value_functions[j].value(state) for state in STATES]
                    
                    errors[j, i, episode] += np.sqrt(np.mean(np.power(true_value[1:-1] - state_values, 2)))
                    
                    
    errors /= runs
    
    
    for i in range(len(alphas)):
        for j in range(len(orders)):
            plt.plot(errors[i, j, :], label=f'{labels[i][j]} order = {orders[j]}')
        plt.xlabel('Episodes')
        # The book plots RMSVE, which is RMSE weighted by a state distribution
        plt.ylabel('RMSE')
        plt.legend()
    

# figure_9_5(true_value)



def figure_9_10(true_value):
    runs = 10
    
    episodes = 5000
    
    num_tilings = 50
    
    tile_width = 200
    
    tiling_offset = 4
    
    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']
  
    errors = np.zeros((len(labels), episodes))
    for run in range(runs):
        # initialize value functions for multiple tilings and single tiling
        value_functions = [TileCoding(num_tilings, tile_width, tiling_offset), StateAggregation(N_STATES // tile_width)]
        
        for i in range(len(value_functions)):
            for ep in tqdm(range(episodes)):
                # I use a changing alpha according to the episode instead of a small fixed alpha
                # With a small fixed alpha, I don't think 5000 episodes is enough for so many
                # parameters in multiple tilings.
                # The asymptotic performance for single tiling stays unchanged under a changing alpha,
                # however the asymptotic performance for multiple tilings improves significantly
                alpha = 1.0 / (ep + 1)
                # al1pha = 1e-4
                
                gradient_monte_carlo(value_functions[i], alpha)
                
                state_values = [value_functions[i].value(state) for state in STATES]
                
                errors[i][ep] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))
                
                
    errors /= runs
    plt.figure()
    for i in range(0, len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()        
                


# figure_9_10(true_value)












































































































