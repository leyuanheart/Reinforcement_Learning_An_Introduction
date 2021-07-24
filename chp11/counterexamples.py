# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 14:50:59 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter11/counterexample.py
"""



import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm



# ==================== Baird counterexample ===================================

# 0-5 are upper states, 6 is lower state
STATES = np.arange(0, 7)
LOWER_STATE = 6
# discount factor
DISCOUNT = 0.99


# each state is represented by a vector of length 8
FEATURE_SIZE = 8
FEATURES = np.zeros((len(STATES), FEATURE_SIZE))
for i in range(LOWER_STATE):
    FEATURES[i, i] = 2
    FEATURES[i, 7] = 1
FEATURES[LOWER_STATE, 6] = 1
FEATURES[LOWER_STATE, 7] = 2

# all possible actions
DASHED = 0
SOLID = 1
ACTIONS = [DASHED, SOLID]

REWARD = 0



def step(state, action):
    if action == SOLID:
        return LOWER_STATE
    return np.random.choice(STATES[:LOWER_STATE])



def target_policy(state):
    return SOLID


BEHAVIOR_SOLID_PROBABILITY = 1.0 / 7
def behavior_policy(state):
    if np.random.binomial(1, BEHAVIOR_SOLID_PROBABILITY) == 1:
        return SOLID
    return DASHED


# state distribution for the behavior policy
STATE_DISTRIBUTION = np.ones(len(STATES)) / 7
STATE_DISTRIBUTION_MAT = np.matrix(np.diag(STATE_DISTRIBUTION))
# projection matrix for minimize MSVE: PI = X (X^TDX)^{-1}X^TD
PROJECTION_MAT = np.matrix(FEATURES) * \
                 np.linalg.pinv(np.matrix(FEATURES.T) * STATE_DISTRIBUTION_MAT * np.matrix(FEATURES)) * \
                 np.matrix(FEATURES.T) * \
                 STATE_DISTRIBUTION_MAT



def semi_gradient_off_policy_TD(state, w, alpha):
    action = behavior_policy(state)
    next_state = step(state, action)
    # get the importance ratio
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1. / BEHAVIOR_SOLID_PROBABILITY
        
    delta = REWARD + DISCOUNT * np.dot(FEATURES[next_state, :], w) - np.dot(FEATURES[state, :], w)
    delta *= rho * alpha
    # update the parameter
    w += FEATURES[state, :] * delta
    
    return next_state
    

def semi_gradient_DP(w, alpha):
    delta = 0.0
    # go through all the states
    for state in STATES:
        expected_return = 0.0
        # compute bellman error for each state
        for next_state in STATES:
            if next_state == LOWER_STATE:
                expected_return += REWARD + DISCOUNT *  np.dot(FEATURES[next_state, :], w)
        bellman_error = expected_return - np.dot(FEATURES[state, :], w)
        
        # accumulate gradients
        delta += bellman_error * FEATURES[state, :]
    
    w += alpha / len(STATES) * delta
    
    return next_state



def TDC(state, w, v, alpha, beta):
    action = behavior_policy(state)
    next_state = step(state, action)
    # get the importance ratio
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY

    delta = REWARD + DISCOUNT * np.dot(FEATURES[next_state, :], w) - np.dot(FEATURES[state, :], w)
    
    
    w += alpha * rho * (delta * FEATURES[state, :] - DISCOUNT * FEATURES[next_state, :] * np.dot(FEATURES[state, :], v))   # 这里一开始写成了w, 比较容易出错，要注意
    v += beta * rho * (delta - np.dot(FEATURES[state, :], v)) * FEATURES[state, :]
    
    return next_state




def expected_TDC(w, v, alpha, beta):
    for state in STATES:
        # When computing expected update target, if next state is not lower state, importance ratio will be 0,
        # so we can safely ignore this case and assume next state is always lower state
        delta = REWARD + DISCOUNT * np.dot(FEATURES[LOWER_STATE, :], w) - np.dot(FEATURES[state, :], w)
        rho = 1 / BEHAVIOR_SOLID_PROBABILITY
        # Under behavior policy, state distribution is uniform, so the probability for each state is 1.0 / len(STATES)
        expected_update_w = BEHAVIOR_SOLID_PROBABILITY * rho * (
            delta * FEATURES[state, :] - DISCOUNT * FEATURES[LOWER_STATE, :] * np.dot(v, FEATURES[state, :]))   # 同理，这里也要注意
        w += alpha / len(STATES) * expected_update_w
        
        expected_update_v = BEHAVIOR_SOLID_PROBABILITY * rho * (delta - np.dot(v, FEATURES[state, :])) * FEATURES[state, :]
        v += beta / len(STATES) * expected_update_v
        
        

# interest is 1 for every state
INTEREST = 1
def expected_emphatic_TD(w, m, alpha):
    expected_update_w = 0
    expected_next_m = 0.0
    
    for state in STATES:
        # compute rho(t-1)
        if state == LOWER_STATE:
            rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
        else:
            rho = 0
            
        # update the emphasis
        next_m = DISCOUNT * rho * m + INTEREST
        expected_next_m += next_m
        # When computing expected update target, if next state is not lower state, importance ratio will be 0,
        # so we can safely ignore this case and assume next state is always lower state2
        delta = REWARD + DISCOUNT * np.dot(FEATURES[LOWER_STATE, :], w) - np.dot(FEATURES[state, :], w)
        expected_update_w += BEHAVIOR_SOLID_PROBABILITY * next_m * 1.0 / BEHAVIOR_SOLID_PROBABILITY * delta * FEATURES[state, :]
    
    w += alpha / len(STATES) * expected_update_w
    
    return expected_next_m / len(STATES)

    
   
    
    


# true value function is always 0 in this example
def compute_RMSVE(w):
    return np.sqrt(np.dot(np.power(np.dot(FEATURES, w), 2), STATE_DISTRIBUTION))


def compute_RMSPBE(w):
    bellman_error = np.zeros(len(STATES))
    for state in STATES:
        for next_state in STATES:
            if next_state == LOWER_STATE:
                bellman_error[state] += REWARD + DISCOUNT *  np.dot(FEATURES[next_state, :], w) - np.dot(FEATURES[state, :], w)
    bellman_error = np.dot(np.asarray(PROJECTION_MAT), bellman_error)
    
    return np.sqrt(np.dot(np.power(bellman_error, 2), STATE_DISTRIBUTION))





def figure_11_2_left():
    # initialization
    w = np.ones(FEATURE_SIZE)
    w[6] = 10
    
    alpha = 0.01
    
    steps = 1000
    ws = np.zeros((FEATURE_SIZE, steps))
    state = np.random.choice(STATES)
    
    for step in tqdm(range(steps)):
        state = semi_gradient_off_policy_TD(state, w, alpha)
        ws[:, step] = w
    
    # plt.figure()
    for i in range(FEATURE_SIZE):
        plt.plot(ws[i, :], label='w' + str(i+1))
    plt.xlabel('Steps')
    plt.ylabel('Weights')
    plt.title('semi-gradient off-policy TD')
    plt.legend()
    
    
# figure_11_2_left()


def figure_11_2_right():
    w = np.ones(FEATURE_SIZE)
    w[6] = 10
    
    alpha = 0.01
    
    sweeps = 1000
    ws = np.zeros((FEATURE_SIZE, sweeps))
    for sweep in tqdm(range(sweeps)):
        semi_gradient_DP(w, alpha)
        ws[:, sweep] = w

    # plt.figure()
    for i in range(FEATURE_SIZE):
        plt.plot(ws[i, :], label='w' + str(i+1))
    plt.xlabel('Steps')
    plt.ylabel('Weights')
    plt.title('semi-gradient DP')
    plt.legend()


# figure_11_2_right()

def figure_11_2():
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    figure_11_2_left()
    plt.subplot(1, 2, 2)
    figure_11_2_right()

figure_11_2()




def figure_11_5_left():
    # initialization
    w = np.ones(FEATURE_SIZE)
    w[6] = 10
    v = np.zeros(FEATURE_SIZE)
    
    alpha = 0.005
    beta = 0.05
    
    steps = 1000
    ws = np.zeros((FEATURE_SIZE, steps))
    RMSVE = np.zeros(steps)
    RMSPBE = np.zeros(steps)
    
    state = np.random.choice(STATES)    
    for step in tqdm(range(steps)):
        state = TDC(state, w, v, alpha, beta)
        ws[:, step] = w
        RMSVE[step] = compute_RMSVE(w)
        RMSPBE[step] = compute_RMSPBE(w)
        
        
    for i in range(FEATURE_SIZE):
        plt.plot(ws[i, :], label='w' + str(i + 1))
    plt.plot(RMSVE, label='RMSVE')
    plt.plot(RMSPBE, label='RMSPBE')
    plt.xlabel('Steps')
    plt.title('TDC')
    plt.legend()
        
# figure_11_5_left()

def figure_11_5_right():
    # initialization
    w = np.ones(FEATURE_SIZE)
    w[6] = 10
    v = np.zeros(FEATURE_SIZE)
    
    alpha = 0.005
    beta = 0.05
    
    sweeps = 1000
    ws = np.zeros((FEATURE_SIZE, sweeps))
    RMSVE = np.zeros(sweeps)
    RMSPBE = np.zeros(sweeps)
    
    
    for sweep in tqdm(range(sweeps)):
        expected_TDC(w, v, alpha, beta)
        ws[:, sweep] = w
        RMSVE[sweep] = compute_RMSVE(w)
        RMSPBE[sweep] = compute_RMSPBE(w)
        
        
    for i in range(FEATURE_SIZE):
        plt.plot(ws[i, :], label='w' + str(i + 1))
    plt.plot(RMSVE, label='RMSVE')
    plt.plot(RMSPBE, label='RMSPBE')
    plt.xlabel('Sweeps')
    plt.title('Expected TDC')
    plt.legend()


# figure_11_5_right()


def figure_11_5():
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    figure_11_5_left()
    plt.subplot(1, 2, 2)
    figure_11_5_right()

figure_11_5()



def figure_11_6():
    # initialization
    w = np.ones(FEATURE_SIZE)
    w[6] = 10
       
    alpha = 0.03
   
    sweeps = 1000
    ws = np.zeros((FEATURE_SIZE, sweeps))
    RMSVE = np.zeros(sweeps)
    m = 0.0
    
    
    for sweep in tqdm(range(sweeps)):
        m = expected_emphatic_TD(w, m, alpha)   # 这里一定要用函数的return去重新赋值m, 因为m是随着t不断更新的
        ws[:, sweep] = w
        RMSVE[sweep] = compute_RMSVE(w)
        
    plt.figure()
    for i in range(FEATURE_SIZE):
        plt.plot(ws[i, :], label='w' + str(i + 1))
    plt.plot(RMSVE, label='RMSVE')
    plt.xlabel('Sweeps')
    plt.title('Expected emphatic TD')
    plt.legend()



figure_11_6()


















































































