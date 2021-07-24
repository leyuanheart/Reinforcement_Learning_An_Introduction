# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:38:30 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter12/random_walk.py
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



# base class for lambda-based algorithms in this chapter
# In this example, we use the simplest linear feature function, state aggregation.
# And we use exact 19 groups, so the weights for each group is exact the value for that state
class ValueFunction:
    def __init__(self, lamda, alpha):
        '''
        Parameters
        ----------
        lamda : TYPE
            param for exp moving average
        alpha : TYPE
            step size for param update
        '''
        self.lamda = lamda
        self.alpha = alpha
        self.weights = np.zeros(N_STATES + 2)
        
        
    # the state value is just the weight
    def value(self, state):
        return self.weights[state]
    
    
    # feed the algorithm with new observation
    # derived class should override this function
    def learn(self, state, reward):
        return
    
    
    # initialize some variables at the beginning of each episode
    # must be called at the very beginning of each episode
    # derived class should override this function
    def new_episode(self):
        return
    
    

# off-line lambda-return algorithm
class OfflineLambdaReturn(ValueFunction):
    def __init__(self, lamda, alpha):
        super(OfflineLambdaReturn, self).__init__(lamda, alpha)
        # To accelerate learning, set a truncate value for power of lambda
        self.lamda_truncate = 1e-3
        
    def new_episode(self):
        # initialize the trajectory
        self.trajectory = [START_STATE]
        # only need to track the last reward in an episode, as all others are 0
        self.reward = 0.0
        
        
    def learn(self, state, reward):
        self.trajectory.append(state)
        if state in END_STATES:
            self.reward = reward
            self.T = len(self.trajectory) - 1
            self.offline_learn()
            
            
    def n_step_return_from_time(self, n, time):
        # gamma is always 1 and rewards are zero except for the last reward
        # the formula can be simplified
        end_time = min(time + n, self.T)
        returns = self.value(self.trajectory[end_time])
        if end_time == self.T:
            returns += self.reward
        
        return returns
    
    
    
    def lambda_return_from_time(self, time):
        returns = 0.0
        lambda_pow = 1
        for n in range(1, self.T - time):
            returns += lambda_pow * self.n_step_return_from_time(n, time)
            lambda_pow *= self.lamda
            if lambda_pow < self.lamda_truncate:
                # If the power of lambda has been too small, discard all the following sequences
                break
        
        returns *= 1 - self.lamda
        
        if lambda_pow >= self.lamda_truncate:
            returns += lambda_pow * self.reward
            
        return returns
    
    
    # perform offline learning ate the end of an episode
    def offline_learn(self):
        for t in range(self.T):
            state = self.trajectory[t]
            delta = self.lambda_return_from_time(t) - self.value(state)
            delta *= self.alpha
            self.weights[state] += delta
            
            
            
class TDLambda(ValueFunction):
    def __init__(self, lamda, alpha):
        ValueFunction.__init__(self, lamda, alpha)
        self.new_episode()
        
    def new_episode(self):
        # initialize the eligibility trace
        self.z = np.zeros(N_STATES + 2)
        # initialize the beginning state
        self.last_state = START_STATE
        
    
    
    def learn(self, state, reward):
        # update the eligibility trace, the trace here is actually replacing trace or accumulating trace
        self.z *= self.lamda       # gamma = 1
        self.z[self.last_state] += 1
        
        # update the weights
        delta = reward + self.value(state) - self.value(self.last_state)
        delta *= self.alpha
        self.weights += delta * self.z
        self.last_state = state
        
        

class TrueOnlineTDLambda(ValueFunction):
    def __init__(self, lamda, alpha):
        super(TrueOnlineTDLambda, self).__init__(lamda, alpha)
        
        
    def new_episode(self):
        # initialize the eligibility trace
        self.z = np.zeros(N_STATES + 2)
        # initialize the beginning state
        self.last_state = START_STATE
        # initialize the old state value
        self.old_state_value = 0.0
        
    
    def learn(self, state, reward):
        last_state_value = self.value(self.last_state)
        state_value = self.value(state)
        # update the eligibility trace, the trace here is actually replacing trace or accumulating trace
        dutch = 1 - self.alpha * self.lamda * self.z[self.last_state]
        self.z *= self.lamda
        self.z[self.last_state] += dutch
        # update the weights
        delta = reward + state_value - last_state_value
        self.weights += self.alpha * (delta + last_state_value - self.old_state_value) * self.z
        self.weights[self.last_state] -= self.alpha * (last_state_value - self.old_state_value)
        self.old_state_value = state_value
        self.last_state = state
        
        

def random_walk(value_function):
    value_function.new_episode()
    state = START_STATE
    while state not in END_STATES:
        next_state = state + np.random.choice([-1, 1])
        if next_state == 0:
            reward = -1
        elif next_state == N_STATES + 1:
            reward == 1
        else:
            reward = 0        
        value_function.learn(next_state, reward)
        state = next_state
        
        
        

def parameter_sweep(value_function_generator, runs, lambdas, alphas):
    '''
    Parameters
    general plot framework
    @valueFunctionGenerator: generate an instance of value function
    @runs: specify the number of independent runs
    @lambdas: a series of different lambda values
    @alphas: sequences of step size for each lambda
    '''
    episodes = 10  # play for 10 episodes for each run
    errors = [np.zeros(len(alphas_)) for alphas_ in alphas]
    for run in tqdm(range(runs)):
        for lambda_index, lamda in enumerate(lambdas):
            for alpha_index, alpha in enumerate(alphas[lambda_index]):
                value_function = value_function_generator(lamda, alpha)
                for ep in range(episodes):
                    random_walk(value_function)
                    state_values = [value_function.value(state) for state in STATES]
                    errors[lambda_index][alpha_index] += np.sqrt(np.mean(np.power(state_values - TRUE_VALUE[1:-1], 2)))
        
        
    # average over runs and episodes
    for error in errors:
        error /= episodes * runs
        
    plt.figure()
    for i in range(len(lambdas)):
        plt.plot(alphas[i], errors[i], label='lambda = ' + str(lambdas[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.legend()
    
    
    
# Figure 12.3: Off-line lambda-return algorithm
def figure_12_3():
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01)]
    parameter_sweep(OfflineLambdaReturn, 50, lambdas, alphas)

   

# Figure 12.6: TD(lambda) algorithm
def figure_12_6():
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.99, 0.09),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.33, 0.03),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01),
              np.arange(0, 0.044, 0.004)]
    parameter_sweep(TDLambda, 50, lambdas, alphas)

    

# Figure 12.7: True online TD(lambda) algorithm
def figure_12_8():
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.88, 0.08),
              np.arange(0, 0.44, 0.04),
              np.arange(0, 0.11, 0.01)]
    parameter_sweep(TrueOnlineTDLambda, 50, lambdas, alphas)

    

# figure_12_3()
# figure_12_6()
figure_12_8()




















