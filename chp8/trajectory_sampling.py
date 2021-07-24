# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:44:13 2021

@author: leyuan
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm


# 2 actions
ACTIONS = [0, 1]

# each transition has a probability to terminate with 0
TERMINATION_PROB = 0.1

# maximum expected updates
MAX_STEPS = 20000

# epsilon greedy for behavior policy
EPSILON = 0.1


# break tie randomly
def argmax(value):
    return np.random.choice(np.where(value == np.max(value))[0])


class Task(object):
    def __init__(self, n_states, b):
        '''
        @n_states: num of non-terminal state
        @b: # of branches
        each episode starts at state 0, and terminates at state n_states
        '''
        
        self.n_states = n_states
        self.b = b
        
        
        # transition matrix, each state-action pair leads to b possible states with equal prob
        # and with a different random selection of b states for each state–action pair.
        self.transition = np.random.randint(n_states, size=(n_states, len(ACTIONS), b))
        # self.transition = np.zeros((n_states, len(ACTIONS), b))
        # for i in range(n_states):
        #     for j in range(len(ACTIONS)):
        #         self.transition[i, j] = np.random.sample(n_states, b, replace=False)
        
        self.reward = np.random.randn(n_states, len(ACTIONS), b)
        
        
    def step(self, state, action):
        if np.random.rand() < TERMINATION_PROB:
            return self.n_states, 0
        idx = np.random.choice(self.b)
        return self.transition[state, action, idx], self.reward[state, action, idx]
    


# Evaluate the value of the start state for the greedy policy
def evaluate_pi(q_value, task):
    # use MC method to estimate the state value
    runs = 1000
    returns = []
    for r in range(runs):
        reward = 0
        state = 0
        while state < task.n_states:
            action = argmax(q_value[state])
            state, r = task.step(state, action)
            reward += r
        returns.append(reward)
    
    return np.mean(returns)



def uniform(task, eval_interval):
    '''
    perform expected update from a uniform state-action distribution of the MDP @task
    evaluate the learned q value every @eval_interval steps
    '''
    
    performance = []
    q_value = np.zeros((task.n_states, len(ACTIONS)))

    for step in tqdm(range(MAX_STEPS)):
        '''
        因为是state-action pair是均匀分布，所有在MAX_STEPS当中平均个pair更新MAX_STEPS/(n_states * 2)次
        可以采用随机抽样的方式，每次从state里抽一个，从action里抽一个，但是这样程序会比较慢
        参考的代码里给出了一个比较巧妙的近似，就是从头到尾轮着更新
        '''
        state = step // len(ACTIONS) % task.n_states
        action = step % len(ACTIONS)
        
        next_states = task.transition[state, action]
        q_value[state, action] = (1 - TERMINATION_PROB) * np.mean(task.reward[state, action] + np.max(q_value[next_states, :], axis=1))
        
        if step % eval_interval == 0:
            v_pi = evaluate_pi(q_value, task)
            performance.append([step, v_pi])
            
    return zip(*performance)




def on_policy(task, eval_interval):
    performance = []
    q_value = np.zeros((task.n_states, len(ACTIONS)))
    state = 0  # every episode starts at state 0
    for step in tqdm(range(MAX_STEPS)):
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = argmax(q_value[state])
            
        next_state, _ = task.step(state, action)   # 因为用的是期望更新，所以采样出来的r不需要
        
        next_states = task.transition[state, action]
        q_value[state, action] = (1 - TERMINATION_PROB) * np.mean(task.reward[state, action] + np.max(q_value[next_states, :], axis=1))
        '''
        这个更新表达适合uniform是一样的，因为都是期望跟新，但是更要更新的内容q(s,a)是不同的，
        这里是来自于on policy的分布，uniform是来自于均匀分布
        下面判断next_state是否等于terminal state也是两者的差异，因为是轨迹采样（step函数），
        所以有可能会达到terminal
        '''
        
        if next_state == task.n_states:
            next_state = 0
        state = next_state
        
        
        if step % eval_interval == 0:
            v_pi = evaluate_pi(q_value, task)
            performance.append([step, v_pi])
            
    return zip(*performance)
        
        
        
        
            
def figure_8_8():
    num_states = [1000, 10000]
    branch = [1, 3, 10]
    methods =[on_policy, uniform]
    
    # average over 30 tasks
    n_tasks = 30
    
    # num of evaluation points
    x_ticks = 100

    plt.figure(figsize=(10, 20))
    
    for i, n in enumerate(num_states):
        plt.subplot(2, 1, i+1)
        for b in branch:
            tasks = [Task(n, b) for _ in range(n_tasks)]
            for method in methods:
                steps = None
                value = []
                for task in tasks:
                    steps, v = method(task, MAX_STEPS / x_ticks)
                    value.append(v)
                value = np.mean(np.asarray(value), axis=0)
                plt.plot(steps, value, label=f'branch = {b}, {method.__name__}')
        plt.title(f'{n} states')
        plt.xlabel('computation time, in expected updates')
        plt.ylabel('value of start state')
        plt.legend()
        
        
    
    
    
    

figure_8_8()        
        
        
        
        











































































