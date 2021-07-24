# -*- coding: utf-8 -*-
"""
Created on Fri May 21 19:50:33 2021

@author: leyuan

reference: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/random_walk.py
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm



STATES = ['L', 'A', 'B', 'C', 'D', 'E', 'R']
#        [ 0,   1,   2,   3,   4,   5,   6 ]

VALUES = np.zeros(len(STATES))

# initialization
VALUES[1:6] = 0.5
# VALUES[6] = 1             # This is critical !!! 我还没搞明白是为什么，如果最右边的终止状态价值初始化为1，那么batch update是OK的，但前面两问的结果又不行了
                            # 如果初始化为0，前面两问结果是OK的，但batch update就不行了...
'''
参考的程序中用了一个trick，就是并不是按照题目中的要求，当从E到达R时，reward是1，
而是设置全部的reward都是0，然后把R的value设置成1
按道理来说，这两个的效果是等价的，因为没有折扣，所以TD target = r + v(s')，前面是1还是后面是1结果都是一样的
可是不用这个trick的话就会出现我上面说的问题...，头大啊
我把用trick的TD做出来的结果放在最后
'''
                            
# set up the true values
TRUE_VALUES = np.zeros(7)
TRUE_VALUES[1:6] = np.arange(1, 6) / 6.0
# TRUE_VALUES[6] = 1

# actions
LEFT = 0
RIGHT = 1


def temporal_difference(values, alpha=0.1, use_batch=False):
    '''
    @values: current states value, will be updated if @use_batch is False
    @alpha: step size
    @use_batch: whether to update @values
    '''
    
    # start from state C
    state = 3
    trajectory = [state]
    rewards = [0]
    while True:
        old_state = state
        if np.random.binomial(1, 0.5) == LEFT:
            state -= 1
        else:
            state += 1
        
        reward = 1.0 if state == 6 else 0.0
        trajectory.append(state)
        rewards.append(reward)
        # TD update
        if not use_batch:
            values[old_state] = values[old_state] + alpha * (reward + values[state] - values[old_state])
        if state == 6 or state == 0:
            break
        
    return trajectory, rewards


def monte_carlo(values, alpha=0.1, use_batch=False):
    state = 3
    trajectory = [state]
    
    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        if np.random.binomial(1, 0.5) == LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break
        
    if not use_batch:
        for state_ in trajectory[::-1]:
            # MC update
            values[state_] += alpha * (returns - values[state_])
    
    return trajectory, [returns] * (len(trajectory) - 1)





# Example 6.2 left
def compute_state_values():
    episodes =[0, 1, 10, 100]
    current_values = np.copy(VALUES)
    
    plt.figure()
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(current_values[1:6], label=str(i) + ' episodes')
        # if i == 0: 
        #     print(temporal_difference(current_values))
        # else: temporal_difference(current_values)
        temporal_difference(current_values)
    plt.plot(TRUE_VALUES[1:6], label='true values')
    plt.xticks(range(5), ['A', 'B', 'C', 'D', 'E'])
    plt.xlabel('states')
    plt.ylabel('estimated values')
    plt.legend()
    
# compute_state_values()



# Example 6.2 right
def rmse():
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    
    plt.figure()
    
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD'
            linestyle = 'solid'
        else:
            method = 'MC'
            linestyle = 'dashdot'
            
        total_errors = np.zeros(episodes)
        
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            for i in range(episodes):
                error = np.sqrt(np.mean((TRUE_VALUES[1:6] - current_values[1:6])**2))
                errors.append(error)
                
                if method == 'TD':
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
                    
            total_errors += np.array(errors)
        
        total_errors /= runs
        
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
        
    plt.xlabel('episodes')
    plt.ylabel('RMSE')
    plt.legend()
        
    
# rmse()    



# figure 6.2 batch_learning
def batch_update(method, episodes, alpha=0.001):
    '''
    @method: 'TD' or 'MC'
    '''
    
    # run xx times experiments
    runs = 20
    total_errors = np.zeros(episodes)
    
    for r in tqdm(range(runs)):
        current_values = np.copy(VALUES)
        current_values[6] = 1             # need to initialize the value of the right terminal state to be 1, 原因不详...
        errors = []
        
        # record the states and rewards sequences
        trajectories = []
        rewards = []
        
        for ep in range(episodes):
            if method == 'TD':
                trajectory_, rewards_ = temporal_difference(current_values, use_batch=True)
            else:
                trajectory_, rewards_ = monte_carlo(current_values, use_batch=True)
                
            trajectories.append(trajectory_)
            rewards.append(rewards_)
            
            while True:
                # using trajectories seen so far until value function converges
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(len(trajectory_) - 1):
                        if method == 'TD':
                            updates[trajectory_[i]] += rewards_[i] + current_values[trajectory_[i + 1]] - current_values[trajectory_[i]]
                        else:
                            updates[trajectory_[i]] += rewards_[i] - current_values[trajectory_[i]]
                
                updates *= alpha
                
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                # batch updating
                current_values += updates
                
            errors.append(np.sqrt(np.mean((TRUE_VALUES[1:6] - current_values[1:6])**2)))    
            
        total_errors += np.array(errors)
        
    total_errors /= runs
    
    return total_errors


def figure_6_2():
    episodes = 100 + 1
    td_errors = batch_update('TD', episodes)
    mc_errors = batch_update('MC', episodes)
    
    plt.figure()
    plt.plot(td_errors, label='TD')
    plt.plot(mc_errors, label='MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()


figure_6_2()




# ============================ TD for random walk with tricks ===========================================================

STATES = ['L', 'A', 'B', 'C', 'D', 'E', 'R']
#        [ 0,   1,   2,   3,   4,   5,   6 ]

VALUES = np.zeros(len(STATES))

# initialization
VALUES[1:6] = 0.5
VALUES[6] = 1             # This is critical !!! 
                            
# set up the true values
TRUE_VALUES = np.zeros(7)
TRUE_VALUES[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUES[6] = 1

# actions
LEFT = 0
RIGHT = 1


def temporal_difference(values, alpha=0.1, use_batch=False):
    '''
    @values: current states value, will be updated if @use_batch is False
    @alpha: step size
    @use_batch: whether to update @values
    '''
    
    # start from state C
    state = 3
    trajectory = [state]
    rewards = [0]
    while True:
        old_state = state
        if np.random.binomial(1, 0.5) == LEFT:
            state -= 1
        else:
            state += 1
        # assume all rewards are 0
        reward = 0
        trajectory.append(state)
        # TD update
        if not use_batch:
            values[old_state] += alpha * (reward + values[state] - values[old_state])
        if state == 6 or state == 0:
            break
        rewards.append(reward)
        
    return trajectory, rewards


def monte_carlo(values, alpha=0.1, use_batch=False):
    state = 3
    trajectory = [state]
    
    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        if np.random.binomial(1, 0.5) == LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break
        
    if not use_batch:
        for state_ in trajectory[::-1]:
            # MC update
            values[state_] += alpha * (returns - values[state_])
    
    return trajectory, [returns] * (len(trajectory) - 1)



# Example 6.2 left
def compute_state_values():
    episodes =[0, 1, 10, 100]
    current_values = np.copy(VALUES)
    
    plt.figure()
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(current_values[1:6], label=str(i) + ' episodes')
        # if i == 0: 
        #     print(temporal_difference(current_values))
        # else: temporal_difference(current_values)
        temporal_difference(current_values)
    plt.plot(TRUE_VALUES[1:6], label='true values')
    plt.xticks(range(5), ['A', 'B', 'C', 'D', 'E'])
    plt.xlabel('states')
    plt.ylabel('estimated values')
    plt.legend()
    
# compute_state_values()



# Example 6.2 right
def rmse():
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    
    plt.figure()
    
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD'
            linestyle = 'solid'
        else:
            method = 'MC'
            linestyle = 'dashdot'
            
        total_errors = np.zeros(episodes)
        
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            for i in range(episodes):
                error = np.sqrt(np.mean((TRUE_VALUES[1:6] - current_values[1:6])**2))
                errors.append(error)
                
                if method == 'TD':
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
                    
            total_errors += np.array(errors)
        
        total_errors /= runs
        
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
        
    plt.xlabel('episodes')
    plt.ylabel('RMSE')
    plt.legend()
        
    
# rmse()    



# figure 6.2 batch_learning
def batch_update(method, episodes, alpha=0.001):
    '''
    @method: 'TD' or 'MC'
    '''
    
    # run xx times experiments
    runs = 100
    total_errors = np.zeros(episodes)
    
    for r in tqdm(range(runs)):
        current_values = np.copy(VALUES)
        errors = []
        
        # record the states and rewards sequences
        trajectories = []
        rewards = []
        
        for ep in range(episodes):
            if method == 'TD':
                trajectory_, rewards_ = temporal_difference(current_values, use_batch=True)
            else:
                trajectory_, rewards_ = monte_carlo(current_values, use_batch=True)
                
            trajectories.append(trajectory_)
            rewards.append(rewards_)
            
            while True:
                # using trajectories seen so far until value function converges
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(len(trajectory_) - 1):
                        if method == 'TD':
                            updates[trajectory_[i]] += rewards_[i] + current_values[trajectory_[i + 1]] - current_values[trajectory_[i]]
                        else:
                            updates[trajectory_[i]] += rewards_[i] - current_values[trajectory_[i]]
                
                updates *= alpha
                
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                # batch updating
                current_values += updates
                
            errors.append(np.sqrt(np.mean((TRUE_VALUES[1:6] - current_values[1:6])**2)))    
            
        total_errors += np.array(errors)
        
    total_errors /= runs
    
    return total_errors


def figure_6_2():
    episodes = 100 + 1
    td_errors = batch_update('TD', episodes)
    mc_errors = batch_update('MC', episodes)
    
    plt.figure()
    plt.plot(td_errors, label='TD')
    plt.plot(mc_errors, label='MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()


figure_6_2()






























































































