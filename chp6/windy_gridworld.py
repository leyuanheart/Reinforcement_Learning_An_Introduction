# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:47:59 2021

@author: leyuan

reference: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/windy_grid_world.py
"""


import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm



WORLD_HEIGHT = 7
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


# possible actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]


# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0


# start and goal position of the world, origin on top left corner，[height, width]
START = [3, 0]
GOAL = [3, 7]


def step(state, action):
    '''
    注意，这里的风力指的是出发位置的风力，比如是从一个风力为1的地方往左走了一步，
    那么结果会比正常的向上多一步，而不管新到达的列的风力是多少
    '''
    i, j = state
    if action == UP:
        return [max(i - 1 - WIND[j], 0),  j]
    elif action == DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0),  j]
    elif action == LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False, "action must be 'UP', 'DOWN', 'LEFT', 'RIGHT'." 



# play for an episode
def episode(q_val):
    # track the total time steps in this episode
    timesteps = 0
    
    # initialization
    state = START
    
    # choose an action based on the epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values = q_val[state[0], state[1], :]
        action = np.random.choice(np.where(values == np.max(values))[0])
        
    #keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values = q_val[next_state[0], next_state[1], :]
            next_action = np.random.choice(np.where(values == np.max(values))[0])
        
        
        # Sarsa update
        q_val[state[0], state[1], action] += \
                ALPHA * (REWARD + q_val[next_state[0], next_state[1], next_action] 
                         - q_val[state[0], state[1], action])
                
        state = next_state
        action = next_action
        timesteps += 1
        
    return timesteps




def figure_6_3():
    '''
    书中的展示方式很奇怪，图片的纵轴是episode，横轴是每个episode所用step的累积求和，因为越到后面，
    策略会逐渐收敛到最优，所以每一个episode所用的步数就会逐渐下降并稳定在一个值，所以整个曲线表现出来就是
    斜率逐渐上升，其实横过来看就是增长趋于平缓，但是就是挺奇怪的
    '''
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
    episode_limit = 170
    
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        ep += 1
        
    steps = np.cumsum(steps)
    
    plt.figure()
    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    
    # display the optimal policy
    optimal_policy = []
    for i in range(WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(WORLD_WIDTH):
            # if [i, j] == START:
            #     optimal_policy[-1].append('S')
            #     continue
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            best_action = np.argmax(q_value[i, j, :])
            if best_action == UP:
                optimal_policy[-1].append('U')
            elif best_action == DOWN:
                optimal_policy[-1].append('D')
            elif best_action == LEFT:
                optimal_policy[-1].append('L')
            elif best_action == RIGHT:
                optimal_policy[-1].append('R')
    
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
        
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))    
            


figure_6_3()








































































































