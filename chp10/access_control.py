# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:03:15 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter10/mountain_car.py
"""


import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import floor



#====== tile coding is copied http://incompleteideas.net/tiles/tiles3.py-remove =========
basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self):
        return len(self.dictionary)
    
    def fullp(self):
        return len(self.dictionary) >= self.size
    
    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates


def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles
#============tile coding ends ===========================================================


PRIORITIES = [0, 1, 2, 3]
REWARDS = [1, 2, 4, 8]

REJECT = 0
ACCEPT = 1
ACTIONS = [REJECT, ACCEPT]

# total number of servers
NUM_SERVERS = 10
# at each time step, a busy server will be free w.p. 0.06
PROBABILITY_FREE = 0.06

# step size for learning state-action value
ALPHA = 0.01
# step size for learning average reward
BETA = 0.01
# probability for exploration
EPSILON = 0.1


class ValueFunction:
    def __init__(self, num_tilings, alpha=ALPHA, beta=BETA):
        self.num_tilings = num_tilings
        self.max_size = 2048
        self.hash_table = IHT(self.max_size)
        self.w = np.zeros(self.max_size)

        # state features needs scaling to satisfy the tile software
        self.server_scale = self.num_tilings / float(NUM_SERVERS)
        self.priority_scale = self.num_tilings / float(len(PRIORITIES) - 1)

        self.average_reward = 0.0

        # divide step size equally to each tiling
        self.alpha = alpha / self.num_tilings

        self.beta = beta

        
    def get_active_tiles(self, free_servers, priority, action):
        active_tiles = tiles(self.hash_table, self.num_tilings,
                            [self.server_scale * free_servers, self.priority_scale * priority],
                            [action])
        return active_tiles


        
    # estimate the value of given state and action without subtracting average
    def value(self, free_servers, priority, action):
        active_tiles = self.get_active_tiles(free_servers, priority, action)
        return np.sum(self.w[active_tiles])
    
    
    # estimate the value of given state without subtracting average
    def state_value(self, free_servers, priority):
        values = [self.value(free_servers, priority, action) for action in ACTIONS]
        # if no free server, can't accept
        if free_servers == 0:
            return values[REJECT]
        return np.max(values)
    
    
    def learn(self, free_servers, priority, action, target):
        active_tiles = self.get_active_tiles(free_servers, priority, action)
        estimation = np.sum(self.w[active_tiles])
        delta = target - estimation
        # update average reward
        self.average_reward += self.beta * delta
        delta *= self.alpha
        for active_tile in active_tiles:
            self.w[active_tile] += delta
    
    # def learn(self, free_servers, priority, action, new_free_servers, new_priority, new_action, reward):
    #     active_tiles = self.get_active_tiles(free_servers, priority, action)
    #     estimation = np.sum(self.w[active_tiles])
    #     delta = reward - self.average_reward + self.value(new_free_servers, new_priority, new_action) - estimation
    #     # update average reward
    #     self.average_reward += self.beta * delta
    #     delta *= self.alpha
    #     for active_tile in active_tiles:
    #         self.w[active_tile] += delta
            


def epsilon_greedy(free_servers, priority, value_func):
    # if no free server, can't accept
    if free_servers == 0:
        return REJECT
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(value_func.value(free_servers, priority, action))
        
    return ACTIONS[np.random.choice(np.where(values == np.max(values))[0])]
    


def step(free_servers, priority, action):
    if free_servers > 0 and action == ACCEPT:
        free_servers -= 1
    reward = REWARDS[priority] * action
    #some busy servers may become free
    busy_servers = NUM_SERVERS - free_servers
    free_servers += np.random.binomial(busy_servers, PROBABILITY_FREE)
    return free_servers, np.random.choice(PRIORITIES), reward



def differential_semi_gradient_sarsa(value_function, max_steps, n_steps=1):
    '''
    @valueFunction: state value function to learn
    @maxSteps: step limit in the continuing task
    @n_steps: n-step sarsa
    '''
    current_free_servers = NUM_SERVERS
    current_priority = np.random.choice(PRIORITIES)
    current_action = epsilon_greedy(current_free_servers, current_priority, value_function)
    
    # track the hit for each number of free servers, 不记录也可以
    freq = np.zeros(NUM_SERVERS + 1)
    
    # for _ in tqdm(range(max_steps)):
    #     freq[current_free_servers] += 1
    #     new_free_servers, new_priority, reward = step(current_free_servers, current_priority, current_action)
    #     new_action = epsilon_greedy(new_free_servers, new_priority, value_function)
    #     target = reward - value_function.average_reward + value_function.value(new_free_servers, new_priority, new_action)
    #     value_function.learn(current_free_servers, current_priority, current_action,
    #                          target)
    #     current_free_servers = new_free_servers
    #     current_priority = new_priority
    #     current_action = new_action
    
    # track the trajectory
    free_servers = [current_free_servers]
    priorities = [current_priority]
    actions = [current_action]
    rewards = [0.0]
    
    for t in tqdm(range(max_steps)):
        freq[current_free_servers] += 1
        
        new_free_servers, new_priority, reward = step(current_free_servers, current_priority, current_action)
        new_action = epsilon_greedy(new_free_servers, new_priority, value_function)
        
        # track new state and action
        free_servers.append(new_free_servers)
        priorities.append(new_priority)
        actions.append(new_action)
        rewards.append(reward)
        
        
        tau = t + 1 - n_steps
        if tau >= 0:
            target = 0.0
            for i in range(tau + 1, min(max_steps, tau + n_steps) + 1):
                target += rewards[i] - value_function.average_reward
                
            if tau + n_steps <= max_steps:
                target += value_function.value(free_servers[tau + n_steps], 
                                                priorities[tau + n_steps],
                                                actions[tau + n_steps])
                
                
            value_function.learn(free_servers[tau], priorities[tau], actions[tau], target)
                
                
        if tau == max_steps - 1:
            break
        
        current_free_servers = new_free_servers
        current_priority = new_priority
        current_action = new_action
        
    print(f'Frequency of number of free servers: {freq / max_steps}')








# Figure 10.5, Differential semi-gradient Sarsa on the access-control queuing task
def figure_10_5(seed):
    np.random.seed(seed)
    max_steps = int(1e6)
    num_tilings = 8
    value_function = ValueFunction(num_tilings)
    differential_semi_gradient_sarsa(value_function, max_steps, n_steps=1)
    values = np.zeros((len(PRIORITIES), NUM_SERVERS + 1))
    for priority in PRIORITIES:
        for free_servers in range(NUM_SERVERS + 1):
            values[priority, free_servers] = value_function.state_value(free_servers, priority)
    
    fig = plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for priority in PRIORITIES:
        plt.plot(range(NUM_SERVERS + 1), values[priority, :], label='priority %d' % (REWARDS[priority]))
    plt.xlabel('Number of free servers')
    plt.ylabel('Differential value of best action')
    plt.legend()
    
    ax = fig.add_subplot(2, 1, 2)
    policy = np.zeros((len(PRIORITIES), NUM_SERVERS + 1))
    for priority in PRIORITIES:
            for free_servers in range(NUM_SERVERS + 1):
                values = [value_function.value(free_servers, priority, action) for action in ACTIONS]
                if free_servers == 0:
                    policy[priority, free_servers] = REJECT
                else:
                    policy[priority, free_servers] = np.argmax(values)
                    
    fig = sns.heatmap(policy, cmap="YlGnBu", ax=ax, xticklabels=range(NUM_SERVERS + 1), yticklabels=REWARDS)
    fig.set_title('Policy (0 Reject, 1 Accept)')
    fig.set_xlabel('Number of free servers')
    fig.set_ylabel('Priority')



figure_10_5(3)















































