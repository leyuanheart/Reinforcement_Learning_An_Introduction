# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 08:54:47 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter10/mountain_car.py
"""



import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
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

# all possible actions
REVERSE = -1
ZERO = 0
FORWARD = 1
# order is important
ACTIONS = [REVERSE, ZERO, FORWARD]


# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0



def step(position, velocity, action):
    '''
    take an @action at @position and @velocity
    @return: new position, new velocity, reward (always -1)
    '''
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = min(max(VELOCITY_MIN, new_velocity), VELOCITY_MAX)
    new_position = position + new_velocity
    new_position = min(max(POSITION_MIN, new_position), POSITION_MAX)
    reward = -1.0
    if new_position == POSITION_MIN:
        new_velocity = 0.0
    return new_position, new_velocity, reward



class ValueFunction(object):
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    
    def __init__(self, step_size, num_tilings=8, max_size=2048):
        '''
        @step_size: global learning rate
        @max_size: the maximum # of indices
        '''
        self.num_tilings = num_tilings
        self.max_size = max_size
        
        # allocate step size equally to each tiling
        self.step_size = step_size / num_tilings
        
        self.hash_table = IHT(max_size)
        
        # weights for each tile ???
        self.w = np.zeros(max_size)
        
        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_tilings / (VELOCITY_MAX - VELOCITY_MIN)
        
        
        
    def get_active_tiles(self, position, velocity, action):
        active_tiles = tiles(self.hash_table, self.num_tilings,
                            [self.position_scale * position, self.velocity_scale * velocity],
                            [action])
        
        return active_tiles
    
    
    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.w[active_tiles])
    
    
    def learn(self, position, velocity, action, target):
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.w[active_tiles])
        delta = self.step_size * (target - estimation)
        
        for active_tile in active_tiles:
            self.w[active_tile] += delta
            
        
    def cost_to_go(self, position, velocity):
        '''
        get # of steps to reach the goal under current state value function
        '''
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        
        return -np.max(costs)
        
     
        
     
def epsilon_greedy(position, velocity, value_func):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(value_func.value(position, velocity, action))
        
    return ACTIONS[np.random.choice(np.where(values == np.max(values))[0])]
        
    
    

def semi_gradient_n_step_sarsa(value_function, n=1):
    # initialize the position and velocity
    current_position = np.random.uniform(-0.6, -0.4) 
    current_velocity = 0
    
    current_action = epsilon_greedy(current_position, current_velocity, value_function)
    
    # track previous position, velocity, action and reward
    positions = [current_position]
    velocities = [current_velocity]
    actions = [current_action]
    rewards = [0.0]

    # track the time
    t = 0
    
    
    T = 500 # float('inf')
    while True:
        if t < T:
            # take current action and go to the new state
            new_position, new_velocity, reward = step(current_position, current_velocity, current_action)
            # choose new action
            new_action = epsilon_greedy(new_position, new_velocity, value_function)
            
            # track new state and action
            positions.append(new_position)
            velocities.append(new_velocity)
            actions.append(new_action)
            rewards.append(reward)

            if new_position == POSITION_MAX:
                T = t + 1
            
        tau = t + 1 - n
        if tau >= 0:
            returns = 0.0
            for i in range(tau + 1, min(T, tau + n) + 1):
                returns += rewards[i]
                
            if tau + n <= T:
                returns += value_function.value(positions[tau + n], 
                                                velocities[tau + n],
                                                actions[tau + n])
                
                
            # update the state value function
            if positions[tau] != POSITION_MAX:
                value_function.learn(positions[tau], velocities[tau], actions[tau], returns)
                
                
        if tau == T - 1:
            break
        
        t += 1
        current_position = new_position
        current_velocity = new_velocity
        current_action = new_action
        
    return t



       
def print_cost(value_function, episode, ax):
    grid_size = 40
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)   

    axis_x = []
    axis_y = []
    axis_z = []

    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            axis_z.append(value_function.cost_to_go(position, velocity))
            
    ax.scatter(axis_x, axis_y, axis_z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.set_title('Episode %d' % (episode + 1))
                
                
    
# Figure 10.1, cost to go in a single run
def figure_10_1():
    episodes = 9000
    plot_episodes = [0, 99, episodes - 1]
    fig = plt.figure(figsize=(40, 10))
    axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]
    
    num_tilings = 8
    alpha = 0.3
    value_function = ValueFunction(alpha, num_tilings)
    
    for ep in tqdm(range(episodes)):
        semi_gradient_n_step_sarsa(value_function)
        if ep in plot_episodes:
            print_cost(value_function, ep, axes[plot_episodes.index(ep)])
            
            
            
figure_10_1()



# Figure 10.2, semi-gradient Sarsa with different alphas
def figure_10_2():
    runs = 10
    episodes = 500
    num_tilings = 8
    alphas = [0.1, 0.2, 0.5]
    
    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        value_functions = [ValueFunction(alpha, num_tilings) for alpha in alphas]
        for index in range(len(value_functions)):
            for ep in tqdm(range(episodes)):
                step = semi_gradient_n_step_sarsa(value_functions[index])
                steps[index, ep] += step
                
    steps /= runs
    
    for i in range(len(alphas)):
        plt.plot(steps[i], label='alpha = ' + str(alphas[i]) + '/' +str(num_tilings))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()


figure_10_2()



# Figure 10.3, one-step semi-gradient Sarsa vs multi-step semi-gradient Sarsa
def figure_10_3():
    runs = 10
    episodes = 500
    num_tilings = 8
    alphas = [0.5, 0.3]
    n_steps = [1, 8]
    
    
    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        value_functions = [ValueFunction(alpha, num_tilings) for alpha in alphas]
        for index in range(len(value_functions)):
            for ep in tqdm(range(episodes)):
                step = semi_gradient_n_step_sarsa(value_functions[index], n_steps[index])
                steps[index, ep] += step
                
    steps /= runs
    
    for i in range(len(alphas)):
        plt.plot(steps[i], label=f'n = {n_steps[i]}' )
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()


figure_10_3()




# Figure 10.4, effect of alpha and n on multi-step semi-gradient Sarsa
def figure_10_4():
    runs = 5
    episodes = 50
    alphas = np.arange(0.05, 1.75, 0.25)
    n_steps = np.power(2, np.arange(0, 5))
    
    max_steps = 300
    steps = np.zeros((len(n_steps), len(alphas)))
    for run in range(runs):
        for n_step_index, n_step in enumerate(n_steps):
            for alpha_index, alpha in enumerate(alphas):
                if (n_step == 8 and alpha > 1) or (n_step == 16 and alpha > 0.75):
                    # In these cases it won't converge, so ignore them
                    steps[n_step_index, alpha_index] += max_steps * episodes
                    continue
                value_function = ValueFunction(alpha)
                for ep in tqdm(range(episodes)):
                    step = semi_gradient_n_step_sarsa(value_function, n_step)
                    steps[n_step_index, alpha_index] += step
                    
                    
    steps /= runs * episodes
    
    
    for i in range(0, len(n_steps)):
        plt.plot(alphas, steps[i, :], label='n = '+str(n_steps[i]))
    plt.xlabel('alpha * number of tilings(8)')
    plt.ylabel('Steps per episode')
    plt.ylim([200, max_steps])
    plt.legend()
    
    

figure_10_4()











