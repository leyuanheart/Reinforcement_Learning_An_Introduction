# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:37:45 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter12/mountain_car.py
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import floor



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

# discount is always 1.0 in these experiments
DISCOUNT = 1.0

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0

# max steps per episodes
STEP_LIMIT = 5000


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




def accumulating_trace(trace, active_tiles, lamda):
    '''
    Parameters
    ----------
    trace : TYPE
        old trace
    active_tiles : TYPE
        current active tile indices
    lamda : TYPE
        param for exp weighted moving avg

    Returns
    -------
    new trace
    '''
    trace *= DISCOUNT * lamda
    trace[active_tiles] += 1
    return trace


def replacing_trace(trace, active_tiles, lamda):
    active = np.isin(np.arange(len(trace)), active_tiles)
    trace[active] = 1
    trace[~active] *= DISCOUNT * lamda
    return trace



def replacing_trace_with_clearing(trace, active_tiles, lamda, clearing_tiles):
    '''
    "clearing" means set all tiles corresponding to non-selected actions to 0
    '''
    active = np.isin(np.arange(len(trace)), active_tiles)
    trace[~active] *= DISCOUNT * lamda
    trace[clearing_tiles] = 0
    trace[active] = 1
    return trace



def dutch_trace(trace, active_tiles, lamda, alpha):
    extra = 1 - alpha * DISCOUNT * lamda * np.sum(trace[active_tiles])
    trace *= DISCOUNT * lamda
    trace[active_tiles] += extra
    return trace




class SarsaLambda:
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @maxSize: the maximum # of indices
    def __init__(self, alpha, lamda, trace_update=accumulating_trace, num_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.trace_update = trace_update
        self.lamda = lamda
        if trace_update == dutch_trace:
            self.q_old = 0.0
        
        
        # divide alpha equally to each tiling 
        self.alpha = alpha / num_tilings
        
        self.hash_table = IHT(max_size)
        
        self.w = np.zeros(max_size)
        
        self.trace = np.zeros(max_size)
        
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
        delta = target - estimation
        
        if self.trace_update == accumulating_trace or self.trace_update == replacing_trace:
            self.trace_update(self.trace, active_tiles, self.lamda)
        elif self.trace_update == dutch_trace:
            self.trace_update(self.trace, active_tiles, self.lamda, self.alpha)
        elif self.trace_update == replacing_trace_with_clearing:
            clearing_tiles = []
            for a in ACTIONS:
                if a != action:
                    clearing_tiles.extend(self.get_active_tiles(position, velocity, a))
            self.trace_update(self.trace, active_tiles, self.lamda, clearing_tiles)
        else:
            raise Exception('Unexpected Trace Type')
        
        # if self.trace_update == dutch_trace:
        #     self.w += self.alpha * (delta + estimation - self.q_old) * self.trace
        #     self.w[active_tiles] -= self.alpha * (estimation - self.q_old)
        #     self.q_old = target
        # else:
        self.w += self.alpha * delta * self.trace
        '''
        这里其实是有些没太明白的，因为如果使用荷兰迹的话，应该是指用的true online Sarsa(λ)
        那么关于w的更新就不再是w_{t+1} = w_t + alpha * delta * z_t了
        而是更复杂的一个形式（参见书中的算法），但是我按照书中的写了（就是我注释掉的部分），结果却不对，
        反而就是直接简单版本的结果
        '''
            
            
            
    def cost_to_go(self, position, velocity): # 在这个例子中没用上
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




def play(evaluator):
    '''
      play Mountain Car for one episode based on given method @evaluator
      @return: total steps in this episode
    '''
    current_position = np.random.uniform(-0.6, -0.4) 
    current_velocity = 0
    
    current_action = epsilon_greedy(current_position, current_velocity, evaluator)
    steps = 0
    
    while True:
        new_position, new_velocity, reward = step(current_position, current_velocity, current_action)
        # choose new action
        new_action = epsilon_greedy(new_position, new_velocity, evaluator)
        
        target = reward + DISCOUNT * evaluator.value(new_position, new_velocity, new_action)
        evaluator.learn(current_position, current_velocity, current_action, target)
        current_position = new_position
        current_velocity = new_velocity
        current_action = new_action
        steps += 1
        
        if new_position == POSITION_MAX:
            break
        if steps >= STEP_LIMIT:
            print('step limit exceeded')
            break
        
    return steps
        
        
    
def figure_12_10():
    runs = 30
    episodes = 50
    alphas = np.arange(1, 8) / 4.0
    lams = [0.99, 0.95, 0.5, 0]
    '''
    由于电脑配置有限，只选择了几个λ的值来跑
    '''

    steps = np.zeros((len(lams), len(alphas), runs, episodes))
    for lamInd, lam in enumerate(lams):
        for alphaInd, alpha in enumerate(alphas):
            for run in tqdm(range(runs)):
                evaluator = SarsaLambda(alpha, lam, replacing_trace)
                for ep in range(episodes):
                    step = play(evaluator)
                    steps[lamInd, alphaInd, run, ep] = step

    # average over episodes
    steps = np.mean(steps, axis=3)

    # average over runs
    steps = np.mean(steps, axis=2)
    
    plt.figure()
    for lamInd, lam in enumerate(lams):
        plt.plot(alphas, steps[lamInd, :], label='lambda = %s' % (str(lam)))
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged steps per episode')
    plt.ylim([160, 300])
    plt.legend()


figure_12_10()  # 好像是半个小时吧。。。  
    
        
def figure_12_11():
    traceTypes = [dutch_trace, replacing_trace, replacing_trace_with_clearing, accumulating_trace]
    alphas = np.arange(0.2, 2.2, 0.2)
    episodes = 20
    runs = 30
    lam = 0.9
    rewards = np.zeros((len(traceTypes), len(alphas), runs, episodes))

    for traceInd, trace in enumerate(traceTypes):
        for alphaInd, alpha in enumerate(alphas):
            for run in tqdm(range(runs)):
                evaluator = SarsaLambda(alpha, lam, trace)
                for ep in range(episodes):
                    if trace == accumulating_trace and alpha > 0.6:
                        steps = STEP_LIMIT
                    else:
                        steps = play(evaluator)
                    rewards[traceInd, alphaInd, run, ep] = -steps

    # average over episodes
    rewards = np.mean(rewards, axis=3)

    # average over runs
    rewards = np.mean(rewards, axis=2)

    for traceInd, trace in enumerate(traceTypes):
        plt.plot(alphas, rewards[traceInd, :], label=trace.__name__)
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged rewards pre episode')
    plt.ylim([-550, -150])
    plt.legend()


figure_12_11()   # 大约1个小时
































































































