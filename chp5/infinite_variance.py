# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:11:04 2021

@author: leyuan

reference: 
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter05/blackjack.py

"""

import numpy as np
import matplotlib.pyplot as plt

LEFT = 0
RIGHT = 1
ACTIONS = [LEFT, RIGHT]

# behavior policy
def behavior_policy():
    return np.random.binomial(1, 0.5)

# target policy
def target_policy():
    return LEFT

# one turn
def play():
    # track the action for importance ratio
    trajectory = []
    while True:
        action = behavior_policy()
        trajectory.append(action)
        if action == RIGHT:
            return 0, trajectory
        if np.random.binomial(1, 0.9) == 0:
            return 1, trajectory



def figure_5_4():
    np.random.seed(1)
    runs = 10
    num_episode = 100000
    for run in range(runs):
        rewards = []
        for episode in range(num_episode):
            reward, trajectory = play()
            if trajectory[-1] == RIGHT:
                rho = 0
            else:
                rho = 1.0 / pow(0.5, len(trajectory))
            rewards.append(rho * reward)
        rewards = np.add.accumulate(rewards)
        estimations = np.array(rewards) / np.arange(1, num_episode + 1)
        plt.plot(estimations)
    plt.hlines(y=1, linestyle='dashed', colors='black', xmin=0, xmax=num_episode)
    plt.hlines(y=2, linestyle='dashed', colors='black', xmin=0, xmax=num_episode)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel(r'Monte-Carlo estimate of $v_{\pi}(s)$')
    plt.title('Ordinary Importance Sampling')
    plt.xscale('log')
    
    
    
    
figure_5_4()    
