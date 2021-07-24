# -*- coding: utf-8 -*-
"""
Created on Sun May 23 09:34:45 2021

@author: leyuan

reference: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/cliff_walking.py
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm


WORLD_HEIGHT = 4
WORLD_WIDTH = 12


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

# gamma for Sarsa and Q-learning
GAMMA = 1


# start and goal position of the world, origin on top left corner，[height, width]
START = [3, 0]
GOAL = [3, 11]



def step(state, action):
    i, j = state
    if action == UP:
        next_state = [max(i - 1, 0),  j]
    elif action == DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1),  j]
    elif action == LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False, "action must be 'UP', 'DOWN', 'LEFT', 'RIGHT'." 
        
    reward = -1
    if (action == DOWN and i == 2 and 1 <= j <= 10) or (action == RIGHT and state == START):
        reward = -100
        next_state = START
    
    return next_state, reward



def epsilon_greedy(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        q_val = q_value[state[0], state[1], :]
        action = np.random.choice(np.where(q_val == np.max(q_val))[0])
        
    return action


def sarsa(q_value, expected=False, step_size=ALPHA):
    '''
    @q_value: values for state action pair, will be updated
    @expected: if True, will use expected Sarsa algorithm
    @step_size: step size for updating
    @return: total rewards within this episode
    '''
    
    state = START
    action = epsilon_greedy(state, q_value)
    rewards = 0.0
    
    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = epsilon_greedy(next_state, q_value)
        rewards += reward
        
        if not expected:
            target = reward + GAMMA * q_value[next_state[0], next_state[1], next_action]
        else:
            # calculate the expected value of new state
            target = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            
            for action_ in ACTIONS:
                if action_ in best_actions:
                    target += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_next[action_]
                else:
                    target += EPSILON / len(ACTIONS) * q_next[action_]
            
            target = reward + GAMMA * target
            
        # Sarsa update
        q_value[state[0], state[1], action] += step_size * (target - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        
    return rewards



                
def q_learning(q_value, step_size=ALPHA):
    state = START
    rewards = 0.0
    
    while state != GOAL:
        action = epsilon_greedy(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        
        # Q_learning update
        q_value[state[0], state[1], action] += step_size * \
            (reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) - q_value[state[0], state[1], action])

        state = next_state
        
    return rewards


# print optimal policy
def print_policy(q_value):
    policy = []
    for i in range(WORLD_HEIGHT):
        policy.append([])
        for j in range(WORLD_WIDTH):
            # if [i, j] == START:
            #     policy[-1].append('S')
            #     continue
            if [i, j] == GOAL:
                policy[-1].append('G')
                continue
            best_action = np.argmax(q_value[i, j, :])
            if best_action == UP:
                policy[-1].append('U')
            elif best_action == DOWN:
                policy[-1].append('D')
            elif best_action == LEFT:
                policy[-1].append('L')
            elif best_action == RIGHT:
                policy[-1].append('R')
                
    for row in policy:
        print(row)



'''
Use multiple runs instead of a single run and a sliding window
a single run fails to present a smooth curve
However the optimal policy converges well with a single run
Sarsa converges to the safe path, while Q-Learning converges to the optimal path
'''

def example_6_6(runs=50, episodes=500):    
    
    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    
    for r in tqdm(range(runs)):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
        q_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
        
        for i in range(episodes):
            rewards_sarsa[i] += sarsa(q_sarsa)
            rewards_q_learning[i] += q_learning(q_q_learning)
            
    
    rewards_sarsa /= runs
    rewards_q_learning /= runs
    
    
    # draw the curves
    plt.figure()
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()
    
    
    # display optimal policy
    print('Sarsa Optimal Policy:')
    print_policy(q_sarsa)
    print('Q-Learning Optimal Policy:')
    print_policy(q_q_learning)
    

example_6_6(runs=50, episodes=500)
    



# figure 6.3
'''I can't complete this experiment
with 100,000 episodes and 50,000 runs to get the fully averaged performance is hard
However even  only play for 1,000 episodes and 10 runs, the curves looks still good.
'''
def figure_6_3(runs=10, episodes=1000):
    step_sizes = np.arange(0.1, 1.1, 0.1)
    
    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_QLEARNING = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_QLEARNING = 5
    
    
    results = np.zeros((6, len(step_sizes)))
    
    for run in range(runs):
        for ind, step_size in tqdm(enumerate(step_sizes)):
            q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
            q_expected_sarsa = np.copy(q_sarsa)
            q_q_learning = np.copy(q_sarsa)
            
            for ep in range(episodes):
                sarsa_reward = sarsa(q_sarsa, expected=False, step_size=step_size)
                expected_sarsa_reward = sarsa(q_expected_sarsa, expected=True, step_size=step_size)
                q_learning_reward = q_learning(q_q_learning, step_size=step_size)
                results[ASY_SARSA, ind] += sarsa_reward
                results[ASY_EXPECTED_SARSA, ind] += expected_sarsa_reward
                results[ASY_QLEARNING, ind] += q_learning_reward
                
                if ep < 100:
                    results[INT_SARSA, ind] += sarsa_reward
                    results[INT_EXPECTED_SARSA, ind] += expected_sarsa_reward
                    results[INT_QLEARNING, ind] += q_learning_reward
                
    results[:3, :] /= episodes * runs
    results[3:, :] /= 100 * runs
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']
                
    for ind, label in enumerate(labels):
        plt.plot(step_sizes, results[ind, :], label=label)
        plt.xlabel('alpha')
        plt.ylabel('reward per episode')
        plt.legend() 
    

figure_6_3(runs=10, episodes=1000)

















