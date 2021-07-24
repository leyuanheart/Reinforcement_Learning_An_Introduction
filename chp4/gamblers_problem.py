# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:29:21 2021

@author: leyuan

reference: 
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/gamblers_problem.py
    https://github.com/brynhayder/reinforcement_learning_an_introduction/blob/master/code/exercises/ex_4_9/gamblers_problem.py
"""


import matplotlib.pyplot as plt
import numpy as np



# goal
GOAL = 100

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)  # [0, 1, ... , 100]

# probability of head
HEAD_PROB = 0.4

# threshold
THETA = 1e-5

# ==========================================================================

state_value = np.zeros(GOAL + 1)
state_value[GOAL] = 1  # v(0)=0,  v(100)=1

sweep_history = []
iteration = 0

# Value Iteration
while True:
    old_state_value = state_value.copy()
    sweep_history.append(old_state_value)
    
    for state in STATES[1:GOAL]:   # [1, 2, ... , 99]
        # get all possible actions for current state
        actions = np.arange(1, min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            # reward = 0 
            action_returns.append(HEAD_PROB * (state_value[state+action]) + (1 - HEAD_PROB) * state_value[state - action])
            
        new_value = np.max(action_returns)
        state_value[state] = new_value
        
    delta = abs(state_value - old_state_value).max()
    
    iteration += 1
    
    if delta < THETA:
        sweep_history.append(state_value)
        print(f'converge at iteration {iteration}')
        break
    
    

# extract policy from optimal value
policy = np.zeros(GOAL + 1)
for state in STATES[1:GOAL]:
    actions = np.arange(1, min(state, GOAL - state) + 1)
    action_returns = []
    for action in actions: 
        action_returns.append(HEAD_PROB * (state_value[state+action]) + (1 - HEAD_PROB) * state_value[state - action])
    
        
    # round to resemble the figure in the book, see
    # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
    # Since numpy.argmax chooses the first option in case of ties, 
    # rounding the near-ties assures the one associated with the smallest action (or bet) is selected. 
    # The output of the app now resembles Figure 4.3. in the Sutton/Bartho's book.
    policy[state] = actions[np.argmax(np.round(action_returns, 5))]
    


# ================== plot ===================================================
plt.figure()
plt.subplot(211)
for sweep, state_value in enumerate(sweep_history):
    plt.plot(state_value, label=f'sweep {sweep}')
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.legend(loc='best', bbox_to_anchor=(1, 1))

plt.subplot(212)
# plt.scatter(STATES[1:GOAL], policy[1:GOAL])
# plt.plot(policy)
plt.bar(STATES, policy)
plt.xlim(0, GOAL)
# plt.ylim(0, GOAL/2+1)
plt.xticks(np.arange(0, GOAL, 5))
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
        
        



































































