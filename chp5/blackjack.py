# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:10:52 2021

@author: leyuan

reference: 
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter05/blackjack.py
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/blackjack.py
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

    

# actions: hit or stick
HIT = 0
STICK = 1
ACTIONS = [HIT, STICK]

# state: [whether player has a usable Ace, sum of player's cards, one card of dealer]

# policy for player
POLICY_PLAYER = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    POLICY_PLAYER[i] = HIT
POLICY_PLAYER[20] = STICK
POLICY_PLAYER[21] = STICK


# function form of target policy of player
def target_policy_player(usable_ace, player_sum, dealer_card):
    return POLICY_PLAYER[player_sum]

# function form of behavior policy of player
def behavior_policy_player(usable_ace, player_sum, dealer_card):
    if np.random.binomial(n=1, p=0.5) == 1:
        return STICK
    return HIT


# policy for dealer
POLICY_DEALER = np.zeros(22, dtype=np.int)
for i in range(12, 17):
    POLICY_DEALER[i] = HIT
for i in range(17, 22):
    POLICY_DEALER[i] = STICK


# get a new card
# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
def get_card():
    return np.random.choice(DECK)



# get the value of a card (11 for ace)
def card_value(card_id):
    return 11 if card_id == 1 else card_id



# play a game
def play(policy_player, initial_state=None, initial_action=None, verbose=False):
       
    # player status
    player_sum = 0
    player_trajectory = []
    usable_ace_player = False
    
    # dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False
    
    if initial_state is None:
        # generate a random initial state
        
        while player_sum < 12:
            card = get_card()
            player_sum += card_value(card)
            
            # 如果玩家的牌和>21，则说明他可能有一张或者两张ace，最后一张一定是ace，且总和是22
            if player_sum > 21:
                assert player_sum == 22, 'in initalization phase, if player_sum > 21, player_sum must equal 22'
                # last card must be ace
                player_sum -= 10
            else:
                usable_ace_player |= (card == 1)
                
        # initialize cards for dealer
        dealer_card1 = get_card()
        dealer_card2 = get_card()
    
    else:
        # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()
        
    
    # inital state of the game
    state = [usable_ace_player, player_sum, dealer_card1]
    
    # initialize dealer's sum
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    # if dealer_sum > 21, he must hold two aces
    if dealer_sum > 21:
        assert dealer_sum == 22, 'if dealer_sum > 21, dealer_sum=22'
        # use one ace as 1 rather than 11
        dealer_sum -= 10
    assert dealer_sum <= 21
    assert player_sum <= 21

    if verbose:
        print('========================================================')
        print(f"usable_ace_player: {usable_ace_player}, \
                player_sum: {player_sum}, \
                dealer's first card: {dealer_card1}")
        print('Game Start!')
        print('========================================================')
        
        
        
    # player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # get action based on current sum
            action = policy_player(usable_ace_player, player_sum, dealer_card1)
            
        # track player's trajectory for importance sampling
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])
        
        
        if action == STICK:
            break
        
        # if hit, get a new card
        card = get_card()
        # Keep track of the ace count. the usable_ace_player flag is insufficient alone as it cannot
        # distinguish between having one ace or two.
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        player_sum += card_value(card)
        
        # if the player has a usable ace, use it as 1 to avoid busting and continue
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1
            
        # player bust
        if player_sum > 21:
            return state, -1, player_trajectory
        
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1)
        
        
    # dealer's turn
    while True:
        action = POLICY_DEALER[dealer_sum]
        if action == STICK:
            break
        
        card = get_card()
        ace_count = int(usable_ace_dealer)
        if card == 1:
            ace_count += 1
        dealer_sum += card_value(card)
        
        # if the player has a usable ace, use it as 1 to avoid busting and continue
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
            
        # dealer bust
        if dealer_sum > 21:
            return state, 1, player_trajectory
        
        assert dealer_sum <= 21
        usable_ace_dealer = (ace_count == 1)
        
        
    # compare the sum between player and dealer
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory
    
    
        
# On-Policy Monte Carlo Evaluation
def mc_evalation_on_policy(num_episode):
    '''
    分别考虑有无usable_ace的情况，所以状态table是一个10 * 10的矩阵， 当然直接构造一个10 * 10 * 2的数组也是没问题的
    横轴表示 player sum: [12, 21]
    纵轴表示 dealer showing: [1, 10]
    '''
    states_usable_ace = np.zeros((10, 10))
    # initalize counts to 1 to avoid 0 being divided
    states_usable_ace_count = np.ones((10, 10))
    
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))
    
    
    for i in tqdm(range(num_episode)):
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            player_sum -= 12    # for matching the index of the state table
            dealer_card -= 1    # for matching the index of the state table
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
                
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count
                
def figure_5_1():
    states_usable_ace_1, states_no_usable_ace_1 = mc_evalation_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = mc_evalation_on_policy(500000)

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]
    
    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']  
    player_axis, dealer_axis = np.meshgrid(range(12, 22), range(1, 11))
    fig = plt.figure()
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.plot_surface(dealer_axis, player_axis, states[i-1].T, cmap=plt.cm.bwr)
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(12, 22))
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_title(titles[i-1])
           



# Monte Carlo Control with Exploring Starts
def mc_control_es(num_episode):
    '''
    因为是control问题，所以针对的是state-action value function Q(s,a)
    '''
    # (playerSum, dealerCard, usableAce, action)
    state_action_values = np.zeros((10, 10, 2, 2))
    state_action_pair_count = np.ones((10, 10, 2, 2))
    
    # target policy is greedy
    def greedy_policy(usable_ace, player_sum, dealer_card):
        usable_ace = int(usable_ace)
        player_sum -= 12
        dealer_card -= 1
        # get argmax of the average returns(s, a)
        values = state_action_values[player_sum, dealer_card, usable_ace, :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
    
    for i in tqdm(range(num_episode)):
        # randomly initialize a state and action
        initial_state = [
                          bool(np.random.choice([0, 1])),
                          np.random.choice(range(12, 22)),
                          np.random.choice(range(1, 11))
                          ]
        initial_action = np.random.choice(ACTIONS)
        
        _, reward, trajectory = play(greedy_policy, initial_state, initial_action)
        
        first_visit_check = set()  # use first-visit MC
        for (usable_ace, player_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, player_sum, dealer_card, action)
            if state_action in first_visit_check:
                continue
            first_visit_check.add(state_action)
            # update values
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1
            state_action_values[player_sum, dealer_card, usable_ace, action] += (reward - state_action_values[player_sum, dealer_card, usable_ace, action]) / state_action_pair_count[player_sum, dealer_card, usable_ace, action] 
            # state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            # state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1
            
    return state_action_values

def figure_5_2():
    state_action_values = mc_control_es(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)
    
    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)
    
    qs = [action_usable_ace,
          state_value_usable_ace,
          action_no_usable_ace,
          state_value_no_usable_ace]
    
    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    player_axis, dealer_axis = np.meshgrid(range(12, 22), range(1, 11))

    fig = plt.figure()
    for i in range(4):
        if i % 2 != 0:
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.plot_surface(dealer_axis, player_axis, qs[i].T, cmap=plt.cm.bwr)
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(12, 22))
            ax.set_xlabel("Dealer showing")
            ax.set_ylabel("Player sum")
             
        else:
            ax = fig.add_subplot(2, 2, i+1)
            
            sns.heatmap(pd.DataFrame(np.flip(qs[i], axis=0), index=range(21, 11, -1), columns=range(1,11)), 
                alpha=0.5, annot=True, cbar=False)
                       
        ax.set_title(titles[i])         

 

# Monte Carlo Control without Exploring Starts

def mc_control_epsilon_greedy(num_episode):
    '''
    因为已经没有exploring start这个条件了，所以要优化的策略必须是 epsilon-soft
    '''
    # (playerSum, dealerCard, usableAce, action)
    state_action_values = np.zeros((10, 10, 2, 2))
    state_action_pair_count = np.ones((10, 10, 2, 2))
    
    # target policy is greedy
    def epsilon_greedy_policy(usable_ace, player_sum, dealer_card, eps=0.1):
        usable_ace = int(usable_ace)
        player_sum -= 12
        dealer_card -= 1
        # get argmax of the average returns(s, a)
        values = state_action_values[player_sum, dealer_card, usable_ace, :]
        a_star = np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
        if np.random.rand() < eps:
            return np.random.choice(ACTIONS)
        return a_star
    
    for i in tqdm(range(num_episode)):
        # randomly initialize a state and action
        initial_state = [
                          bool(np.random.choice([0, 1])),
                          np.random.choice(range(12, 22)),
                          np.random.choice(range(1, 11))
                          ]
        initial_action = np.random.choice(ACTIONS)
        
        _, reward, trajectory = play(epsilon_greedy_policy, initial_state, initial_action)
        
        first_visit_check = set()  # use first-visit MC
        for (usable_ace, player_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, player_sum, dealer_card, action)
            if state_action in first_visit_check:
                continue
            first_visit_check.add(state_action)
            # update values
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1
            state_action_values[player_sum, dealer_card, usable_ace, action] += (reward - state_action_values[player_sum, dealer_card, usable_ace, action]) / state_action_pair_count[player_sum, dealer_card, usable_ace, action] 
            # state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            # state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1
            
    return state_action_values
   
def mc_control_with_eps_greedy():
    state_action_values = mc_control_epsilon_greedy(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)
    
    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)
    
    qs = [action_usable_ace,
          state_value_usable_ace,
          action_no_usable_ace,
          state_value_no_usable_ace]
    
    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    player_axis, dealer_axis = np.meshgrid(range(12, 22), range(1, 11))

    fig = plt.figure()
    for i in range(4):
        if i % 2 != 0:
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.plot_surface(dealer_axis, player_axis, qs[i].T, cmap=plt.cm.bwr)
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(12, 22))
            ax.set_xlabel("Dealer showing")
            ax.set_ylabel("Player sum")
             
        else:
            ax = fig.add_subplot(2, 2, i+1)
            
            sns.heatmap(pd.DataFrame(np.flip(qs[i], axis=0), index=range(21, 11, -1), columns=range(1,11)), 
                alpha=0.5, annot=True, cbar=False)
                       
        ax.set_title(titles[i])         



# Off-Policy Monte Carlo evaluation
def mc_evalation_off_policy(num_episode):
    '''
    根据书中例5.4的描述，评估的状态是[usable_ace=True, player_sum=13, dealer_card=2]
    behavior policy是completely random
    target policy和之前一样——stick only on a sum of 20 or 21
    '''
    initial_state = [True, 13, 2]
    
    rhos = []
    returns = []
    
    for i in range(num_episode):
        _, reward, trajectory = play(behavior_policy_player, initial_state)
        
        # get importance ratio
        '''
        这里的ratio计算有些trick，因为behavior policy是完全随机，所以每个动作被选择的概率是0.5， 
        target policy是deterministic的，所以如果随机选出的动作是target policy对应的动作，那么概率就是1，否则就是0
        '''
        numerator = 1.0
        denominator = 1.0
        for (usable_ace, player_sum, dealer_card), action in trajectory:
            if action == target_policy_player(usable_ace, player_sum, dealer_card):
                denominator *= 0.5
            else:
                numerator = 0.0
                break
        rho = numerator / denominator
        rhos.append(rho)
        returns.append(reward)
    
    
    rhos = np.array(rhos)
    returns = np.array(returns)
    weighted_returns = rhos * returns
    
    # 为了计算随episode变化的结果，需要记录一个累加的array
    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)
        
    ordinary_sampling = weighted_returns / np.arange(1, num_episode + 1)
    
    with np.errstate(divide='ignore',invalid='ignore'):
        weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)

    return ordinary_sampling, weighted_sampling
    
    
def figure_5_3():    
    true_value = 0.27726
    num_episode = 10000
    runs = 100
    error_ordinary = np.zeros(num_episode)        
    error_weighted = np.zeros(num_episode) 
    
    for i in tqdm(range(runs)):
        ordinary_sampling, weighted_sampling = mc_evalation_off_policy(num_episode)
        error_ordinary += np.power(ordinary_sampling - true_value, 2)
        error_weighted += np.power(weighted_sampling - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs
    
    plt.plot(np.arange(1, num_episode+ 1), error_ordinary, color='green', label='Ordinary Importance Sampling')
    plt.plot(np.arange(1, num_episode + 1), error_weighted, color='red', label='Weighted Importance Sampling')
    plt.ylim(-0.1, 5)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel(f'Mean square error\n(average over {runs} runs)')
    plt.xscale('log')
    plt.legend()
    



## ============================= test =====================================================
    
# states_usable_ace_1, states_no_usable_ace_1 = mc_evalation_on_policy(10000)
# player_axis, dealer_axis = np.meshgrid(range(12, 22), range(1, 11))
# fig = plt.figure()
# axe = plt.axes(projection='3d')
# axe.plot_surface(dealer_axis, player_axis, states_usable_ace_1.T, cmap=plt.cm.bwr)
# axe.set_xticks(range(1, 11))
# axe.set_yticks(range(12, 22))
# axe.set_xlabel("Dealer showing")
# axe.set_ylabel("Player sum")
# axe.set_title('MC')


# states_usable_ace_1, states_no_usable_ace_1 = mc_evalation_on_policy(10000)
# states_usable_ace_2, states_no_usable_ace_2 = mc_evalation_on_policy(500000)

# states = [states_usable_ace_1,
#           states_usable_ace_2,
#           states_no_usable_ace_1,
#           states_no_usable_ace_2]

# titles = ['Usable Ace, 10000 Episodes',
#           'Usable Ace, 500000 Episodes',
#           'No Usable Ace, 10000 Episodes',
#           'No Usable Ace, 500000 Episodes']  

    
    
    
# state_action_values = mc_control_es(500000)

# state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
# state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

# # get the optimal policy
# action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
# action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

# qs = [action_usable_ace,
#       state_value_usable_ace,
#       action_no_usable_ace,
#       state_value_no_usable_ace]

# titles = ['Optimal policy with usable Ace',
#           'Optimal value with usable Ace',
#           'Optimal policy without usable Ace',
#           'Optimal value without usable Ace']




# player_axis, dealer_axis = np.meshgrid(range(12, 22), range(1, 11))

# fig = plt.figure()
# for i in range(4):
#     if i % 2 != 0:
#         ax = fig.add_subplot(2, 2, i+1, projection='3d')
#         ax.plot_surface(dealer_axis, player_axis, qs[i].T, cmap=plt.cm.bwr)
#         ax.set_xticks(range(1, 11))
#         ax.set_yticks(range(12, 22))
#         ax.set_xlabel("Dealer showing")
#         ax.set_ylabel("Player sum")
         
#     else:
#         ax = fig.add_subplot(2, 2, i+1)
        
#         sns.heatmap(pd.DataFrame(np.flip(qs[i], axis=0), index=range(21, 11, -1), columns=range(1,11)), 
#             alpha=0.5, annot=True, cbar=False)
                   
#     ax.set_title(titles[i])         

































































































































































