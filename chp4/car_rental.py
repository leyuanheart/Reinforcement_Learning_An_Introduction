# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:57:02 2021

@author: leyuan

reference: 
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/car_rental.py
    https://github.com/brynhayder/reinforcement_learning_an_introduction/blob/master/code/exercises/ex_4_7/analysis.py
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson



max_cars = 20      # maximum num of cars in each location
max_move_cars = 5  # maximum num of cars to move in one night
request_mean1 = 3  # expectation for rental requests in first location
request_mean2 = 4  # expectation for rental requests in second location
return_mean1 = 3   # expectation for # of cars returned in first location
return_mean2 = 2   # expectation for # of cars returned in second location
discount = 0.9
credit = 10        # credit by renting a car
move_cost = 2      # cost of moving a car


# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
# 如果不考虑计算时间的话，直接用每个location的最大车数20也是可以的
poisson_upper_bound = 11


# all possible actions, 
# "+" means moving from first loc to second loc
# "-" means moving from second loc to first loc
actions_range = np.arange(-max_move_cars, max_move_cars+1)




# =====================如何计算泊松分布的概率=======================================================
class Possion():
    def __init__(self, lamda):
        self.lamda = lamda
        
    def pmf(self, n):
        return poisson.pmf(n, self.lamda)
    
request_dis1 = Possion(request_mean1)
request_dis2 = Possion(request_mean2)
return_dis1 = Possion(return_mean1)
return_dis2 = Possion(return_mean2)

'''
上面的方法虽然应用起来比较方便，但是计算起来非常耗时，
我参考的代码中是通过把每种可能的结果都用字典存起来，这样只需要第一遍计算，之后的更新直接索引就可以，怎么操作我放在下面
'''

# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
# lamda < 10是为了区分每种不同的可能pair [n, lamda], 下面的函数里用了 n*10+lamda作为key，当然你也可以自己定义key，只要能区分开就OK
poisson_cache = dict()


def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]
# ==================================================================================



def expected_return(state, action, value, constant_returned_cars=True):
    """
    @state: [# of cars in first location, # of cars in second location]
    @action: positive if moving cars from first location to second location,
            negative if moving cars from second location to first location
    @value: state value matrix
    @constant_returned_cars:  if set True, model is simplified such that
    the # of cars returned in daytime becomes constant
    rather than a random value from poisson distribution, which will reduce calculation time
    and leave the optimal policy/value state matrix almost the same
    
    
    例子中说了，还回来的车要在第二天才available，所以计算的顺序是先处理租车的请求，再处理还车的结果，
    特别注意在计算期望收益的时候，是在4个泊松分布的联合分布下求期望。
    
    """
    
    # initial total return
    expected_return = 0.0
    
    # cost for moving cars
    expected_return -= move_cost * abs(action)
    
    # moving cars 一定要在go through 外面定义实际的car数，go through里面是模拟所有可能的情况
    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, max_cars)
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, max_cars)
    
    
    # go through all possible rental requests
    for rental_num_first_loc in range(poisson_upper_bound):
        for rental_num_second_loc in range(poisson_upper_bound):
            # prob for current combination of rental requests
            # prob = request_dis1.pmf(rental_num_first_loc) * request_dis2.pmf(rental_num_second_loc)
            '''
            use the following code for quicker computation
            '''
            prob = poisson_probability(rental_num_first_loc, request_mean1) * poisson_probability(rental_num_second_loc, request_mean1)
            
            
            num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC
            num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC
            
            
            # valid rental requests should not be larger than the actual num of cars
            valid_rental_num_first = min(num_of_cars_first_loc, rental_num_first_loc)
            valid_rental_num_second = min(num_of_cars_second_loc, rental_num_second_loc)
            
            # get credits for renting
            reward = (valid_rental_num_first + valid_rental_num_second) * credit
            
            num_of_cars_first_loc -= valid_rental_num_first
            num_of_cars_second_loc -= valid_rental_num_second
            
            
            # get returned cars, those cars can be available for renting tomorrow
            if constant_returned_cars:
                # get returned cars, those cars can be used for renting tomorrow
                returned_num_first_loc = return_mean1
                returned_num_second_loc = return_mean2
                
                num_of_cars_first_loc = min(num_of_cars_first_loc + returned_num_first_loc, max_cars)
                num_of_cars_second_loc =  min(num_of_cars_second_loc + returned_num_second_loc, max_cars)
                
                expected_return += prob * (reward + discount * value[num_of_cars_first_loc, num_of_cars_second_loc])
            else:
                for returned_num_first_loc in range(poisson_upper_bound):
                    for returned_num_second_loc in range(poisson_upper_bound):
                        # prob for current combination of rental requests
                        # prob_returned = return_dis1.pmf(returned_num_first_loc) * return_dis2.pmf(returned_num_second_loc)
                        '''
                        use the following code for quicker computation
                        '''
                        prob_returned = poisson_probability(returned_num_first_loc, return_mean1) * poisson_probability(returned_num_second_loc, return_mean1)
                        
                        
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_num_first_loc, max_cars)
                        num_of_cars_second_loc_ =  min(num_of_cars_second_loc + returned_num_second_loc, max_cars)
                        
                        
                        expected_return += prob * prob_returned * (reward + discount * value[num_of_cars_first_loc_, num_of_cars_second_loc_])
                    
                    
    return expected_return


def draw_fig(value, policy, iteration):
    fig = plt.figure(figsize=(15, 15)) 
    ax = fig.add_subplot(121)    
    ax.matshow(policy.T, cmap=plt.cm.bwr, vmin=-max_move_cars, vmax=max_move_cars)
    ax.set_xticks(range(max_cars+1))
    ax.set_yticks(range(max_cars+1))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlabel("Cars at second location")
    ax.set_ylabel("Cars at first location")
    for x in range(max_cars+1):
        for y in range(max_cars+1):
            ax.text(x=x, y=y, s=int(policy.T[x, y]), va='center', ha='center', fontsize=8)
    ax.set_title(r'$\pi_{}$'.format(iteration), fontsize=20)

    y, x = np.meshgrid(range(max_cars+1), range(max_cars+1))
    ax = fig.add_subplot(122, projection='3d')   
    ax.scatter3D(y, x, value.T)
    ax.set_xlim3d(0, max_cars)
    ax.set_ylim3d(0, max_cars)
    ax.set_xlabel("Cars at second location")
    ax.set_ylabel("Cars at first location")
    ax.set_title('value for ' + r'$\pi_{}$'.format(iteration), fontsize=20)
    plt.savefig(f'{iteration}.png', bbox_inches='tight')
        
    
    
# =============================policy iteration================================================
'''
行代表第一个loc，列代表第二loc
'''
value = np.zeros((max_cars+1, max_cars+1))    
policy = np.zeros(value.shape, dtype=np.int)

iteration = 0
while True:    
    while True:
        # policy evaluation
        old_value = value.copy()
        
        for i in range(max_cars+1):
            for j in range(max_cars+1):
                new_state_value = expected_return([i, j], policy[i, j], value)
                value[i, j] = new_state_value
                
        max_value_change = abs(old_value - value).max()
        print(f'max value change: {max_value_change}')
        if max_value_change < 1e-4:
            break
        
        
    # policy improvement
    policy_stable = True
    for i in range(max_cars+1):
        for j in range(max_cars+1):
            old_action = policy[i, j]
            
            action_returns = []
            for action in actions_range:
                if -j <= action <= i:  # valid action
                    action_returns.append(expected_return([i, j], action, value))
                else:
                    action_returns.append(-np.inf)
            
            action_returns = np.array(action_returns)      
            new_action = actions_range[np.where(action_returns == action_returns.max())[0]]
            policy[i, j] = np.random.choice(new_action)
            if policy_stable and (old_action not in new_action):
                policy_stable = False
                
    iteration += 1
    print('iteration: {}, policy stable {}'.format(iteration, policy_stable))
    
    draw_fig(value, policy, iteration)
    
    if policy_stable:
        break
    
    
# fig = plt.figure() 
# ax = fig.add_subplot(121)    
# ax.matshow(policy.T, cmap=plt.cm.bwr, vmin=-max_move_cars, vmax=max_move_cars)
# ax.set_xticks(range(max_cars+1))
# ax.set_yticks(range(max_cars+1))
# ax.invert_yaxis()
# ax.xaxis.set_ticks_position('none')
# ax.yaxis.set_ticks_position('none')
# ax.set_xlabel("Cars at second location")
# ax.set_ylabel("Cars at first location")
# for x in range(max_cars+1):
#     for y in range(max_cars+1):
#         ax.text(x=x, y=y, s=int(policy.T[x, y]), va='center', ha='center', fontsize=8)
# ax.set_title(r'$\pi_{}$'.format(iteration), fontsize=20)



# y, x = np.meshgrid(range(max_cars+1), range(max_cars+1))
# ax = fig.add_subplot(122, projection='3d')   
# ax.scatter3D(y, x, value.T)
# ax.set_xlim3d(0, max_cars)
# ax.set_ylim3d(0, max_cars)
# ax.set_xlabel("Cars at second location")
# ax.set_ylabel("Cars at first location")
# ax.set_title('value for ' + r'$\pi_{}$'.format(iteration), fontsize=20)
# plt.savefig(f'{iteration}.png', bbox_inches='tight')





