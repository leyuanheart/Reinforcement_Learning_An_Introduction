# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:44:01 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter08/maze.py
"""


import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import heapq
from copy import deepcopy


# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, 
# however it can be easily adapted to blocking maze or shortcut maze

class Maze(object):
    def __init__(self):
        self.MAZE_WIDTH = 9
        self.MAZE_HEIGHT = 6
        
        # all possible actions
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        
        
        self.START_STATE = [2, 0]
        self.GOAL_STATES = [[0, 8]]  # for extend the resolution
        
        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = None
        self.new_obstacles = None

        # time to change obstacles
        self.obstacle_switch_time = None

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.MAZE_HEIGHT, self.MAZE_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

        
    def extend_state(self, state, factor):
        '''
        extend a state to a higher resolution maze

        Parameters
        ----------
        state : list
            state in lower resolution maze
        factor : int
            extension factor, one state will become factor^2 states after extension

        Returns
        -------
        new_state

        '''
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(factor):
            for j in range(factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
                
        return new_states
    
    
    def extend_maze(self, factor):
        new_maze = Maze()
        new_maze.MAZE_HEIGHT = self.MAZE_HEIGHT * factor
        new_maze.MAZE_WIDTH = self.MAZE_WIDTH * factor
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        new_maze.GOAL_STATES = self.extend_state(self.GOAL_STATES[0], factor)
        
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
            
        new_maze.q_size = (new_maze.MAZE_HEIGHT, new_maze.MAZE_WIDTH, len(new_maze.actions))
        
        new_maze.resolution = factor
        
        return new_maze
    
    
    def step(self, state, action):
        x, y = state
        if action == self.UP:
            x = max(x - 1, 0)
        elif action == self.DOWN:
            x = min(x + 1, self.MAZE_HEIGHT - 1)
        elif action == self.LEFT:
            y = max(y - 1, 0)
        elif action == self.RIGHT:
            y = min(y + 1, self.MAZE_WIDTH - 1)
            
        if [x, y] in self.obstacles:
            x, y = state
        
        
        reward = 1.0 if [x, y] in self.GOAL_STATES else 0.0
        
        return [x, y], reward
        
        
            
# a wrapper class for parameters of dyna algorithms
class DynaParams(object):
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0            
            
            
            
            
def epsilon_greedy(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        action = np.random.choice(maze.actions)
    else:
        q_val = q_value[state[0], state[1], :]
        action = np.random.choice(np.where(q_val == np.max(q_val))[0])
        
    return action            
            

# Trivial model for planning in Dyna-Q            
class TrivialModel(object):
    def __init__(self):
        self.model = dict()
        
        
    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        s = deepcopy(state)
        s_tp1 = deepcopy(next_state)
        if tuple(s) not in self.model.keys():
            self.model[tuple(s)] = dict()
        self.model[tuple(s)][action] = [list(s_tp1), reward]
        
    
    # randomly sample from previous experience
    def sample(self):
        state_index = np.random.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = np.random.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        
        next_state, reward = self.model[state][action]
        
        s = deepcopy(state)
        s_tp1 = deepcopy(next_state)
        
        return list(s), action, list(s_tp1), reward


# Time-based model for planning in Dyna-Q+
class TimeModel(object):
    def __init__(self, maze, time_weight=1e-4):
        '''
        @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
        @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
        '''
        self.model = dict()
        
        # track the total time 
        self.time = 0
        
        self.kappa = time_weight
        self.maze = maze
        
        
    def feed(self, state, action, next_state, reward):
        s = deepcopy(state)
        s_tp1 = deepcopy(next_state)
        self.time += 1
        if tuple(s) not in self.model.keys():
            self.model[tuple(s)] = dict()
        
            # actions that had been never tried before from a state were allowed to be considered in the planning phase
            for action_ in self.maze.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of 0
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[tuple(s)][action_] = [list(s), 0, 1]
       
        self.model[tuple(s)][action] = [list(s_tp1), reward, self.time]
        
    
    def sample(self):
        state_index = np.random.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = np.random.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        
        next_state, reward, time = self.model[state][action]
        
        # adjust reward with elapsed tmie since last visit
        reward += self.kappa * np.sqrt(self.time - time)
        
        s = deepcopy(state)
        s_tp1 = deepcopy(next_state)
        
        return list(s), action, list(s_tp1), reward




        
        
            
def dyna_q(q_value, model, maze, dyna_params):
    '''
    play for an episode for Dyna-Q algorithm
    @q_value: state action pair values, will be updated
    @model: model instance for planning
    @maze: a maze instance containing all information about the environment
    @dyna_params: several params for the algorithm
    '''
    
    state = maze.START_STATE
    steps = 0
    
    while state not in maze.GOAL_STATES:
        steps += 1
        
        action = epsilon_greedy(state, q_value, maze, dyna_params)
        
        next_state, reward = maze.step(state, action)
        
        # Q-learning update
        q_value[state[0], state[1], action] += \
        dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) 
                             - q_value[state[0], state[1], action])
            
        # feed the model with experience
        model.feed(state, action, next_state, reward)
        
        
        # sample experience from the model
        for n in range(dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
            dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) 
                                 - q_value[state_[0], state_[1], action_])
            
        state = next_state
        
        
        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break
        
        
    return steps
            
        

def figure_8_2(runs=30, episodes=50):
    dyna_maze = Maze()
    dyna_params = DynaParams()
    
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))
    
    for run in tqdm(range(runs)):
        
        for i, planning_step in enumerate(planning_steps):
            np.random.seed(run)
            dyna_params.planning_steps = planning_step
            q_value = np.zeros(dyna_maze.q_size)
            
            model = TrivialModel()
            for ep in range(episodes):
                steps[i, ep] += dyna_q(q_value, model, dyna_maze, dyna_params)  # 一开始这里忘了“+”
                
                
    steps /= runs
    
    for i in range(len(planning_steps)):
        plt.plot(steps[i, 1:], label=f'{planning_steps[i]} planning steps') # 文中是从episode=2开始的
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()            
            

figure_8_2(runs=30, episodes=50)            
            
      
               



# wrapper function for changing maze          
def changing_maze(maze, dyna_params):
    
    max_steps = maze.max_steps
    
    # track the cumulative reward
    rewards = np.zeros((dyna_params.runs, 2, max_steps))   # 2 represents Dyna-Q and Dyna-Q+
    
    for run in tqdm(range(dyna_params.runs)):
        models = [TrivialModel(), TimeModel(maze, time_weight=dyna_params.time_weight)]
        
        q_values = [np.zeros(maze.q_size), np.zeros(maze.q_size)]
        
        for i in range(len(dyna_params.methods)):
            # print('run:', run, dyna_params.methods[i])
            
            # set old obstacles for the maze
            maze.obstacles = maze.old_obstacles
            
            step = 0
            last_step = step
            while step < max_steps:
                # play for an episode
                step += dyna_q(q_values[i], models[i], maze, dyna_params)
                
                # update cumulative reward, 挺巧妙的
                rewards[run, i, last_step:step] = rewards[run, i, last_step]
                rewards[run, i, min(step, max_steps - 1)] = rewards[run, i, last_step] + 1
                last_step = step
                
                if step > maze.obstacle_switch_time:
                    # change the obstacles
                    maze.obstacles = maze.new_obstacles
                    
                    
                    
    # averaging over runs
    rewards = rewards.mean(axis=0)
    
    return rewards



def figure_8_4():
    blocking_maze = Maze()
    blocking_maze.START_STATE = [5, 3]
    blocking_maze.GOAL_STATES = [[0, 8]]
    blocking_maze.old_obstacles = [[3, i] for i in range(0, 8)]
    
    # new obstalces will block the optimal path
    blocking_maze.new_obstacles = [[3, i] for i in range(1, 9)]
    
    # step limit
    blocking_maze.max_steps = 3000
    
    
    blocking_maze.obstacle_switch_time = 1000
    
    # set up parameters
    dyna_params = DynaParams()
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 10
    dyna_params.runs = 20
        
    # kappa must be small, as the reward for getting the goal is only 1
    dyna_params.time_weight = 1e-4
    
    
    rewards = changing_maze(blocking_maze, dyna_params)
    
    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()
    
    
figure_8_4()
    
    
    
def figure_8_5():
    # set up a shortcut maze instance
    shortcut_maze = Maze()
    shortcut_maze.START_STATE = [5, 3]
    shortcut_maze.GOAL_STATES = [[0, 8]]
    shortcut_maze.old_obstacles = [[3, i] for i in range(1, 9)]

    # new obstacles will have a shorter path
    shortcut_maze.new_obstacles = [[3, i] for i in range(1, 8)]

    # step limit
    shortcut_maze.max_steps = 6000

    shortcut_maze.obstacle_switch_time = 3000

    # set up parameters
    dyna_params = DynaParams()

    # 50-step planning
    dyna_params.planning_steps = 50
    dyna_params.runs = 5
    dyna_params.time_weight = 1e-3
    dyna_params.alpha = 1.0

    # play
    rewards = changing_maze(shortcut_maze, dyna_params)

    for i in range(len(dyna_params.methods)):
        plt.plot( rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()
   
    
figure_8_5()



                

#     max_steps = maze.max_steps
    
#     # track the cumulative reward
#     rewards = np.zeros((dyna_params.runs, 2, max_steps))   # 2 reprensents Dyna-Q and Dyna-Q+
    
#     for run in tqdm(range(dyna_params.runs)):
#         models = [TrivialModel(), TimeModel(maze, time_weight=dyna_params.time_weight)]
        
#         q_values = [np.zeros(maze.q_size), np.zeros(maze.q_size)]
        
#         for i in range()
class PriorityQueue(object):
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0
        
    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)
        
        
    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
        
    
    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')
        
    
    def empty(self):
        return not self.entry_finder
    
    

# Model containing a priority queue for Prioritized Sweeping
class PriorityModel(TrivialModel):
    def __init__(self):
        super(PriorityModel, self).__init__()
        
        # maintain a priority queue
        self.priority_queue = PriorityQueue()
        
        # track predecessors for every state
        self.predecessors = dict()
        
        
    def insert(self, priority, state, action):
        # note the priority queue is a minimum heap, so we use -priority
        self.priority_queue.add_item((tuple(state), action), -priority)
        
    def empty(self):
        return self.priority_queue.empty()
        
        
    def sample(self):
        (state, action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[state][action]
        s = deepcopy(state)
        s_tp1 = deepcopy(next_state)
        return -priority, list(s), action, list(s_tp1), reward
    
    
    def feed(self, state, action, next_state, reward):
        s = deepcopy(state)
        s_tp1 = deepcopy(next_state)
        TrivialModel.feed(self, s, action, s_tp1, reward)
        
        if tuple(s_tp1) not in self.predecessors.keys():
            self.predecessors[tuple(s_tp1)] = set()
        self.predecessors[tuple(s_tp1)].add((tuple(s), action))
        
        
    def predecessor(self, state):
        if tuple(state) not in self.predecessors.keys():
            return []
        
        predecessors = []
        for state_pre, action_pre in list(self.predecessors[tuple(state)]):
            predecessors.append([list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
            
        return predecessors
        
    
    
            
def prioritized_sweeping(q_value, model, maze, dyna_params):       
    '''
    play for an episode for prioritized sweeping algorithm
    @q_value: state action pair values, will be updated
    @model: model instance for planning
    @maze: a maze instance containing all information about the environment
    @dyna_params: several params for the algorithm
    @return: # of backups during this episode
    '''
    state = maze.START_STATE

    # track the steps in this episode
    steps = 0

    # track the backups in planning phase
    backups = 0

    while state not in maze.GOAL_STATES:
        steps += 1

        # get action
        action = epsilon_greedy(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # get the priority for current state action pair
        priority = np.abs(reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                          q_value[state[0], state[1], action])

        if priority > dyna_params.theta:
            model.insert(priority, state, action)

        # start planning
        planning_step = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planning_step < dyna_params.planning_steps and not model.empty():
            # get a sample with highest priority from the model
            priority, state_, action_, next_state_, reward_ = model.sample()

            # update the state action value for the sample
            delta = reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - \
                    q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta

            # deal with all the predecessors of the sample state
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(reward_pre + dyna_params.gamma * np.max(q_value[state_[0], state_[1], :]) -
                                  q_value[state_pre[0], state_pre[1], action_pre])
                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)
            planning_step += 1

        state = next_state

        # update the # of backups
        backups += planning_step + 1

    return backups
            


                        
# Check whether state-action values are already optimal
def check_path(q_values, maze):
    # get the length of optimal path
    # 14 is the length of optimal path of the original maze
    # 1.2 means it's a relaxed optifmal path
    max_steps = 14 * maze.resolution * 1.2
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        action = np.argmax(q_values[state[0], state[1], :])
        state, _ = maze.step(state, action)
        steps += 1
        if steps > max_steps:
            return False
    return True




def example_8_4(runs=5):
    original_maze = Maze()
    
    # get the original 6 * 9 maze
    original_maze = Maze()

    # set up the parameters for each algorithm
    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95

    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, params_dyna]

    # set up models for planning
    models = [PriorityModel, TrivialModel]
    method_names = ['Prioritized Sweeping', 'Dyna-Q']
    
    
    # due to limitation of my machine, only perform experiments for 5 mazes
    # assuming the 1st maze has w * h states, then k-th maze has (w * k) * (h * k) states
    num_of_mazes = 5
    
    
    # build all the mazes
    mazes = [original_maze.extend_maze(i) for i in range(1, num_of_mazes + 1)]
    methods = [prioritized_sweeping, dyna_q]
    
    
    # track the # of backups
    backups = np.zeros((runs, 2, num_of_mazes))
    
    for run in range(runs):
        for i in range(len(method_names)):
            for maze_index, maze in enumerate(mazes):
                print(f'run {run}, {method_names[i]}, maze size {maze.MAZE_HEIGHT * maze.MAZE_WIDTH}')
                
                q_value = np.zeros(maze.q_size)
                
                # track steps/backups for each episode
                steps = []
                
                model = models[i]()
                
                
                while True:
                    steps.append(methods[i](q_value, model, maze, params[i]))
                    
                    
                    # check whether the (relaxed) optimal path is found
                    if check_path(q_value, maze):
                        break
                    
                # update the total steps/backups for this maze
                backups[run, i, maze_index] = np.sum(steps)
                
    backups = backups.mean(axis=0)
    
    
    # Dyna-Q performs several backups per step, 这里是一个容易弄错的地方
    backups[1, :] *= params_dyna.planning_steps + 1
    
    for i in range(len(method_names)):
        plt.plot(np.arange(1, num_of_mazes + 1), backups[i, :], label=method_names[i])
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution') 
    plt.yscale('log')
    plt.legend()
    


example_8_4(5)
    
                    
            




























