# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:32:43 2021

@author: leyuan

reference: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter02/ten_armed_testbed.py
"""


import matplotlib.pyplot as plt
import numpy as np




# ============= figure 2.1 ==============
bandit_arms = 10
q_star = np.random.randn(bandit_arms)
dat = np.random.randn(100, bandit_arms) + q_star

plt.violinplot(dat, showmeans=True)
plt.xlabel('Action')
plt.ylabel('reward distribution')
plt.show()



# ============= figure 2.2 ==============
bandit_arms = 10
bandit_num = 500
training_steps = 1000

def get_reward(q):
    return np.random.randn() + q

def epsilon_greedy(q, eps=0.1):
    q_max = np.max(q)
    dim = np.size(q)
    if np.random.rand() < eps:
        return np.random.choice(range(dim))
    else:
        return np.random.choice(np.where(q == q_max)[0])    # 处理最大值不止一个的情况

def update(q_old, target, alpha):
    return q_old + alpha * (target - q_old)


def bandit(eps):
    '''
    eps: epsilon-greedy method
    '''
    q_star = np.random.randn(bandit_arms)
    a_best = np.argmax(q_star)
    q = np.zeros(bandit_arms)
    reward = []
    best_action = []
    
    for step in range(training_steps):       
        a = epsilon_greedy(q, eps)
        r = get_reward(q_star[a])        
        q[a] = update(q[a], r, 1/(step+1))    # 步长为1/n
        reward.append(r)
        best_action.append(int(a == a_best))
        
        
    return reward, best_action

    
def run(eps):
    results = [bandit(eps) for _ in range(bandit_num)]
    mean_reward = np.array([results[i][0] for i in range(bandit_num)]).mean(axis=0)
    best_action_prob = np.array([results[i][1] for i in range(bandit_num)]).mean(axis=0)
    return mean_reward, best_action_prob


eps_list = [0, 0.01, 0.1]

dat = [run(eps) for eps in eps_list]
mean_rewards = [dat[i][0] for i in range(len(eps_list))]
best_action_probs = [dat[i][1] for i in range(len(eps_list))]

plt.subplot(2, 1, 1)
for eps, reward in zip(eps_list, mean_rewards):
    plt.plot(reward, label=f'$\epsilon$ = {eps}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.legend()

plt.subplot(2, 1, 2)
for eps, best_action_prob in zip(eps_list, best_action_probs):
    plt.plot(best_action_prob, label=f'$\epsilon$ = {eps}')
plt.xlabel('steps')
plt.ylabel('best_action_prob')
plt.legend()

plt.show()



# ============= figure 2.3 ==============
bandit_arms = 10
bandit_num = 500
training_steps = 1000

def get_reward(q):
    return np.random.randn() + q

def epsilon_greedy(q, eps=0.1):
    q_max = np.max(q)
    dim = np.size(q)
    if np.random.rand() < eps:
        return np.random.choice(range(dim))
    else:
        return np.random.choice(np.where(q == q_max)[0])

def update(q_old, target, alpha):
    return q_old + alpha * (target - q_old)


def bandit(eps, q_init):
    '''
    eps: epsilon-greedy method
    q_init: initialization of action-value
    '''
    q_star = np.random.randn(bandit_arms)
    a_best = np.argmax(q_star)
    q = np.zeros(bandit_arms) + q_init
    best_action = []
    
    for step in range(training_steps):       
        a = epsilon_greedy(q, eps)
        r = get_reward(q_star[a])        
        q[a] = update(q[a], r, 0.1)     # 步长为alpha=0.1
        best_action.append(int(a == a_best))
        
        
    return best_action

    
def run(eps, q_init):
    results = [bandit(eps, q_init) for _ in range(bandit_num)]
    best_action_prob = np.array(results).mean(axis=0)
    return best_action_prob


eps_list = [0, 0.1]
q_init_list = [5, 0]

best_action_probs = [run(eps, q_init) for eps, q_init in zip(eps_list, q_init_list)]


for eps, q_init, best_action_prob in zip(eps_list, q_init_list, best_action_probs):
    plt.plot(best_action_prob, label=f'$\epsilon$ = {eps}, q_init = {q_init}')
plt.xlabel('steps')
plt.ylabel('best_action_prob')
plt.legend()
plt.show()


# ============= figure 2.4 ==============
bandit_arms = 10
bandit_num = 500
training_steps = 1000

def get_reward(q):
    return np.random.randn() + q

def epsilon_greedy(q, eps=0.1):
    q_max = np.max(q)
    dim = np.size(q)
    if np.random.rand() < eps:
        return np.random.choice(range(dim))
    else:
        return np.random.choice(np.where(q == q_max)[0])
    
    
def ucb(q, step, count, c=2):
    m = q + c * np.sqrt(np.log(step+1) / (count + 1e-5))
    m_best = np.max(m)
    return np.random.choice(np.where(m == m_best)[0])


def update(q_old, target, alpha):
    return q_old + alpha * (target - q_old)


def bandit(eps=None, c=None):
    '''
    eps: epsilon-greedy method
    c: upper confidence bound method
    '''
    q_star = np.random.randn(bandit_arms)
    q = np.zeros(bandit_arms)
    count = np.zeros(bandit_arms)
    reward = []
    
    for step in range(training_steps): 
        
        if eps is not None:
            a = epsilon_greedy(q, eps)
        if c is not None:
            a = ucb(q, step, count, c)
            
        count[a] += 1
        r = get_reward(q_star[a])        
        q[a] = update(q[a], r, 1/(step+1))    # 步长为1/n
        reward.append(r)
        
        
    return reward

    
def run(eps=None, c=None):
    results = [bandit(eps, c) for _ in range(bandit_num)]
    mean_reward = np.array(results).mean(axis=0)
    return mean_reward


eps = 0.1
c = 1

eps_rewards = run(eps=eps)
ucb_rewards = run(c=c)


plt.plot(eps_rewards, label=f'$\epsilon$ = {eps}')
plt.plot(ucb_rewards, label=f'UCB c = {c}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.legend()




# ============= figure 2.5 ==============
bandit_arms = 10
bandit_num = 500
training_steps = 1000

def get_reward(q):
    return np.random.randn() + q


def get_action(pi):
    return np.random.choice(np.arange(bandit_arms), p=pi)
    # pi_max = np.max(pi)
    # return np.random.choice(np.where(pi == pi_max)[0]) 

def softmax(h):
    h_max = np.max(h)
    h = h - h_max
    return np.exp(h)/np.exp(h).sum()


def gradient(pi, a):
    one_hot = (np.arange(bandit_arms) == a).astype(np.float)
    return (one_hot - pi)


def update(h, r, grad, alpha):
    return h + alpha * r * grad


def gradient_bandit(alpha, baseline=True):
    '''
    alpha: stepsize parameter
    baseline: wheter to add a baseline for reward, default True
    '''
    q_star = np.random.randn(bandit_arms) + 4   # 书中是以N(4,1)生成的
    a_best = np.argmax(q_star)
    h = np.zeros(bandit_arms)
    pi = softmax(h)
    
    best_action = []
    mean_reward = 0
    
    for step in range(training_steps):
        a = get_action(pi)
        r = get_reward(q_star[a])   
        pi = softmax(h)
        grad = gradient(pi, a)
        
        mean_reward += (r - mean_reward) / (step+1)
        if baseline:
            r = r - mean_reward
            
        h = update(h, r, grad, alpha)
                
        best_action.append(int(a == a_best))
                
    return best_action

    
def run(alpha, baseline):
    results = [gradient_bandit(alpha, baseline) for _ in range(bandit_num)]
    best_action_prob = np.array(results).mean(axis=0)
    return best_action_prob



alpha_list = [0.1, 0.4]
baseline_list = [True, False]
x, y = np.meshgrid(alpha_list, baseline_list)


best_action_probs = [run(alpha, baseline) for alpha, baseline in zip(x.flatten(), y.flatten())]


for alpha, baseline, best_action_prob in zip(x.flatten(), y.flatten(), best_action_probs):
    plt.plot(best_action_prob, label=r'$\alpha$ = {}, baseline = {}'.format(alpha, baseline))
plt.xlabel('steps')
plt.ylabel('best_action_prob')
plt.title('True rewards are chosen to be near +4 rather than near 0')
plt.legend()
plt.show()



# ======================== figure 2.6 ======================================================
class Bandit(object):
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, 
                 UCB_param=None, gradient=False, gradient_baseline=False, true_reward=0.):
        
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        
    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward
        
        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial
        
        # times the action being selected
        self.action_count = np.zeros(self.k)
        
        self.best_action = np.argmax(self.q_true)
        
        self.time = 0
        
        
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        
        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + self.UCB_param * np.sqrt(np.log(self.time+1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])
        
        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)
        
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])
    

    
    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time
        
        if self.sample_averages:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward
    
    
def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in range(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] += 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards
        

     
def figure_2_6(runs=2000, time=1000):
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()
        
       
figure_2_6(runs=500, time=1000)        
        




# ========================练习2.5 10臂测试平台 非平稳=================================================================
import numpy as np 
import matplotlib.pyplot as plt 


def random_walk(x):
    dim=np.size(x)
    walk_set=[-1, 1, 0]
    for i in range(dim):
        x[i]=x[i]+np.random.choice(walk_set)
    return x
    # return x + np.random.normal(0, 0.01, x.shape)


def epsilon_greedy(q, eps=0.1):
    a = np.argmax(q)
    dim = np.size(q)
    if np.random.rand() < eps:
        return np.random.choice(range(dim))
    else:
        return int(a)


def update(q_old, target, alpha):

    return q_old + alpha * (target - q_old)


bandit_arms = 10
bandit_num = 500
training_steps = 1000

average_rewards_var = []
average_rewards_const = []
opt_action_probs_var = []
opt_action_probs_const = []


for bandit in range(bandit_num):
    
    q_star = np.zeros(bandit_arms)
    q_var = np.zeros(bandit_arms)
    q_const = np.zeros(bandit_arms)
    
    average_reward_var = []
    average_reward_const = []
    opt_action_prob_var = []
    opt_action_prob_const = []
    
    for step in range(training_steps):
        # random walk for bandit
        q_star = random_walk(q_star)
        a_opt = np.argmax(q_star)
        # 1/n 步长
        a_var = epsilon_greedy(q_var)
        reward_var = np.random.normal(q_star[a_var], 1)
        q_var[a_var] = update(q_var[a_var], reward_var, 1/(step+1))
        average_reward_var.append(reward_var)
        opt_action_prob_var.append(int(a_opt == a_var))
        
        # 固定步长        
        a_const = epsilon_greedy(q_const)        
        reward_const = np.random.normal(q_star[a_const], 1)       
        q_const[a_const] = update(q_const[a_const], reward_const, 0.1)
        average_reward_const.append(reward_const)
        opt_action_prob_const.append(int(a_opt == a_const))
        
    
    average_rewards_var.append(average_reward_var)
    average_rewards_const.append(average_reward_const)
    opt_action_probs_var.append(opt_action_prob_var)
    opt_action_probs_const.append(opt_action_prob_const)


average_rewards_var = np.array(average_rewards_var)
average_rewards_const = np.array(average_rewards_const)
average_rewards_var = average_rewards_var.mean(axis=0)     
average_rewards_const = average_rewards_const.mean(axis=0)   

plt.plot(average_rewards_var, 'b', label='sample average: $1/n$')
plt.plot(average_rewards_const, 'r--', label='constant stepsize: 0.1')
plt.legend()


opt_action_probs_var = np.array(opt_action_probs_var)
opt_action_probs_const = np.array(opt_action_probs_const)
opt_action_probs_var = opt_action_probs_var.mean(axis=0)     
opt_action_probs_const = opt_action_probs_const.mean(axis=0)         
        
plt.figure()
plt.plot(opt_action_probs_var, 'b', label='sample average: $1/n$')
plt.plot(opt_action_probs_const, 'r--', label='constant stepsize: 0.1')
plt.legend()

