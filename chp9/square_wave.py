# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:34:46 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter09/square_wave.py
"""


import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm



class Interval(object):
    '''
    wrapper class for an interval
    '''
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
        
    def contain(self, x):
        '''
        whether a point is in this interval
        '''
        return self.left <= x < self.right
    
    def size(self):
        '''
        length of this interval
        '''
        return self.right - self.left
    


# domain of the square wave, [0, 2)
DOMAIN = Interval(0.0, 2.0)


# square wave function
def square_wave(x):
    if 0.5 < x < 1.5:
        return 1
    return 0


def sample(n):
    '''
    get @n samples randomly from the square wave
    '''
    samples = []
    for i in range(n):
        x = np.random.uniform(DOMAIN.left, DOMAIN.right)
        y = square_wave(x)
        samples.append([x, y])
    return samples




class ValueFunction(object):
    '''
    wrapper class for value function
    '''
    def __init__(self, feature_width, domain=DOMAIN, alpha=0.2, num_features=50):
        '''
        Parameters
        ----------
        feature_width : TYPE, float
        domain : 
            domain of this function, an instance of Interval
        alpha : TYPE, float
            basic step size for one update. The default is 0.2.
        num_features : TYPE, int
            The default is 50.
        '''
        self.feature_width = feature_width
        self.num_featrues = num_features
        self.features = []
        self.alpha = alpha
        self.domain = domain
        
        # there are many ways to place those feature windows,
        # following is just one possible way
        step = (domain.size() - feature_width) / (num_features - 1)
        left = domain.left
        for i in range(num_features - 1):
            self.features.append(Interval(left, left + feature_width))
            left += step
        self.features.append(Interval(left, domain.right))
        
        self.w = np.zeros(num_features)
        
        
    def get_active_features(self, x):
        '''
        for point @x, return the indices of corresponding feature windows
        '''
        active_features = []
        for i in range(len(self.features)):
            if self.features[i].contain(x):
                active_features.append(i)
        return active_features
    
    
    def value(self, x):
        '''
        estimate the value for point @x
        '''
        active_features = self.get_active_features(x)
        return np.sum(self.w[active_features])
    
    
    def update(self, delta, x):
        '''
        update weights given sample of point @x
        @delta: y - x
        '''
        active_features = self.get_active_features(x)
        delta *= self.alpha / len(active_features)
        for index in active_features:
            self.w[index] += delta
            


def approximate(samples, value_function):
    '''
    train @value_function with a set of samples @samples
    '''
    for x, y in samples:
        delta = y - value_function.value(x)
        value_function.update(delta, x)
        
        

def figure_9_8():
    num_of_samples = [10, 40, 160, 640, 2560, 10240]
    feature_widths = [0.2, 0.4, 1.0]
    
    plt.figure(figsize=(30, 20))
    axis_x = np.arange(DOMAIN.left, DOMAIN.right, 0.02)
    for index, num_of_sample in enumerate(num_of_samples):
        print(num_of_sample, 'samples')
        samples = sample(num_of_sample)
        value_functions = [ValueFunction(feature_width) for feature_width in feature_widths]    
        plt.subplot(2, 3, index + 1)
        plt.title(f'{num_of_sample} samples')
        for value_function in value_functions:
            approximate(samples, value_function)
            values = [value_function.value(x) for x in axis_x]
            plt.plot(axis_x, values, label=f'feature width {value_function.feature_width}')
        plt.legend()


figure_9_8()


































