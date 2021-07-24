# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:21:08 2021

@author: leyuan

reference:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter08/expectation_vs_sample.py
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm


'''
这里我们对问题做出一些假定，
我们假设从当前状态s采取动作a后可能转移到的b个分支的value是来自N(0,1)
，并且根据书中的要求，采样跟新的步长是1/t，即求期望，所有我们不采用incremental mean而是直接用样本均值来表示更新
'''



# for figure 8.7, run a simulation of 2 * @b steps
def b_steps(b):
    # set the value of the next b states
    # it is not clear how to set this
    distribution = np.random.randn(b)

    # true value of the current state
    true_v = np.mean(distribution)

    samples = []
    errors = []

    # sample 2b steps
    for t in range(2 * b):
        v = np.random.choice(distribution)
        samples.append(v)
        errors.append(np.abs(np.mean(samples) - true_v))

    return errors

def figure_8_7():
    runs = 100
    branch = [2, 10, 100, 1000]
    for b in branch:
        errors = np.zeros((runs, 2 * b))
        for r in tqdm(np.arange(runs)):
            errors[r] = b_steps(b)
        errors = errors.mean(axis=0)
        x_axis = (np.arange(len(errors)) + 1) / float(b)
        plt.plot(x_axis, errors, label='b = %d' % (b))

    plt.xlabel('number of computations')
    plt.xticks([0, 1.0, 2.0], ['0', 'b', '2b'])
    plt.ylabel('RMS error')
    plt.legend()

if __name__ == '__main__':
    figure_8_7()