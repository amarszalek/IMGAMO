# -*- coding: utf-8 -*-

import numpy as np
import imgamo as mo
import math

__author__ = "amarszalek"

MEAN = np.array([0.1, 0.08, 0.04])
COV = np.array([[0.25, 0.0, -0.1],
               [0.0, 0.16, 0.05],
               [-0.1, 0.05, 0.0625]])


def obj_mean(x, mean):
    return -np.dot(x, mean)


def obj_var(x, cov):
    return np.dot(x, np.dot(cov,x))


def portfolio_init(n_var, bounds, ndigits):
    if bounds == 1:
        ran = np.random.random(n_var)
    else:
        ran = 2.0 * np.random.random(n_var) - 1.0
    ind = ran/np.sum(ran)
    return np.round(ind, ndigits)


def portfolio_mutate(individual, pattern, bounds, ndigits):
    #print(pattern)
    sum_pattern = np.sum(individual[pattern])
    if bounds == 1:
        ran = np.random.random(len(pattern))
    else:
        ran = 2.0 * np.random.random(len(pattern)) - 1.0
    ran = (ran * sum_pattern) / np.sum(ran)
    #print(ran)
    for i, k in enumerate(pattern):
        individual[k] = round(ran[i], ndigits)
    return individual


OPTIONS = mo.IMGAMOOptions(population_size=50, max_iterations=100, clone_number=25, exchange_iter=8,
                           distance_level=0.05, ndigits=12, individual_init=portfolio_init, mutate=portfolio_mutate)
OBJ_NUMS = 2
VAR_NUMS = len(MEAN)
OBJ_FUNCS = [obj_mean, obj_var]
OBJ_ARGS = [(MEAN,), (COV,)]
BOUNDS = -1 # 1: only long, -1: long and short
PROBLEM = mo.IMGAMOProblem(OBJ_NUMS, VAR_NUMS, OBJ_FUNCS, OBJ_ARGS, BOUNDS)

solver = mo.IMGAMOAlgorithm(PROBLEM, OPTIONS)

if __name__ == '__main__':
    res = solver.run_algorithm()
    res.plot_2d(1, 0)