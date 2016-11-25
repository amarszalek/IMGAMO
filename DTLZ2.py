# -*- coding: utf-8 -*-

import numpy as np
import imgamo as mo
import math
import functools
import operator

__author__ = "amarszalek"


OBJ_NUMS = 3
VAR_NUMS = 12
K_NUM = VAR_NUMS - OBJ_NUMS + 1


def g(x):
    return np.sum((x[VAR_NUMS-K_NUM:]-0.5)**2)


def obj_func1(x):
    f = 1.0 + g(x)
    f *= functools.reduce(operator.mul, np.cos(0.5 * math.pi * x[:OBJ_NUMS-1]), 1)
    return f


def obj_func2(x):
    f = 1.0 + g(x)
    f *= functools.reduce(operator.mul, np.cos(0.5 * math.pi * x[:OBJ_NUMS - 2]), 1)
    f *= math.sin(0.5 * math.pi * x[OBJ_NUMS - 2])
    return f


def obj_func3(x):
    f = 1.0 + g(x)
    f *= functools.reduce(operator.mul, np.cos(0.5 * math.pi * x[:OBJ_NUMS - 3]), 1)
    f *= math.sin(0.5 * math.pi * x[OBJ_NUMS - 3])
    return f


def hiper_mutate(ind, pattern, bounds):
    r = np.random.random()
    if r < 0.45:
        ind = mo.uniform_mutate_in(ind, pattern, bounds)
    elif r < 0.9:
        ind = mo.gaussian_mutate(ind, pattern, bounds)
    else:
        ind = mo.bound_mutate(ind, pattern, bounds)
    return ind


"""
population_size:  int,                 default: 100
max_evaluations:  int,                 default: -1
max_iterations:   int,                 default: 1000
exchange_iter:    int,                 default: 3
change_iter:      int,                 default: 3
clone_number:     int,                 default: 15
distance_level_f: float,               default: 0.05
distance_level_x: float,               default: 0.01
mutate:           callable (function), default: uniform_mutate
individual_init:  callable (function), default: create_individual
distance:         callable (function), default: euclidean_distance
verbose:          bool,                default: True
"""
OPTIONS = mo.IMGAMOOptions(population_size=25, max_evaluations=10000, clone_number=15, exchange_iter=1, change_iter=1,
                           distance_level_f=0.10, distance_level_x=0.01, mutate=hiper_mutate, verbose=False)


OBJ_FUNCS = [obj_func1, obj_func2, obj_func3]
OBJ_ARGS = [(), (), ()]
BOUNDS = tuple((0.0, 1.0) for i in range(VAR_NUMS)) + (0.1,)
PROBLEM = mo.IMGAMOProblem(OBJ_NUMS, VAR_NUMS, OBJ_FUNCS, OBJ_ARGS, BOUNDS)

solver = mo.IMGAMOAlgorithm(PROBLEM, OPTIONS)

if __name__ == '__main__':
    res = solver.run_algorithm()
    res.plot_3d(0,1,2)