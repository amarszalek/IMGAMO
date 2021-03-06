# -*- coding: utf-8 -*-

import numpy as np
import imgamo as mo
import math

__author__ = "amarszalek"


def obj_func1(x):
    return (-10.0*np.exp(-0.2 * np.sqrt(x[:-1]**2 + x[1:]**2))).sum()


def obj_func2(x):
    return (np.abs(x)**0.8 + 5.0*np.sin(x**3)).sum()


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
OPTIONS = mo.IMGAMOOptions(population_size=25, max_iterations=100, clone_number=25, exchange_iter=1,
                           distance_level_f=0.05, distance_level_x=0.01, mutate=hiper_mutate)

OBJ_NUMS = 2
VAR_NUMS = 3
OBJ_FUNCS = [obj_func1, obj_func2]
OBJ_ARGS = [(), ()]
BOUNDS = tuple((-5.0, 5.0) for i in range(VAR_NUMS)) + (0.1,)
PROBLEM = mo.IMGAMOProblem(OBJ_NUMS, VAR_NUMS, OBJ_FUNCS, OBJ_ARGS, BOUNDS)

solver = mo.IMGAMOAlgorithm(PROBLEM, OPTIONS)

if __name__ == '__main__':
    res = solver.run_algorithm()
    res.plot_2d(0, 1)