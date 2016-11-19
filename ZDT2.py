# -*- coding: utf-8 -*-

import numpy as np
import imgamo as mo
import math

__author__ = "amarszalek"


def obj_func1(x):
    return x[0]


def fun_g(x):
    n = len(x)
    return 1.0 + 9.0 * np.sum(x[1:]) / (n - 1.0)


def obj_func2(x):
    g = fun_g(x)
    return g * (1.0 - (x[0]/g)**2)


"""
population_size:  int,                 default: 100
max_evaluations:  int,                 default: -1
max_iterations:   int,                 default: 1000
exchange_iter:    int,                 default: 3
clone_number:     int,                 default: 15
distance_level:   float,               default: 0.01
mutate:           callable (function), default: uniform_mutate
individual_init:  callable (function), default: create_individual
distance:         callable (function), default: euclidean_distance
ndigits:          int,                 default: 8
verbose:          bool,                default: True
"""
OPTIONS = mo.IMGAMOOptions(population_size=100, max_iterations=500, clone_number=50, exchange_iter=1,
                           distance_level=0.05, ndigits=12)

OBJ_NUMS = 2
VAR_NUMS = 2
OBJ_FUNCS = [obj_func1, obj_func2]
OBJ_ARGS = [(), ()]
BOUNDS = tuple((0.0, 1.0) for i in range(VAR_NUMS))
PROBLEM = mo.IMGAMOProblem(OBJ_NUMS, VAR_NUMS, OBJ_FUNCS, OBJ_ARGS, BOUNDS)

solver = mo.IMGAMOAlgorithm(PROBLEM, OPTIONS)

if __name__ == '__main__':
    res = solver.run_algorithm()
    res.plot_2d(0, 1)

