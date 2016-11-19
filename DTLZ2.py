# -*- coding: utf-8 -*-

import numpy as np
import imgamo as mo
import math

__author__ = "amarszalek"


def g(x):
    return ((x[2:]-0.5)**2).sum()


def obj_func1(x):
    return (1 + g(x))*math.cos(x[0]*(math.pi/2.0))*math.cos(x[1]*(math.pi/2.0))


def obj_func2(x):
    return (1 + g(x))*math.cos(x[0]*(math.pi/2.0))*math.sin(x[1]*(math.pi/2.0))


def obj_func3(x):
    return (1 + g(x))*math.sin(x[0]*(math.pi/2.0))


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
OPTIONS = mo.IMGAMOOptions(population_size=50, max_iterations=100, clone_number=25, exchange_iter=5,
                           distance_level=0.05, ndigits=12)

OBJ_NUMS = 3
VAR_NUMS = 12
OBJ_FUNCS = [obj_func1, obj_func2, obj_func3]
OBJ_ARGS = [(), (), ()]
BOUNDS = tuple((0.0, 1.0) for i in range(VAR_NUMS))
PROBLEM = mo.IMGAMOProblem(OBJ_NUMS, VAR_NUMS, OBJ_FUNCS, OBJ_ARGS, BOUNDS)

solver = mo.IMGAMOAlgorithm(PROBLEM, OPTIONS)

if __name__ == '__main__':
    res = solver.run_algorithm()
    res.plot_2d(0, 1)
    res.plot_3d(0,1,2)