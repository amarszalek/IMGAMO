# -*- coding: utf-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import time

__author__ = "amarszalek"


class IMGAMOOptions(object):
    def __init__(self, **kwargs):
        self.population_size = kwargs['population_size'] if 'population_size' in kwargs else 100
        self.max_evaluations = kwargs['max_evaluations'] if 'max_evaluations' in kwargs else -1
        self.max_iterations = kwargs['max_iterations'] if 'max_iterations' in kwargs else 1000
        self.exchange_iter = kwargs['exchange_iter'] if 'exchange_iter' in kwargs else 3
        self.change_iter = kwargs['change_iter'] if 'change_iter' in kwargs else 3
        self.clone_number = kwargs['clone_number'] if 'clone_number' in kwargs else 15
        self.distance_level_f = kwargs['distance_level_f'] if 'distance_level_f' in kwargs else 0.05
        self.distance_level_x = kwargs['distance_level_x'] if 'distance_level_x' in kwargs else 0.01
        self.mutate = kwargs['mutate'] if 'mutate' in kwargs else uniform_mutate
        self.exchange_repair = kwargs['exchange_repair'] if 'exchange_repair' in kwargs else None
        self.individual_init = kwargs['individual_init'] if 'individual_init' in kwargs else create_individual
        self.distance = kwargs['distance'] if 'distance' in kwargs else euclidean_distance
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else True


class IMGAMOProblem(object):
    def __init__(self, objectives_num, variables_num, objectives_funcs, objectives_args, bounds):
        self.objectives_num = objectives_num
        self.variables_num = variables_num
        self.objectives_funcs = objectives_funcs
        self.objectives_args = objectives_args
        self.bounds = bounds
        self.evaluation_count = np.zeros(objectives_num)

    def evaluate_population(self, pop, func_no):
        if pop.shape[1] != self.variables_num:
            raise ValueError('pop.shape[1] != variables_num')
        self.evaluation_count[func_no] += pop.shape[0]
        return np.array([self.objectives_funcs[func_no](ind, *self.objectives_args[func_no]) for ind in pop])

    def get_not_dominated(self, pop, distance, level, eval_pop=None, func_no=-1):
        eval_pop_full = np.zeros((pop.shape[0], self.objectives_num))
        for i in range(self.objectives_num):
            if i == func_no and eval_pop is not None:
                eval_pop_full[:, i] = eval_pop
            else:
                eval_pop_full[:, i] = self.evaluate_population(pop, i)
        to_remove = []
        for i in range(eval_pop_full.shape[0]):
            for j in range(eval_pop_full.shape[0]):
                if j not in to_remove and i != j and (eval_pop_full[i] >= eval_pop_full[j]).all():
                    to_remove.append(i)
                    break
        not_dominated = []
        not_dominated_eval = []
        for i in range(eval_pop_full.shape[0]):
            if i not in to_remove:
                not_dominated.append(pop[i])
                not_dominated_eval.append(eval_pop_full[i])
        to_remove = []
        for i in range(len(not_dominated_eval)):
            for j in range(len(not_dominated_eval)):
                if j not in to_remove and i != j and distance(not_dominated_eval[i], not_dominated_eval[j]) < level:
                    to_remove.append(i)
                    break
        not_dominated_s = []
        not_dominated_eval_s = []
        for i in range(len(not_dominated_eval)):
            if i not in to_remove:
                not_dominated_s.append(not_dominated[i])
                not_dominated_eval_s.append(not_dominated_eval[i])
        return np.array(not_dominated_s), np.array(not_dominated_eval_s)


class IMGAMOResult(object):
    def __init__(self):
        self.front = []
        self.evaluated_front = []
        self.front_size = len(self.front)
        self.evaluation_count = -1
        self.elapsed_time = 0

    def add_to_result(self, not_dominated, not_dominated_eval, distance, level):
        if self.front_size == 0:
            self.front = copy.deepcopy(not_dominated)
            self.evaluated_front = copy.deepcopy(not_dominated_eval)
            self.front_size = len(self.front)
        else:
            to_remove = []
            for i in range(len(not_dominated_eval)):
                for j in range(len(self.evaluated_front)):
                    if (not_dominated_eval[i] >= self.evaluated_front[j]).all():
                        to_remove.append(i)
                        break
            temp_ndom = []
            temp_ndom_eval = []
            for i in range(len(not_dominated_eval)):
                if i not in to_remove:
                    temp_ndom.append(not_dominated[i])
                    temp_ndom_eval.append(not_dominated_eval[i])
            temp_ndom = np.array(temp_ndom)
            temp_ndom_eval = np.array(temp_ndom_eval)
            to_remove = []
            for i in range(len(self.evaluated_front)):
                for j in range(len(temp_ndom_eval)):
                    if (self.evaluated_front[i] >= temp_ndom_eval[j]).all():
                        to_remove.append(i)
                        break
            new_front = []
            new_front_eval = []
            for i in range(len(self.evaluated_front)):
                if i not in to_remove:
                    new_front.append(self.front[i])
                    new_front_eval.append(self.evaluated_front[i])
            new_front_s = copy.deepcopy(new_front)
            new_front_eval_s = copy.deepcopy((new_front_eval))
            for i in range(len(temp_ndom_eval)):
                flag = True
                for j in range(len(new_front_eval)):
                    dist = distance(temp_ndom_eval[i], new_front_eval[j])
                    if dist < level:
                        flag = False
                        break
                if flag:
                    new_front_s.append(temp_ndom[i])
                    new_front_eval_s.append(temp_ndom_eval[i])
            self.front = np.array(new_front_s)
            self.evaluated_front = np.array(new_front_eval_s)
            self.front_size = len(self.front)

    def plot_2d(self, func_no1, func_no2, show=True):
        ax = plt.figure('Front Pareto', figsize=(6, 6)).add_subplot(1, 1, 1)
        data = np.array(self.evaluated_front)
        ax.scatter(data[:,func_no1], data[:,func_no2], marker='o', label='IMGAMO')
        ax.grid(True)
        ax.legend()
        if show:
            plt.show()
        return ax

    def plot_3d(self, func_no1, func_no2, func_no3, show=True):
        ax = plt.figure('Front Pareto', figsize=(6, 6)).add_subplot(111, projection='3d')
        data = np.array(self.evaluated_front)
        ax.scatter(data[:,func_no1], data[:,func_no2], data[:,func_no3], marker='o', label='IMGAMO')
        ax.grid(True)
        ax.legend()
        if show:
            plt.show()
        return ax


class IMGAMOPlayer(object):
    def __init__(self, player_id, problem, options, pattern):
        self.player_id = player_id
        self.problem = problem
        self.options = options
        self.population = create_population(options.population_size, problem.variables_num, problem.bounds,
                                            options.individual_init)
        self.evaluated_population = problem.evaluate_population(self.population, self.player_id)
        self.pattern = pattern

    def clonal_selection(self):
        temp_pop_eval = copy.deepcopy(self.evaluated_population)
        arg_sort = temp_pop_eval.argsort()
        eval_count = 0
        clone_num = self.options.clone_number
        for arg in arg_sort:
            temp_pop = np.array([self.options.mutate(copy.deepcopy(self.population[arg]), self.pattern,
                                                     self.problem.bounds) for _ in range(clone_num)])
            temp_eval = np.array([self.problem.objectives_funcs[self.player_id]
                                  (ind, *self.problem.objectives_args[self.player_id]) for ind in temp_pop])
            eval_count += temp_eval.shape[0]
            argmin = temp_eval.argmin()
            if temp_eval[argmin] < self.evaluated_population[arg]:
                self.population[arg] = temp_pop[argmin]
                self.evaluated_population[arg] = temp_eval[argmin]
            clone_num = clone_num - 1 if clone_num > 2 else 1
        return eval_count

    def suppression(self):
        dist_mat = distance_matrix(self.population, self.pattern, self.options.distance)
        computed = []
        eval_count = 0
        for k, row in enumerate(dist_mat):
            if k in computed:
                continue
            similar = [j for j in range(len(row)) if row[j] < self.options.distance_level_x and j not in computed]
            if len(similar) == 1:
                computed += similar
                continue
            eval_similar = [self.evaluated_population[j] for j in similar]
            arg_sort = np.argsort(eval_similar)
            new_ind = self.options.individual_init(self.problem.variables_num, self.problem.bounds)
            for k in self.pattern:
                self.population[similar[arg_sort[1]]][k] = new_ind[k]
            self.evaluated_population[similar[arg_sort[1]]] = self.problem.objectives_funcs[self.player_id]\
                (self.population[similar[arg_sort[1]]], *self.problem.objectives_args[self.player_id])
            computed += [similar[arg_sort[1]]]
            eval_count += 1
        return eval_count

    def set_pattern(self, pattern):
        self.pattern = copy.deepcopy(pattern)

    def get_best(self):
        arg_min = np.argmin(self.evaluated_population)
        return self.population[arg_min]

    def update_gens(self, best_inds, patterns):
        # update_gens
        for p in range(len(patterns)):
            if p == self.player_id:
                continue
            for i in patterns[p]:
                self.population[:, i] = best_inds[p][i]
        # for portfolio
        if self.options.exchange_repair is not None:
            for ind in self.population:
                ind = self.options.exchange_repair(ind)
        # reevaluate population
        self.evaluated_population = self.problem.evaluate_population(self.population, self.player_id)


class IMGAMOAlgorithm(object):
    def __init__(self, problem, options):
        st = time.process_time()
        random.seed()
        self.options = options
        self.problem = problem
        self.patterns = assigning_gens(self.problem.variables_num, self.problem.objectives_num)
        self.players = [IMGAMOPlayer(i, problem, options, self.patterns[i]) for i in range(self.problem.objectives_num)]
        self.result = IMGAMOResult()
        self.result.elapsed_time += time.process_time() - st

    def run_algorithm(self):
        # Main loop
        st = time.process_time()
        iteration = 0
        exchange_flag = False
        while iteration < self.options.max_iterations:
            # exchange condition
            if iteration % self.options.exchange_iter == 0 and iteration < self.options.max_iterations - 1:
                best_inds = []
                exchange_flag = True

            # for each player
            for player in self.players:
                # clonal selection
                self.problem.evaluation_count[player.player_id] += player.clonal_selection()
                # get not dominated
                not_dominated, not_dominated_eval = \
                    self.problem.get_not_dominated(player.population, self.options.distance,
                                                   self.options.distance_level_f, eval_pop=player.evaluated_population,
                                                   func_no=player.player_id)
                # add to result
                self.result.add_to_result(not_dominated, not_dominated_eval, self.options.distance,
                                          self.options.distance_level_f)

                # suppression
                self.problem.evaluation_count[player.player_id] += player.suppression()

                # get best individual
                if exchange_flag:
                    best_inds.append(player.get_best())

            # exchange gens
            if exchange_flag:
                for player in self.players:
                    player.update_gens(best_inds, self.patterns)
                exchange_flag = False

            # change patterns
            if iteration % self.options.change_iter == 0 and iteration < self.options.max_iterations - 1:
                new_patterns = assigning_gens(self.problem.variables_num, self.problem.objectives_num)
                for player in self.players:
                    player.set_pattern(new_patterns[player.player_id])
                self.patterns = new_patterns

            self.result.evaluation_count = self.problem.evaluation_count

            # stop condition
            if np.max(self.problem.evaluation_count) > self.options.max_evaluations > -1:
                break

            # verbose
            if self.options.verbose:
                print('Iteration: ', iteration)
                print('Evaluation_count: ', self.result.evaluation_count)
                print('Front size: ', self.result.front_size)
                print('')

            # incrementation
            iteration += 1
        self.result.elapsed_time += time.process_time() - st
        if self.options.verbose:
            print('Elapsed time:', self.result.elapsed_time)
        print('Elapsed time:', self.result.elapsed_time)
        return self.result


# assigning_gens
def assigning_gens(n_var, n_func):
    temp = np.arange(n_var)
    random.shuffle(temp)
    q, r = divmod(n_var, n_func)
    if r == 0:
        pat = [temp[i::n_func] for i in range(n_func)]
    else:
        temp_pat = [temp[i:-r:n_func] for i in range(n_func)]
        pat = copy.deepcopy(temp_pat)
        for i in range(1,r+1):
            ran = random.randint(0,n_func-1)
            pat[ran] = np.append(temp_pat[ran],temp[-i])
    return pat


def create_individual(n_var, bounds):
    ind = np.random.random(n_var)
    for k in range(n_var):
        if bounds[k][0] is None and bounds[k][1] is None:
            ind[k] = math.tan(math.pi * ind[k] - np.math.pi)
        elif bounds[k][0] is not None and bounds[k][1] is None:
            a = bounds[k][0]
            ind[k] = -math.log(ind[k]) + a
        elif bounds[k][0] is None and bounds[k][1] is not None:
            b = bounds[k][1]
            ind[k] = math.log(ind[k]) + b
        else:
            a = bounds[k][0]
            b = bounds[k][1]
            ind[k] = (b - a) * ind[k] + a
    return ind


def create_population(n_pop, n_var, bounds, init_func):
    pop = np.zeros((n_pop, n_var))
    for i in range(n_pop):
        pop[i] = init_func(n_var, bounds)
    return pop


def uniform_mutate(individual, pattern, bounds):
    i = np.random.choice(pattern)
    r = random.random()
    if bounds[i][0] is None and bounds[i][1] is None:
        r2 = random.random()
        individual[i] = individual[i] + r if r2 > 0 else individual[i] - r
    elif bounds[i][0] is not None and bounds[i][1] is None:
        a = bounds[i][0]
        d = a - individual[i] if r < 0.5 else 1.0
        individual[i] = individual[i] + d * r
    elif bounds[i][0] is None and bounds[i][1] is not None:
        b = bounds[i][1]
        d = b - individual[i] if r > 0.5 else -1.0
        individual[i] = individual[i] + d * r
    else:
        a = bounds[i][0]
        b = bounds[i][1]
        d = a - individual[i] if r < 0.5 else b - individual[i]
        individual[i] = individual[i] + d * r
    return individual


def uniform_mutate_in(individual, pattern, bounds):
    k = np.random.choice(pattern)
    individual[k] = random.random()
    if bounds[k][0] is None and bounds[k][1] is None:
        individual[k] = math.tan(math.pi * individual[k] - np.math.pi)
    elif bounds[k][0] is not None and bounds[k][1] is None:
        a = bounds[k][0]
        individual[k] = -math.log(individual[k]) + a
    elif bounds[k][0] is None and bounds[k][1] is not None:
        b = bounds[k][1]
        individual[k] = math.log(individual[k]) + b
    else:
        a = bounds[k][0]
        b = bounds[k][1]
        individual[k] = (b - a) * individual[k] + a
    return individual


def bound_mutate(individual, pattern, bounds):
    k = np.random.choice(pattern)
    if bounds[k][0] is None and bounds[k][1] is None:
        a = -100.0
        b = 100.0
    elif bounds[k][0] is not None and bounds[k][1] is None:
        a = bounds[k][0]
        b = a + 100.0
    elif bounds[k][0] is None and bounds[k][1] is not None:
        b = bounds[k][1]
        a = b - 100.0
    else:
        a = bounds[k][0]
        b = bounds[k][1]
    r = np.random.random()
    if r < 0.5:
        r = np.random.random()
        individual[k] = a + (individual[k] - a) * r
    else:
        r = np.random.random()
        individual[k] = (b - individual[k]) * r + individual[k]
    return individual


def gaussian_mutate(individual, pattern, bounds):
    sigma = bounds[len(individual)]
    k = np.random.choice(pattern)
    if bounds[k][0] is None and bounds[k][1] is None:
        a = -100.0
        b = 100.0
    elif bounds[k][0] is not None and bounds[k][1] is None:
        a = bounds[k][0]
        b = a + 100.0
    elif bounds[k][0] is None and bounds[k][1] is not None:
        b = bounds[k][1]
        a = b - 100.0
    else:
        a = bounds[k][0]
        b = bounds[k][1]
    ran = sigma * np.random.randn() + individual[k]
    if a < ran < b:
        individual[k] = ran
    return individual


def euclidean_distance(x, y):
    z = (x - y)**2
    return np.sqrt(z.sum())


def distance_matrix(pop, pattern, distance):
    n = pop.shape[0]
    dist_mat = np.zeros((n, n))
    for i in range(n):
        vec1 = pop[i][pattern]
        for j in range(i, n):
            vec2 = pop[j][pattern]
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                dist_mat[i, j] = distance(vec1, vec2)
                dist_mat[j, i] = dist_mat[i, j]
    return dist_mat
