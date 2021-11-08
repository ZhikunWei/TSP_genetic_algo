#!/usr/bin/python 
# -*-coding:utf-8 -*-
import random
import math
import numpy as np


class MyGAForTSP():
    def __init__(self, pop_size=20, mutation_rate=0.05, crossover_rate=0.8, max_generation=500, back_to_start=True,
                 use_eula=True):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generation = max_generation
        self.back_to_start = back_to_start
        self.use_eula = use_eula
        self.sequential = None
        self.time_window = None
        self.cluster = None
        self.city_n = 0
        self.pop = []
        self.data = []
        self.fitness = []
        self.best_fitness = 0
        self.best_individual = []
    
    def run(self, data, back_to_start=True, use_eula=True, sequential=None, time_window=None, cluster=None):
        self.data = data
        self.back_to_start = back_to_start
        self.use_eula = use_eula
        self.sequential = sequential
        self.time_window = time_window
        self.cluster = cluster
        self.city_n = len(data)
        running_rec = []
        init_chromosome = [i for i in range(self.city_n)]
        self.pop = []
        while len(self.pop) < self.pop_size:
            random.shuffle(init_chromosome)
            if self.check_conditions(init_chromosome):
                self.pop.append(init_chromosome.copy())
        fitnesses, best_individual, best_fitness = self.evaluation_fitness(self.pop)
        running_rec.append((best_individual, best_fitness))
        for gen in range(self.max_generation):
            children = []
            for i in range(len(self.pop)):
                for j in range(i):
                    children.append(self.cross(self.pop[i], self.pop[j]))
            for i in range(len(children)):
                children[i] = self.mutate(children[i])
            children.append(best_individual)
            valid_children = []
            for i in range(len(children)):
                if self.check_conditions(children[i]):
                    valid_children.append(children[i])
            valid_children = list(set(valid_children))
            while len(valid_children) < self.pop_size:
                random.shuffle(init_chromosome)
                if self.check_conditions(init_chromosome) and init_chromosome not in valid_children:
                    valid_children.append(init_chromosome.copy())
            fitnesses, best_individual, best_fitness = self.evaluation_fitness(valid_children)
            running_rec.append((best_individual, best_fitness))
            sorted_children = sorted([(c, f) for c, f in zip(valid_children, fitnesses)], key=lambda x: x[1])[
                              :self.pop_size]
            self.pop = [c[0] for c in sorted_children]
            if gen % 10 == 0:
                print(gen, rec[-1])
        return running_rec
    
    def check_conditions(self, chromosome):
        return self.check_sequential(chromosome) and self.check_time_window(chromosome) and self.check_cluster(chromosome)
    
    def check_sequential(self, chromosome):
        if not self.sequential:
            return True
        for c in self.sequential:
            index = chromosome.index(c)
            for post_c in self.sequential[c]:
                if post_c in chromosome[:index]:
                    return False
        return True
    
    def check_time_window(self, chromosome):
        if not self.time_window:
            return True
        time_table = [0]
        if not self.time_window[chromosome[0]][0] <= time_table[0] <= self.time_window[chromosome[0]][1]:
            return False
        for i in range(1, len(chromosome)):
            time_table.append(time_table[-1] +
                              math.sqrt((self.data[chromosome[i]][0] - self.data[chromosome[i - 1]][0]) ** 2 +
                                        (self.data[chromosome[i]][1] - self.data[chromosome[i - 1]][1]) ** 2))
            if not self.time_window[chromosome[i]][0] <= time_table[i] <= self.time_window[chromosome[i]][1]:
                return False
        return True
    
    def check_cluster(self, chromosome):
        if not self.cluster:
            return True
        cur_cluster = 0
        for i in range(len(self.cluster)):
            if chromosome[0] in self.cluster[i]:
                cur_cluster = i
        index = 0
        while index < len(chromosome):
            node_cnt = 0
            while node_cnt < len(self.cluster[cur_cluster]) and chromosome[index] in self.cluster[cur_cluster]:
                node_cnt += 1
                index += 1
            if node_cnt == len(self.cluster[cur_cluster]):
                for i in range(len(self.cluster)):
                    if chromosome[index] in self.cluster[i]:
                        cur_cluster = i
            else:
                return False
        return True
        
    def cal_distance(self, chromosome):
        dis = 0
        if self.use_eula:
            for i in range(1, self.city_n):
                dis += math.sqrt((self.data[chromosome[i]][0] - self.data[chromosome[i - 1]][0]) ** 2 +
                                 (self.data[chromosome[i]][1] - self.data[chromosome[i - 1]][1]) ** 2)
            if self.back_to_start:
                dis += math.sqrt((self.data[chromosome[0]][0] - self.data[chromosome[-1]][0]) ** 2 +
                                 (self.data[chromosome[0]][1] - self.data[chromosome[-1]][1]) ** 2)
        else:
            for i in range(1, self.city_n):
                x = min(chromosome[i], chromosome[i - 1])
                y = max(chromosome[i], chromosome[i - 1])
                dis += abs(self.data[x][0] - self.data[y][0]) + abs(
                    self.data[x][1] - self.data[y][1])
            if self.back_to_start:
                x = min(chromosome[0], chromosome[-1])
                y = max(chromosome[0], chromosome[-1])
                dis += abs(self.data[x][0] - self.data[y][0]) + abs(
                    self.data[x][1] - self.data[y][1])
        return dis
    
    def evaluation_fitness(self, pops):
        fitnesses = [self.cal_distance(chromosome) for chromosome in pops]
        best_individual_index = np.argmin(fitnesses)
        best_fitness = fitnesses[best_individual_index]
        best_individual = pops[best_individual_index]
        return fitnesses, best_individual, best_fitness
    
    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            for j in range(i):
                if random.random() < self.mutation_rate:
                    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome
    
    def cross(self, c1, c2):
        if random.random() > self.crossover_rate:
            return c1
        index1 = random.randint(0, self.city_n - 1)
        index2 = random.randint(index1, self.city_n)
        child = c1[index1:index2]
        child_set = set(child)
        for city in c2:
            if city not in child_set:
                child.append(city)
                child_set.add(city)
        return child


if __name__ == '__main__':
    from util import plot_path
    tsp = MyGAForTSP(max_generation=50)
    # data = [[0.3642, 0.7770], [0.7185, 0.8312], [0.0986, 0.5891], [0.2954, 0.9606], [0.5951, 0.4647],
    #         [0.6697, 0.7657], [0.4353, 0.1709], [0.2131, 0.8349], [0.3479, 0.6984], [0.4516, 0.0488]]
    # rec = tsp.run(data)
    # plot_path(rec, data)
    
    ######################################### experiment on parameters ##########################
    # for p in [5, 10, 20, 30, 40, 50]:
    #     res = []
    #     for _ in range(5):
    #         tsp = MyGAForTSP(pop_size=p, mutation_rate=0.05, crossover_rate=0.8, max_generation=100)
    #         rec = tsp.run(data)
    #         res.append(rec[-1][1])
    #     print('pop size', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))
    # for p in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7]:
    #     res = []
    #     for _ in range(5):
    #         tsp = MyGAForTSP(pop_size=20, mutation_rate=p, crossover_rate=0.8, max_generation=100)
    #         rec = tsp.run(data)
    #         res.append(rec[-1][1])
    #     print('mutation rate', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))
    # for p in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    #     res = []
    #     for _ in range(5):
    #         tsp = MyGAForTSP(pop_size=20, mutation_rate=0.05, crossover_rate=p, max_generation=100)
    #         rec = tsp.run(data)
    #         res.append(rec[-1][1])
    #     print('crossover rate', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))
    ###################################################################################################
    
    ################################## sequential order #########################################
    # data = [[0.3642, 0.7770], [0.7185, 0.8312], [0.0986, 0.5891], [0.2954, 0.9606], [0.5951, 0.4647],
    #         [0.6697, 0.7657], [0.4353, 0.1709], [0.2131, 0.8349], [0.3479, 0.6984], [0.4516, 0.0488]]
    # sequential = {3: [1, 2], 5: [3, 4]}
    # rec = tsp.run(data=data, sequential=sequential)
    # print(rec[-1])
    # plot_path(rec, data)
    ##############################################################################################
    
    ################################## Time Window ############################################
    from util import load_time_window_data
    data, time_window = load_time_window_data('Dataset/TSPTW_dataset.txt')
    rec = tsp.run(data, time_window=time_window)
    print(rec[-1])
    plot_path(rec, data)
    ################################################################################################
    
    ################################# Cluster ################################################
    from util import load_cluster_data
    data = load_cluster_data('Dataset/Cluster_dataset.txt')
    cluster = [set(), set(), set()]
    for i in range(len(data)):
        k = (data[i][1]-13) / (data[i][0]-6)
        if k > 1:
            cluster[0].add(i)
        elif k > 0:
            cluster[1].add(i)
        else:
            cluster[2].add(i)
    rec = tsp.run(data=data, cluster=cluster)
    print(rec[-1])
    plot_path(rec, data)
    ##################################################################################################
    
