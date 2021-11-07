#!/usr/bin/python 
# -*-coding:utf-8 -*-
import random
import math
import numpy as np


class MyGAForTSP():
    def __init__(self, pop_size, mutation_rate=0.05, crossover_rate=0.8, max_generation=500, back_to_start=True):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generation = max_generation
        self.back_to_start = back_to_start
        self.city_n = 0
        self.pop = []
        self.data = []
        self.fitness = []
        self.best_fitness = 0
        self.best_individual = []
    
    def run(self, data):
        self.data = data
        self.city_n = len(data)
        running_rec = []
        init_chromosome = [i for i in range(self.city_n)]
        self.pop = []
        for i in range(self.pop_size):
            random.shuffle(init_chromosome)
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
            fitnesses, best_individual, best_fitness = self.evaluation_fitness(children)
            running_rec.append((best_individual, best_fitness))
            sorted_children = sorted([(c, f) for c, f in zip(children, fitnesses)], key=lambda x: x[1])[:self.pop_size]
            self.pop = [c[0] for c in sorted_children]
        return running_rec
        
    def cal_distance(self, chromosome):
        dis = 0
        for i in range(1, self.city_n):
            dis += math.sqrt((self.data[chromosome[i]][0] - self.data[chromosome[i - 1]][0]) ** 2 +
                             (self.data[chromosome[i]][1] - self.data[chromosome[i - 1]][1]) ** 2)
        if self.back_to_start:
            dis += math.sqrt((self.data[chromosome[0]][0] - self.data[chromosome[-1]][0]) ** 2 +
                             (self.data[chromosome[0]][1] - self.data[chromosome[-1]][1]) ** 2)
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
        index1 = random.randint(0, self.city_n-1)
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
    data = [[0.3642, 0.7770], [0.7185, 0.8312], [0.0986, 0.5891], [0.2954, 0.9606], [0.5951, 0.4647],
            [0.6697, 0.7657], [0.4353, 0.1709], [0.2131, 0.8349], [0.3479, 0.6984], [0.4516, 0.0488]]
    # tsp = MyGAForTSP(pop_size=10, mutation_rate=0.05, crossover_rate=0.8, max_generation=100)
    # rec = tsp.run(data)
    # plot_path(rec, data)
    for p in [5, 10, 20, 30, 40, 50]:
        res = []
        for _ in range(5):
            tsp = MyGAForTSP(pop_size=p, mutation_rate=0.05, crossover_rate=0.8, max_generation=100)
            rec = tsp.run(data)
            res.append(rec[-1][1])
        print('pop size', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))
    for p in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7]:
        res = []
        for _ in range(5):
            tsp = MyGAForTSP(pop_size=20, mutation_rate=p, crossover_rate=0.8, max_generation=100)
            rec = tsp.run(data)
            res.append(rec[-1][1])
        print('mutation rate', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))
    for p in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        res = []
        for _ in range(5):
            tsp = MyGAForTSP(pop_size=20, mutation_rate=0.05, crossover_rate=p, max_generation=100)
            rec = tsp.run(data)
            res.append(rec[-1][1])
        print('crossover rate', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))