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
        self.time_matrix = None
        self.city_n = 0
        self.pop = []
        self.data = []
        self.fitness = []
        self.best_fitness = 0
        self.best_individual = []
    
    def run(self, data, back_to_start=True, use_eula=True, sequential=None, time_window=None, cluster=None):
        self.data = data
        self.city_n = len(data)
        self.back_to_start = back_to_start
        self.use_eula = use_eula
        self.sequential = sequential
        self.time_window = time_window
        self.time_matrix = []
        for i in range(self.city_n):
            t = []
            for j in range(self.city_n):
                t.append(math.sqrt((self.data[i][0] - self.data[j][0]) ** 2 + (self.data[i][1] - self.data[j][1]) ** 2))
            self.time_matrix.append(t)
        # for i in range(self.city_n):
        #     for j in range(self.city_n):
        #         for k in range(self.city_n):
        #             if self.time_matrix[i][j] > self.time_matrix[i][k] + self.time_matrix[k][j]:
        #                 print(i, j, k, self.time_matrix[i][j], self.time_matrix[i][k] + self.time_matrix[k][j])
        #                 self.time_matrix[i][j] = self.time_matrix[i][k] + self.time_matrix[k][j]
        self.cluster = cluster
        
        running_rec = []
        init_chromosome = [i for i in range(self.city_n)]
        # chromo = '1 17 10 20 18 19 11 6 16 2 12 13 7 14 8 3 5 9 21 4 15'
        # chromo = '1 38 25 45 43 3 52 20 7 18 86 9 82 84 46 29 74 37 15 30 79 91 33 50 76 81 10 34 59 61 80 99 62 88 22 75 40 69 16 11 78 12 96 97 28 27 44 14 55 51 13 36 53 35 47 77 19 4 2 98 83 60 63 66 41 68 49 73 8 24 89 57 85 87 26 93 65 48 54 32 5 39 71 94 42 72 17 31 6 23 64 67 56 21 92 95 58 90 70 100 101'
        # chromo = [int(x)-1 for x in chromo.split()]
        # print(self.check_conditions(chromo))
        # exit()
        self.pop = []
        while len(self.pop) < self.pop_size:
            t = self.generate_chromosome()
            if self.check_conditions(t):
                self.pop.append(t)
            # print(len(self.pop))
            # random.shuffle(init_chromosome)
            # if self.check_conditions(init_chromosome):
            #     self.pop.append(init_chromosome.copy())
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
            # print('len(valid_children)', len(valid_children))
            while len(valid_children) < self.pop_size:
                chr = self.generate_chromosome()
                if self.check_conditions(chr):
                    valid_children.append(chr)
                # random.shuffle(init_chromosome)
                # if self.check_conditions(init_chromosome) and init_chromosome not in valid_children:
                #     valid_children.append(init_chromosome.copy())
            fitnesses, best_individual, best_fitness = self.evaluation_fitness(valid_children)
            running_rec.append((best_individual, best_fitness))
            sorted_children = sorted([(c, f) for c, f in zip(valid_children, fitnesses)], key=lambda x: x[1])[
                              :self.pop_size]
            self.pop = [c[0] for c in sorted_children]
            # print(gen, running_rec[-1]) if gen % 50 == 0 else None
        return running_rec
    
    def generate_chromosome(self):
        if self.cluster:
            cluster_order = [i for i in range(len(self.cluster))]
            random.shuffle(cluster_order)
            chromosome = []
            for cluster_id in cluster_order:
                path_in_cluster = list(self.cluster[cluster_id])
                random.shuffle(path_in_cluster)
                chromosome += path_in_cluster.copy()
            return chromosome
        if self.time_window:
            pool = set([i for i in range(self.city_n)])
            time_table = [0]
            cur_pool = []
            for x in pool:
                if self.time_window[x][0] <= time_table[0] <= self.time_window[x][1]:
                    cur_pool.append(x)
            chromosome = [random.sample(cur_pool, 1)[0]]
            pool.remove(chromosome[0])
            for i in range(1, self.city_n):
                cur_pool = []
                for x in pool:
                    if self.time_window[x][0] <= time_table[-1] + self.time_matrix[chromosome[-1]][x] <= self.time_window[x][1]:
                        cur_pool.append(x)
                if len(cur_pool) == 0:
                    break
                c = random.sample(cur_pool, 1)[0]
                time_table.append(time_table[-1] + self.time_matrix[chromosome[-1]][c])
                chromosome.append(c)
                pool.remove(c)
            # print(len(chromosome), self.city_n)
            if len(chromosome) == self.city_n:
                return chromosome
            
        chromosome = [i for i in range(self.city_n)]
        random.shuffle(chromosome)
        return chromosome.copy()
    
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
        # print('city    (x, y)    time     time window')
        # print(chromosome[0]+1,'\t', self.data[chromosome[0]], '%.5f'% time_table[0], self.time_window[chromosome[0]])
        for i in range(1, len(chromosome)):
            time_table.append(time_table[-1] + self.time_matrix[chromosome[i]][chromosome[i-1]])
            # print(chromosome[i]+1, '\t', self.data[chromosome[i]],'%.5f'% time_table[i], self.time_window[chromosome[i]])
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
            if index == len(chromosome):
                return True
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
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        i = random.randint(0, len(chromosome)-1)
        j = random.randint(0, len(chromosome)-1)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome.copy()
    
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
    
    data = [[0.3642, 0.7770], [0.7185, 0.8312], [0.0986, 0.5891], [0.2954, 0.9606], [0.5951, 0.4647],
            [0.6697, 0.7657], [0.4353, 0.1709], [0.2131, 0.8349], [0.3479, 0.6984], [0.4516, 0.0488]]
    tsp = MyGAForTSP(pop_size=10, mutation_rate=0.1, crossover_rate=0.7, max_generation=1000)
    # rec = tsp.run(data)
    # print(rec[-1])
    # plot_path(rec, data)
    
    ######################################### experiment on parameters ##########################
    for p in [5, 10, 20, 30, 40]:
        res = []
        for _ in range(5):
            tsp = MyGAForTSP(pop_size=p, mutation_rate=0.05, crossover_rate=0.8, max_generation=100)
            rec = tsp.run(data)
            res.append(rec[-1][1])
        print('pop size', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))
    for p in [0, 0.01, 0.05,  0.2, 0.5]:
        res = []
        for _ in range(5):
            tsp = MyGAForTSP(pop_size=20, mutation_rate=p, crossover_rate=0.8, max_generation=100)
            rec = tsp.run(data)
            res.append(rec[-1][1])
        print('mutation rate', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))
    for p in [0, 0.2, 0.4, 0.6, 0.8]:
        res = []
        for _ in range(5):
            tsp = MyGAForTSP(pop_size=20, mutation_rate=0.05, crossover_rate=p, max_generation=100)
            rec = tsp.run(data)
            res.append(rec[-1][1])
        print('crossover rate', p, 'distance %.5f %.5f' % (float(np.mean(res)), float(np.std(res))))
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
    # from util import load_time_window_data
    # data, time_window = load_time_window_data('Dataset/n100w60.002.txt')
    # tsp = MyGAForTSP(pop_size=10, mutation_rate=0.1, crossover_rate=0.7, max_generation=1000)
    # rec = tsp.run(data, time_window=time_window)
    # print(rec[-1])
    # plot_path(rec, data)
    ################################################################################################
    
    ################################# Cluster ################################################
    # from util import load_cluster_data
    # data = load_cluster_data('Dataset/Cluster_dataset.txt')
    # cluster = [set(), set(), set()]
    # for i in range(len(data)):
    #     k = (data[i][1]-11.8) / (data[i][0]-2)
    #     if k > 1:
    #         cluster[0].add(i)
    #     elif k > 0:
    #         cluster[1].add(i)
    #     else:
    #         cluster[2].add(i)
    # import matplotlib.pyplot as plt
    # plt.scatter([data[x][0] for x in cluster[0]], [data[x][1] for x in cluster[0]])
    # plt.scatter([data[x][0] for x in cluster[1]], [data[x][1] for x in cluster[1]])
    # plt.scatter([data[x][0] for x in cluster[2]], [data[x][1] for x in cluster[2]])
    # plt.show()
    # print(cluster)
    # tsp = MyGAForTSP(mutation_rate=0.1, crossover_rate=0.7, max_generation=1000)
    # rec = tsp.run(data=data, cluster=cluster)
    # print(rec[-1])
    # plot_path(rec, data)
    ##################################################################################################
    
