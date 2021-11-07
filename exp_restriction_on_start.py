#!/usr/bin/python 
# -*-coding:utf-8 -*-
import random
from simple_ga import MyGAForTSP
from util import plot_path

if __name__ == '__main__':
    data = [[0.3642, 0.7770], [0.7185, 0.8312], [0.0986, 0.5891], [0.2954, 0.9606], [0.5951, 0.4647],
            [0.6697, 0.7657], [0.4353, 0.1709], [0.2131, 0.8349], [0.3479, 0.6984], [0.4516, 0.0488]]
    data = [(random.random(), random.random()) for i in range(20)]
    tsp = MyGAForTSP(pop_size=20, mutation_rate=0.05, crossover_rate=0.9, max_generation=500, back_to_start=False)
    rec = tsp.run(data)
    plot_path(rec, data, back_to_start=False)