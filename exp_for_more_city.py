#!/usr/bin/python 
# -*-coding:utf-8 -*-
import random
from simple_ga import MyGAForTSP
from util import plot_path


def exp_on_city_number(city_n=100):
    data = [(random.random(), random.random()) for i in range(city_n)]
    tsp = MyGAForTSP(pop_size=40, mutation_rate=0.5, crossover_rate=0.9, max_generation=2000)
    rec = tsp.run(data)
    plot_path(rec, data)


if __name__ == '__main__':
    exp_on_city_number(50)