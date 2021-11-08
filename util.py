#!/usr/bin/python 
# -*-coding:utf-8 -*-

def plot_path(rec, data, back_to_start=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2)
    path = rec[0][0]
    x = [data[i][0] for i in path] + ([data[path[0]][0]] if back_to_start else [])
    y = [data[i][1] for i in path] + ([data[path[0]][1]] if back_to_start else [])
    ax[0][0].plot(x, y)
    ax[0][0].set_title('Initial Solution')

    path = rec[len(rec)//2][0]
    x = [data[i][0] for i in path] + ([data[path[0]][0]] if back_to_start else [])
    y = [data[i][1] for i in path] + ([data[path[0]][1]] if back_to_start else [])
    ax[0][1].plot(x, y)
    ax[0][1].set_title('Solution After %d Generations' % (len(rec)//2))
    
    path = rec[-1][0]
    x = [data[i][0] for i in path] + ([data[path[0]][0]] if back_to_start else [])
    y = [data[i][1] for i in path] + ([data[path[0]][1]] if back_to_start else [])
    ax[1][0].plot(x, y)
    ax[1][0].set_title('Solution After %d Generations' % (len(rec)-1))
    
    ax[1][1].plot([c[1] for c in rec])
    ax[1][1].set_title('Fitness Curve')
    ax[1][1].text(len(rec)//2, (rec[0][-1]+rec[-1][1])/2, 'Distance of final solution %f' % (rec[-1][1]))
    plt.show()


def load_time_window_data(filename):
    data = []
    time_window = []
    with open(filename) as f:
        f.readline()
        for line in f:
            line = line.split()
            data.append((float(line[1]), float(line[2])))
            time_window.append((float(line[4]), float(line[5])))
    return data, time_window


def load_cluster_data(filename):
    data = []
    with open(filename) as f:
        f.readline()
        for line in f:
            line = line.split()
            data.append((float(line[0]), float(line[1])))
    return data


if __name__ == '__main__':
    d, t = load_time_window_data('Dataset/TSPTW_dataset.txt')
    print(len(d))
    # print(t)