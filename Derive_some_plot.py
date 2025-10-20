import os
import json
import matplotlib.pyplot as plt
import re
import numpy as np

def natural_sort_key(filename):
    # 使用正则表达式提取文件名中的数字，并返回数字和非数字部分的元组
    return [int(part) if part.isdigit() else part for part in re.split('([0-9]+)', filename)]

def read_objectives(directory):
    objectives = []
    # 使用自定义的排序键进行排序
    for filename in sorted(os.listdir(directory), key=natural_sort_key):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                objectives.append(data['objective'])
    return objectives


def plot_objective_changes(directories, names, line_value=None, target_index=None, num_samples=5):
    objectivess = []
    objectivess_ori = []
    for idx, directory in enumerate(directories):
        objectives = read_objectives(directory)
        objectivess.append(objectives)

        # 对指定的目标目录进行等间隔抽样
        if target_index is not None and idx == target_index:
            sampled_indices = np.linspace(0, len(objectives) - 1, num_samples, dtype=int)
            objectives = [objectives[i] for i in sampled_indices]
        objectivess_ori.append(objectives)

        print(objectives)
        plt.plot(objectives, marker='o', label=names[idx])

    if line_value is not None:
        plt.axhline(y=line_value, color='r', linestyle='--', label=f'Human designed Algorithm (GSPSO)')

    plt.title('Objective Value Changes Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return objectivess, objectivess_ori


# 使用示例
houzhui = ''
algos = ['hie', 'eoh']
# algos = ['sie_full_iteration1update', 'sie_noreflection_iteration1update', 'sie_full_iteration4update', 'sie_noreflection_iteration4update', 'eoh_1', 'eoh_2']
# algos = ['sie_full_iteration1update', 'sie_noreflection_iteration1update', 'sie_full_iteration4update', 'eoh', 'sie_noreflection_iteration4update']
algos = [algo+houzhui for algo in algos]
seed_path = 'single_mode'
# seed_path = 'mec_45S_6500'
whichone = '5'
directory_paths = [f'results/history/{seed_path}/{algo}/pops_best' for algo in algos]  # 替换为实际路径列表
o1, o2 = plot_objective_changes(directory_paths, algos, line_value=None, target_index=None, num_samples=7)
# o1, o2 = plot_objective_changes(directory_paths, algos, line_value=0.03357267092730118, target_index=0, num_samples=7)

# houzhui = '_2'
# algos = ['eoh', 'sie']
# algos = [algo+houzhui for algo in algos]
# directory_paths = [f'results/noseed/{seed_path}/{algo}/pops_best' for algo in algos]  # 替换为实际路径列表
# oo1, oo2 = plot_objective_changes(directory_paths, algos, line_value=None, target_index=None, num_samples=7)
# # oo1, oo2 = plot_objective_changes(directory_paths, algos, line_value=0.03357267092730118, target_index=0, num_samples=7)


def plot_mean_objective_changes(directory, names, times, line_value=None, target_index=None, num_samples=7):
    mean_changes = []
    for idx, dir in enumerate(names):
        mean_change = []
        for time in times:
            pop_best_root = os.path.join(directory, dir+'_'+str(time), 'pops_best')
            objectives = read_objectives(pop_best_root)

            if target_index is not None and idx == target_index:
                sampled_indices = np.linspace(0, len(objectives) - 1, num_samples, dtype=int)
                objectives = [objectives[i] for i in sampled_indices]

            mean_change.append(objectives)
        mean_changes.append(np.mean(np.array(mean_change), axis=0))
    print(mean_changes)

    for idx, y in enumerate(mean_changes):
        # plt.plot(y, label=names[idx])  # 修正了 'lable' 为 'label'
        plt.plot(y, marker='o', label=names[idx])

    if line_value is not None:
        plt.axhline(y=line_value, color='r', linestyle='--', label=f'Human designed Algorithm (GSPSO)')

    plt.legend()
    plt.show()


Times = [1, 2]
mean_dic = 'results/noseed/tsp100/'
# mean_dic = 'results/noseed/mec_45SMD/'
algos = ['eoh', 'sie']
# plot_mean_objective_changes(mean_dic, algos, Times, line_value=0.03357267092730118)
plot_mean_objective_changes(mean_dic, algos, Times, line_value=None)

