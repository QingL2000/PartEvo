import numpy as np
import copy


def algo(in_population, upper, lower, objfunction_input):
    """
    This algorithm is a hybrid optimization method that integrates Particle Swarm Optimization (PSO), Genetic Algorithm (GA), and Simulated Annealing (SA) strategies. It updates particle positions in each iteration by combining these three strategies to find the global optimum in complex search spaces.
    """
    Number_of_partical_gspso = np.shape(in_population)[0]
    K_gspso = np.shape(in_population)[1]
    Up_gspso = upper
    Down_gspso = lower
    objfunction = objfunction_input
    new = 0
    # GSPSO算法参数
    w_max = 0.95
    w_min = 0.4
    c1 = 2
    c2 = 2
    c = 1.5
    # SA算法的相关参数
    a = 0.95
    temperature = 10000000
    # GA 算法相关的参数
    pm = 0.02
    iterations = 1000

    # 建立本地最优position_local和全局最优position_global
    position_local = copy.deepcopy(in_population)
    position_global = np.zeros(K_gspso)

    position_global[:] = in_population[0, :]
    position_global_fitness = objfunction(position_global)

    population_fitnesses = np.zeros(Number_of_partical_gspso)
    # 初始化全局最优
    for i in range(Number_of_partical_gspso):
        population_fitnesses[i] = objfunction(in_population[i])
        if position_global_fitness > population_fitnesses[i]:
            position_global[:] = in_population[i, :]
            position_global_fitness = population_fitnesses[i]

    position_local_fitness = copy.deepcopy(population_fitnesses)

    # 建立样本E和后代O
    O = np.zeros((Number_of_partical_gspso, K_gspso))
    E = np.zeros((Number_of_partical_gspso, K_gspso))

    # 初始化样本E
    for i in range(Number_of_partical_gspso):
        for d in range(K_gspso):
            r1 = np.random.uniform()
            r2 = np.random.uniform()
            # E 是本地最优x↓和全局最优x↑的结合体
            E[i, d] = (c1 * r1 * position_local[i, d] + c2 * r2 * position_global[d]) / (
                    c1 * r1 + c2 * r2)  # xi'
    E_fitnesses = np.zeros(Number_of_partical_gspso)
    # 计算样本E的目标函数值
    for i in range(Number_of_partical_gspso):
        E_fitnesses[i] = objfunction(E[i, :])

    # 初始化速度 Velocity
    velocity = np.zeros((Number_of_partical_gspso, K_gspso))
    for i in range(K_gspso):
        velocity[:, i] = -Up_gspso[i] + 2 * Up_gspso[i] * np.random.rand(Number_of_partical_gspso, 1)[:, 0]

    # 初始的回合为1
    iterationIndex = 1
    w = w_max - (w_max - w_min) / iterations * iterationIndex  # *** line 15***
    # 创建各项指标的记录者
    fsList = np.zeros((iterations, 1))  # 存每个粒子的目标函数值、profit和penalty
    profitList = np.zeros((iterations, 1))
    penaltyList = np.zeros((iterations, 1))
    # 更新
    while iterationIndex <= iterations:
        # 对每个position_with_ofp[i]都更新

        for i in range(Number_of_partical_gspso):
            # 遗传算法GA
            # 交叉
            for d in range(K_gspso):
                k = np.random.randint(0, Number_of_partical_gspso - 1)
                if population_fitnesses[i] < population_fitnesses[k]:
                    rd = np.random.uniform()
                    O[i, d] = rd * position_local[i, d] + (1 - rd) * position_global[d]
                else:
                    O[i, d] = position_local[k, d]
            # 变异
            for d in range(K_gspso):
                if np.random.uniform() < pm:
                    O[i, d] = np.random.uniform(Down_gspso[d], Up_gspso[d])
            # 边界处理
            for d in range(K_gspso):
                if O[i, d] < Down_gspso[d]:
                    O[i, d] = Down_gspso[d]
                if O[i, d] > Up_gspso[d]:
                    O[i, d] = Up_gspso[d]
            # 选择
            fo = objfunction(O[i, :])
            fe = E_fitnesses[i]
            delta = fo - fe
            if delta < 0:
                E[i, :] = O[i, :]
                E_fitnesses[i] = fo
            else:
                # 模拟退货按照一定概率接受坏的结果
                probability = np.exp(-delta / temperature)
                randomNumber = np.random.uniform()
                if probability > randomNumber:
                    E[i, :] = O[i, :]
                    E_fitnesses[i] = fo
            # PSO算法
            velocity[i, :] = w * velocity[i, :] + c * np.random.uniform() * (
                    E[i, 0:K_gspso] - in_population[i, 0:K_gspso])
            # 每个粒子的最大速度不能超过边界
            for j in range(K_gspso):
                # ！！此处需要注意，该代码不适用于上限为0的情况！！
                if velocity[i, j] < -Up_gspso[j]:
                    velocity[i, j] = -Up_gspso[j]
                if velocity[i, j] > Up_gspso[j]:
                    velocity[i, j] = Up_gspso[j]
            # PSO更新
            in_population[i, 0:K_gspso] = in_population[i, 0:K_gspso] + velocity[i]
            # 边界处理
            for d in range(K_gspso):
                if in_population[i, d] < Down_gspso[d]:
                    in_population[i, d] = Down_gspso[d]
                if in_population[i, d] > Up_gspso[d]:
                    in_population[i, d] = Up_gspso[d]
            # 计算适应度
            population_fitnesses[i] = objfunction(in_population[i])
            # 判断是否更新全局最优粒子
            if population_fitnesses[i] < position_local_fitness[i]:
                position_local[i, :] = in_population[i, :]
                position_local_fitness[i] = population_fitnesses[i]
            if population_fitnesses[i] < position_global_fitness:
                position_global[:] = in_population[i, :]
                position_global_fitness = population_fitnesses[i]
                new = 1
        iterationIndex += 1
        w = w_max - (w_max - w_min) / iterations * iterationIndex  # w 线性下降
        temperature = temperature * a  # ***line 14 *** 降温
    return position_global
