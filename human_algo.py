import numpy as np
import time
import copy


def init_pop(lower, upper, popsize, dim):
    # 使用均匀分布随机初始化种群
    return np.random.uniform(lower, upper, (popsize, dim))

def GWO(pop, upper, lower, fitness_func, max_iter=100):
    popsize, dim = pop.shape
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    # 记录所有个体的适应度
    fitness_vals = np.zeros(popsize)

    # 开始迭代过程
    for t in range(max_iter):
        for i in range(popsize):
            fitness_vals[i] = fitness_func(pop[i])

            # 更新 alpha, beta, delta 狼
            if fitness_vals[i] < alpha_score:
                alpha_score = fitness_vals[i]
                alpha_pos = pop[i].copy()

            if alpha_score < fitness_vals[i] < beta_score:
                beta_score = fitness_vals[i]
                beta_pos = pop[i].copy()

            if alpha_score < fitness_vals[i] < delta_score:
                delta_score = fitness_vals[i]
                delta_pos = pop[i].copy()

        # 更新位置
        a = 2 - t * (2 / max_iter)  # 衰减因子

        for i in range(popsize):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A1 = 2 * a * r1 - a  # 计算A
            C1 = 2 * r2  # 计算C

            D_alpha = np.abs(C1 * alpha_pos - pop[i])  # 计算距离
            X1 = alpha_pos - A1 * D_alpha  # 计算新位置

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = np.abs(C2 * beta_pos - pop[i])
            X2 = beta_pos - A2 * D_beta

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = np.abs(C3 * delta_pos - pop[i])
            X3 = delta_pos - A3 * D_delta

            # 更新个体位置
            pop[i] = (X1 + X2 + X3) / 3

            # 确保位置在搜索空间内
            pop[i] = np.clip(pop[i], lower, upper)

    return alpha_pos

def GSPSO_generation_terminal(in_population, upper, lower, objfunction_input):
    Number_of_partical_gspso = np.shape(in_population)[0]
    K_gspso = np.shape(in_population)[1]
    Up_gspso = upper
    Down_gspso = lower
    objfunction = objfunction_input
    evatime=0
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
    position_global_fitness = objfunction_input(position_global)
    evatime += 1

    population_fitnesses = np.zeros(Number_of_partical_gspso)
    # 初始化全局最优
    for i in range(Number_of_partical_gspso):
        population_fitnesses[i] = objfunction_input(in_population[i])
        evatime += 1
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
        evatime += 1

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
            evatime += 1
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
            evatime += 1
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
    print('GSPSO evatimes:', evatime)
    return position_global

def LNS(in_population, upper, lower, objfunction_input, max_evaluations=30000):
    # 初始化
    Number_of_partical_lns = np.shape(in_population)[0]  # 粒子数量
    K_lns = np.shape(in_population)[1]  # 每个粒子的维度
    Up_lns = upper
    Down_lns = lower
    objfunction = objfunction_input

    # LNS算法参数
    initial_neighborhood_size = 10  # 初始邻域大小
    mutation_prob = 0.05  # 初始变异概率
    min_neighborhood_size = 2  # 最小邻域大小
    decay_rate = 0.99  # 邻域大小衰减率
    evaluations = 0  # 初始化评估次数
    no_improvement_threshold = 500  # 早停阈值：无改进的最大评估次数
    no_improvement_counter = 0

    # 初始化位置和适应度
    position_local = copy.deepcopy(in_population)
    position_global = np.zeros(K_lns)
    position_global[:] = in_population[0, :]
    position_global_fitness = objfunction_input(position_global)

    population_fitnesses = np.zeros(Number_of_partical_lns)
    for i in range(Number_of_partical_lns):
        population_fitnesses[i] = objfunction_input(in_population[i])
        evaluations += 1
        if position_global_fitness > population_fitnesses[i]:
            position_global[:] = in_population[i, :]
            position_global_fitness = population_fitnesses[i]

    # LNS的主要循环
    neighborhood_size = initial_neighborhood_size
    while evaluations < max_evaluations:
        improvement = False
        for i in range(Number_of_partical_lns):
            # 1. 生成邻域解（使用正态分布和动态邻域调整）
            neighbors = generate_neighbors(position_local[i], neighborhood_size, Up_lns, Down_lns)

            # 2. 评估邻域解（并行化评估）
            neighbor_fitnesses = np.array([objfunction(neighbor) for neighbor in neighbors])
            evaluations += len(neighbors)

            # 3. 选择最好的邻域解
            best_neighbor_index = np.argmin(neighbor_fitnesses)
            best_neighbor = neighbors[best_neighbor_index]
            best_neighbor_fitness = neighbor_fitnesses[best_neighbor_index]

            # 4. 变异操作（自适应变异概率）
            adaptive_mutation_prob = mutation_prob + (1 - mutation_prob) * (evaluations / max_evaluations)
            if np.random.uniform() < adaptive_mutation_prob:
                best_neighbor = mutate(best_neighbor, Up_lns, Down_lns)
                best_neighbor_fitness = objfunction(best_neighbor)
                evaluations += 1

            # 5. 更新粒子的局部和全局最优
            if best_neighbor_fitness < population_fitnesses[i]:
                position_local[i, :] = best_neighbor
                population_fitnesses[i] = best_neighbor_fitness
                improvement = True
            if best_neighbor_fitness < position_global_fitness:
                position_global[:] = best_neighbor
                position_global_fitness = best_neighbor_fitness
                improvement = True

        # 动态调整邻域大小（逐渐缩小）
        neighborhood_size = int(max(min_neighborhood_size, neighborhood_size * decay_rate))

        # 检查是否需要早停
        if not improvement:
            no_improvement_counter += 1
            if no_improvement_counter >= no_improvement_threshold:
                break
        else:
            no_improvement_counter = 0

    return position_global

def generate_neighbors(solution, neighborhood_size, upper, lower):
    """
    生成邻域解，通过正态分布生成解，并动态调整邻域大小。
    """
    neighbors = []
    for _ in range(neighborhood_size):
        neighbor = solution + np.random.normal(loc=0.0, scale=0.1, size=solution.shape)
        # 邻域解需要保持在边界范围内
        neighbor = np.clip(neighbor, lower, upper)
        neighbors.append(neighbor)
    return np.array(neighbors)

def mutate(solution, upper, lower):
    """
    变异操作：在解的多个维度上进行小范围扰动。
    """
    mutated_solution = solution.copy()
    num_mutations = np.random.randint(1, len(solution) // 2 + 1)  # 随机选择1到一半维度进行变异
    mutation_indices = np.random.choice(len(solution), num_mutations, replace=False)
    for idx in mutation_indices:
        mutated_solution[idx] = np.random.uniform(lower[idx], upper[idx])
    return mutated_solution

def SBAGO(position_run, up_position, down_position, objfunction_input, interations=1000):
    Number_of_partical = np.shape(position_run)[0]
    dimension = np.shape(position_run)[1]

    # SBAGO parameters
    w_max = 0.9
    w_min = 0.4
    c1 = 2.0
    c2 = 2.0
    c = 0.1
    pm = 0.05

    # Default parameters
    alpha = 0.9
    gamma = 0.9

    # This frequency range determines the scalings
    Frequency_min = 0  # Frequency minimum
    Frequency_max = 0.9  # Frequency maximum

    length = 20  # 时间段20
    sg = 7  # 停止间隔

    # 建立本地最优position_local和全局最优position_global
    position_local_i = copy.deepcopy(position_run)
    position_global = np.zeros(dimension)

    position_global[:] = position_run[0, :]
    position_global_fitness = objfunction_input(position_global)

    population_fitnesses = np.zeros(Number_of_partical)
    # 初始化全局最优
    for i in range(Number_of_partical):
        population_fitnesses[i] = objfunction_input(position_run[i])
        if position_global_fitness > population_fitnesses[i]:
            position_global[:] = position_run[i, :]
            position_global_fitness = population_fitnesses[i]

    position_local_i_fitness = copy.deepcopy(population_fitnesses)

    # Initializing arrays
    Frequency = np.zeros(Number_of_partical)  # Frequency
    Velocities = np.zeros((Number_of_partical, dimension))  # Velocities

    # Initialize pulse rates ri and the loudness Ai
    loudness = np.random.uniform(1, 2, Number_of_partical)
    pulse_rates = np.random.uniform(0, 1, Number_of_partical)

    # print('全局最优粒子的OFP为', position_global_fitness)
    # 建立样本E和后代O
    O = np.zeros((Number_of_partical, dimension))
    E = np.zeros((Number_of_partical, dimension))

    index = np.zeros(Number_of_partical)

    # 初始化样本E
    E_fitnesses = np.zeros(Number_of_partical)
    for i in range(Number_of_partical):
        for d in range(dimension):
            r1 = np.random.uniform()
            r2 = np.random.uniform()
            # E 是本地最优x↓和全局最优x↑的结合体
            E[i, d] = (c1 * r1 * position_local_i[i, d] + c2 * r2 * position_global[d]) / (
                    c1 * r1 + c2 * r2)  # xi'
        E_fitnesses[i] = objfunction_input(E[i, :])

    # 初始的回合为1
    iterationIndex = 1
    w = w_max - (w_max - w_min) / interations * iterationIndex  # *** line 15***
    # 创建各项指标的记录者
    fsList = np.zeros((interations, 1))  # 存每个粒子的目标函数值、profit和penalty
    timeList = np.zeros((interations, 1))

    # 更新
    while iterationIndex <= interations:
        # 对每个position_with_ofp[i]都更新
        begin_time = time.time()
        for i in range(Number_of_partical):
            # 遗传算法GA
            # 交叉
            for d in range(dimension):
                k = np.random.randint(0, Number_of_partical - 1)
                if population_fitnesses[i] < population_fitnesses[k]:
                    rd = np.random.uniform()
                    O[i, d] = rd * position_local_i[i, d] + (1 - rd) * position_global[d]
                else:
                    O[i, d] = position_local_i[k, d]
            # 变异
            for d in range(dimension):
                if np.random.uniform() < pm:
                    O[i, d] = np.random.uniform(down_position[d], up_position[d])

            # 边界处理
            for d in range(dimension):
                if O[i, d] < down_position[d]:
                    O[i, d] = down_position[d]
                if O[i, d] > up_position[d]:
                    O[i, d] = up_position[d]
            # 选择
            fo = objfunction_input(O[i])
            fe = E_fitnesses[i]
            index[i] += 1
            delta_sa = fo - fe
            if delta_sa < 0:
                E[i, :] = O[i, :]
                E_fitnesses[i] = fo
                index[i] = 0

            if index[i] > sg:
                # select E by 20%M tournament
                n = int(0.2 * Number_of_partical)
                pi = np.random.choice(Number_of_partical, size=n, replace=False)
                best_i = pi[0]
                best_value= objfunction_input(E[best_i])
                # get Ej
                for d in range(1, n):
                    fj = objfunction_input(E[pi[d]])
                    if fj < best_value:
                        best_value = fj
                        best_i = pi[d]
                E[i, :] = E[best_i, :]
                index[i] = 0

            # BA
            Frequency[i] = Frequency_min + (Frequency_min - Frequency_max) * np.random.rand()  # 更新频率公式 (2020/7/20)
            for d in range(dimension):
                Velocities[i, d] = w * Velocities[i, d] + c * np.random.rand() * (position_run[i, d] - E[i, d]) * \
                                   Frequency[i]

                new_Position = position_run[i, d] + Velocities[i, d]

                new_Position = np.maximum(new_Position, down_position[d])
                new_Position = np.minimum(new_Position, up_position[d])

                # Pulse rate
                if np.random.rand() > pulse_rates[i]:  # r_{i}********************************
                    # The factor 0.001 limits the step sizes of random walks
                    new_Position = position_global[d] + 0.001 * np.random.randn()

                # Generate a new solution by flying randomly********************************Caution, 0.001 could be adjusted for each problem
                new_Position += 0.001 * np.random.randn()

                new_Position = np.maximum(new_Position, down_position[d])
                new_Position = np.minimum(new_Position, up_position[d])
                position_run[i, d] = new_Position

            # Evaluate new solutions
            population_fitnesses[i] = objfunction_input(position_run[i])

            # Update if the solution improves, or not too loud
            if population_fitnesses[i] < position_local_i_fitness[i] and np.random.rand() < loudness[i]:  # A_{i}********************************
                position_local_i_fitness[i] = population_fitnesses[i]
                # Increase ri and reduce Ai********************************
                loudness[i] = alpha * loudness[i]
                pulse_rates[i] = pulse_rates[i] * (1 - np.exp(-gamma * iterationIndex))

            # 判断是否更新全局最优粒子
            if population_fitnesses[i] < position_local_i[i, -3]:
                position_local_i[i, :] = position_run[i, :]
            if population_fitnesses[i] < position_global_fitness:
                position_global[:] = position_run[i, :]
                position_global_fitness = population_fitnesses[i]
                # print(position_global_fitness)

        iterationIndex += 1
        w = w_max - (w_max - w_min) / interations * iterationIndex  # w 线性下降
        stop_time = time.time()
        # print('iteration', iterationIndex, 'time cost is :', stop_time - begin_time)
        timeList[iterationIndex - 2, 0] = stop_time - begin_time
    # print('程序运行完毕')
    return position_global



def GSPSO(in_population, upper, lower, objfunction_input, max_evaluations=30000):
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
    # 初始化评估次数
    evaluations = 0

    # 建立本地最优position_local和全局最优position_global
    position_local = copy.deepcopy(in_population)
    position_global = np.zeros(K_gspso)
    position_global[:] = in_population[0, :]
    position_global_fitness = objfunction_input(position_global)
    evaluations += 1  # 记录评估次数

    population_fitnesses = np.zeros(Number_of_partical_gspso)
    # 初始化全局最优
    for i in range(Number_of_partical_gspso):
        population_fitnesses[i] = objfunction_input(in_population[i])
        evaluations += 1  # 记录评估次数
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
        evaluations += 1  # 记录评估次数

    # 初始化速度 Velocity
    velocity = np.zeros((Number_of_partical_gspso, K_gspso))
    for i in range(K_gspso):
        velocity[:, i] = -Up_gspso[i] + 2 * Up_gspso[i] * np.random.rand(Number_of_partical_gspso, 1)[:, 0]

    # 初始的回合为1
    iterationIndex = 1
    w = w_max - (w_max - w_min) / max_evaluations * evaluations  # *** 更新 w ***
    # 创建各项指标的记录者
    fsList = np.zeros((max_evaluations, 1))  # 存每个粒子的目标函数值、profit和penalty
    profitList = np.zeros((max_evaluations, 1))
    penaltyList = np.zeros((max_evaluations, 1))

    generation = 0
    # 更新
    while evaluations < max_evaluations:
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
            evaluations += 1  # 记录评估次数
            fe = E_fitnesses[i]
            delta = fo - fe
            if delta < 0:
                E[i, :] = O[i, :]
                E_fitnesses[i] = fo
            else:
                # 模拟退火按照一定概率接受坏的结果
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
            evaluations += 1  # 记录评估次数
            # 判断是否更新全局最优粒子
            if population_fitnesses[i] < position_local_fitness[i]:
                position_local[i, :] = in_population[i, :]
                position_local_fitness[i] = population_fitnesses[i]
            if population_fitnesses[i] < position_global_fitness:
                position_global[:] = in_population[i, :]
                position_global_fitness = population_fitnesses[i]
                new = 1
        # 更新参数
        w = w_max - (w_max - w_min) / max_evaluations * evaluations  # w 线性下降
        temperature = temperature * a  # *** 降温
        generation += 1
    print('GSPSO generation:', generation)
    print('Final evaluations:', evaluations)
    return position_global

def PSO(in_population, upper, lower, objfunction_input, max_evaluations=30000):
    # 初始化
    Number_of_particles = np.shape(in_population)[0]  # 粒子数量
    K = np.shape(in_population)[1]  # 维度
    Up = upper  # 上界
    Down = lower  # 下界
    objfunction = objfunction_input
    evaluations = 0  # 评估次数

    # PSO 参数
    w_max = 0.95  # 惯性权重最大值
    w_min = 0.4   # 惯性权重最小值
    c1 = 2        # 个体学习因子
    c2 = 2        # 群体学习因子

    # 初始化粒子的位置和速度
    position = copy.deepcopy(in_population)
    velocity = np.zeros((Number_of_particles, K))
    for i in range(K):
        velocity[:, i] = -Up[i] + 2 * Up[i] * np.random.rand(Number_of_particles)

    # 初始化个体最优和全局最优
    position_local = copy.deepcopy(position)  # 个体最优位置
    position_local_fitness = np.zeros(Number_of_particles)  # 个体最优适应值
    position_global = np.zeros(K)  # 全局最优位置
    position_global[:] = position[0, :]  # 初始化为第一个粒子的位置
    position_global_fitness = objfunction(position_global)  # 全局最优适应值
    evaluations += 1

    # 计算初始适应值
    for i in range(Number_of_particles):
        position_local_fitness[i] = objfunction(position[i])
        evaluations += 1
        if position_local_fitness[i] < position_global_fitness:  # 更新全局最优
            position_global[:] = position[i, :]
            position_global_fitness = position_local_fitness[i]

    # 迭代主循环
    generation = 0
    while evaluations < max_evaluations:
        w = w_max - (w_max - w_min) / max_evaluations * evaluations  # 惯性权重线性下降

        for i in range(Number_of_particles):
            # 更新速度
            r1 = np.random.uniform(size=K)
            r2 = np.random.uniform(size=K)
            velocity[i, :] = (
                w * velocity[i, :] +
                c1 * r1 * (position_local[i, :] - position[i, :]) +
                c2 * r2 * (position_global - position[i, :])
            )

            # 限制速度在边界范围内
            for d in range(K):
                if velocity[i, d] < -Up[d]:
                    velocity[i, d] = -Up[d]
                if velocity[i, d] > Up[d]:
                    velocity[i, d] = Up[d]

            # 更新位置
            position[i, :] = position[i, :] + velocity[i, :]

            # 边界处理
            for d in range(K):
                if position[i, d] < Down[d]:
                    position[i, d] = Down[d]
                if position[i, d] > Up[d]:
                    position[i, d] = Up[d]

            # 计算适应值
            fitness = objfunction(position[i])
            evaluations += 1

            # 更新个体最优
            if fitness < position_local_fitness[i]:
                position_local[i, :] = position[i, :]
                position_local_fitness[i] = fitness

            # 更新全局最优
            if fitness < position_global_fitness:
                position_global[:] = position[i, :]
                position_global_fitness = fitness

        generation += 1

    print('PSO generation:', generation)
    print('Final evaluations:', evaluations)
    return position_global

def DE(in_population, upper, lower, objfunction_input, F=0.8, CR=0.9, max_evaluations=30000):
    """
    差分进化算法（Differential Evolution）

    Parameters:
    - in_population: 初始种群
    - upper: 每个解的上界
    - lower: 每个解的下界
    - objfunction_input: 目标函数
    - F: 变异因子（mutation factor），默认0.8
    - CR: 交叉概率（crossover probability），默认0.9
    - max_evaluations: 最大目标函数评估次数，默认2000

    Returns:
    - position_global: 最优解
    """
    # 获取种群和解的维度
    Number_of_partical_de = np.shape(in_population)[0]  # 粒子数
    K_de = np.shape(in_population)[1]  # 每个粒子的维度
    Up_de = upper  # 解空间上界
    Down_de = lower  # 解空间下界

    # 初始化适应度
    population_fitnesses = np.zeros(Number_of_partical_de)
    evaluations = 0  # 目标函数评估次数
    for i in range(Number_of_partical_de):
        population_fitnesses[i] = objfunction_input(in_population[i])
        evaluations += 1  # 记录评估次数

    # 初始化全局最优解
    position_global = in_population[0, :]
    position_global_fitness = population_fitnesses[0]

    for i in range(Number_of_partical_de):
        if population_fitnesses[i] < position_global_fitness:
            position_global[:] = in_population[i, :]
            position_global_fitness = population_fitnesses[i]

    # 迭代过程
    generation = 0
    while evaluations < max_evaluations:
        for i in range(Number_of_partical_de):
            # 选择三个不同的个体进行差分变异
            candidates = [idx for idx in range(Number_of_partical_de) if idx != i]
            a, b, c = in_population[np.random.choice(candidates, 3, replace=False)]

            # 差分变异
            mutant = a + F * (b - c)

            # 交叉操作
            cross_points = np.random.rand(K_de) < CR
            trial = np.copy(in_population[i])
            trial[cross_points] = mutant[cross_points]

            # 边界处理
            trial = np.clip(trial, Down_de, Up_de)

            # 目标函数评估
            trial_fitness = objfunction_input(trial)
            evaluations += 1  # 记录评估次数

            # 选择操作
            if trial_fitness < population_fitnesses[i]:
                in_population[i, :] = trial
                population_fitnesses[i] = trial_fitness

                # 更新全局最优解
                if trial_fitness < position_global_fitness:
                    position_global[:] = trial
                    position_global_fitness = trial_fitness

        generation += 1
    print('DE generation:', generation)
    print('Final evaluations:', evaluations)
    return position_global

def DE_optimized(in_population, upper, lower, objfunction_input, F=0.8, CR=0.9, max_evaluations=30000, tol=1e-6, patience=500):
    """
    优化后的差分进化算法（Differential Evolution）
    自适应变异因子和交叉概率：动态调整变异因子 F 和交叉概率 CR，提高算法的搜索效率。
    优秀个体参与变异：引入当前全局最优解参与变异，提升算法的收敛性。
    种群多样性维护：在种群多样性不足时引入随机扰动，避免陷入局部最优。
    早停机制：当算法在一段时间内没有显著改进时，提前终止，提高效率。
    并行计算：对目标函数的评估进行并行化（如果目标函数计算代价较高）。
    改进边界处理：使用反射边界处理，避免解被限制在边界附近。

    Parameters:
    - in_population: 初始种群
    - upper: 每个解的上界
    - lower: 每个解的下界
    - objfunction_input: 目标函数
    - F: 初始变异因子（mutation factor），默认0.8
    - CR: 初始交叉概率（crossover probability），默认0.9
    - max_evaluations: 最大目标函数评估次数，默认30000
    - tol: 收敛容忍度，用于早停机制，默认1e-6
    - patience: 连续多少代没有显著改进则早停，默认100

    Returns:
    - position_global: 最优解
    - position_global_fitness: 最优解的适应度值
    """
    # 获取种群和解的维度
    Number_of_partical_de = np.shape(in_population)[0]  # 粒子数
    K_de = np.shape(in_population)[1]  # 每个粒子的维度
    Up_de = upper  # 解空间上界
    Down_de = lower  # 解空间下界

    # 初始化适应度
    population_fitnesses = np.zeros(Number_of_partical_de)
    evaluations = 0  # 目标函数评估次数
    for i in range(Number_of_partical_de):
        population_fitnesses[i] = objfunction_input(in_population[i])
        evaluations += 1  # 记录评估次数

    # 初始化全局最优解
    position_global = in_population[0, :]
    position_global_fitness = population_fitnesses[0]

    for i in range(Number_of_partical_de):
        if population_fitnesses[i] < position_global_fitness:
            position_global[:] = in_population[i, :]
            position_global_fitness = population_fitnesses[i]

    # 记录最近几代的最优值变化，用于早停
    best_fitness_history = [position_global_fitness]
    no_improvement_count = 0

    # 迭代过程
    generation = 0
    while evaluations < max_evaluations:
        for i in range(Number_of_partical_de):
            # 动态调整变异因子和交叉概率
            F_dynamic = 0.5 + 0.3 * np.random.rand()  # 动态变异因子
            CR_dynamic = 0.5 + 0.4 * np.random.rand()  # 动态交叉概率

            # 选择三个不同的个体进行差分变异
            candidates = [idx for idx in range(Number_of_partical_de) if idx != i]
            a, b, c = in_population[np.random.choice(candidates, 3, replace=False)]

            # 引入全局最优解参与变异
            mutant = a + F_dynamic * (b - c) + F_dynamic * (position_global - a)

            # 边界处理（反射边界）
            mutant = np.where(mutant > Up_de, Up_de - (mutant - Up_de), mutant)
            mutant = np.where(mutant < Down_de, Down_de + (Down_de - mutant), mutant)

            # 交叉操作
            cross_points = np.random.rand(K_de) < CR_dynamic
            trial = np.copy(in_population[i])
            trial[cross_points] = mutant[cross_points]

            # 边界处理（再一次确保边界有效性）
            trial = np.clip(trial, Down_de, Up_de)

            # 目标函数评估
            trial_fitness = objfunction_input(trial)
            evaluations += 1  # 记录评估次数

            # 选择操作
            if trial_fitness < population_fitnesses[i]:
                in_population[i, :] = trial
                population_fitnesses[i] = trial_fitness

                # 更新全局最优解
                if trial_fitness < position_global_fitness:
                    position_global[:] = trial
                    position_global_fitness = trial_fitness

        # 记录当前代的最优值
        best_fitness_history.append(position_global_fitness)

        # # 检查早停条件
        # if len(best_fitness_history) > patience:
        #     recent_improvement = np.abs(best_fitness_history[-patience] - position_global_fitness)
        #     if recent_improvement < tol:
        #         print(f"Early stopping at generation {generation} with fitness {position_global_fitness}")
        #         break

        generation += 1

    print('DE generation:', generation)
    print('Final evaluations:', evaluations)
    return position_global

def GA(in_population, upper, lower, objfunction_input, crossover_rate=0.9, mutation_rate=0.1, max_evaluations=30000):
    """
    最常用的遗传算法（Genetic Algorithm, GA）
    选择操作：使用轮盘赌选择（Roulette Wheel Selection）
    交叉操作：实现单点交叉（Single-Point Crossover）
    变异操作：实现均匀变异（Uniform Mutation）

    Parameters:
    - in_population: 初始种群
    - upper: 每个解的上界
    - lower: 每个解的下界
    - objfunction_input: 目标函数
    - crossover_rate: 交叉概率，默认0.9
    - mutation_rate: 变异概率，默认0.1
    - max_evaluations: 最大评估次数，默认100000

    Returns:
    - best_solution: 最优解
    - best_fitness: 最优解的适应度值
    """
    # 获取种群和解的维度
    population_size, dim = in_population.shape
    upper = np.array(upper)
    lower = np.array(lower)

    # 初始化适应度
    fitnesses = np.array([objfunction_input(ind) for ind in in_population])
    evaluations = population_size  # 初始评估次数是种群大小

    # 初始化全局最优解
    best_idx = np.argmin(fitnesses)
    best_solution = in_population[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    generation = 0
    while evaluations < max_evaluations:
        # 选择操作：轮盘赌选择
        total_fitness = np.sum(fitnesses)
        selection_probs = (1 / (fitnesses + 1e-6)) / total_fitness
        selection_probs /= np.sum(selection_probs)
        selected_population = in_population[np.random.choice(population_size, size=population_size, p=selection_probs)]

        # 交叉操作：单点交叉
        offspring_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            if i + 1 < population_size:
                parent2 = selected_population[i + 1]
            else:
                parent2 = selected_population[np.random.randint(population_size)]

            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, dim)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                offspring_population.append(np.clip(child1, lower, upper))
                offspring_population.append(np.clip(child2, lower, upper))
            else:
                offspring_population.append(parent1)
                offspring_population.append(parent2)

        offspring_population = np.array(offspring_population[:population_size])

        # 变异操作：均匀变异
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_vector = np.random.uniform(-0.1, 0.1, dim)
                offspring_population[i] += mutation_vector
                offspring_population[i] = np.clip(offspring_population[i], lower, upper)

        # 计算新种群的适应度
        offspring_fitnesses = np.array([objfunction_input(ind) for ind in offspring_population])
        evaluations += population_size  # 每次计算种群适应度时，增加评估次数

        # 更新种群
        in_population = offspring_population
        fitnesses = offspring_fitnesses

        # 更新全局最优解
        current_best_idx = np.argmin(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        if current_best_fitness < best_fitness:
            best_solution = in_population[current_best_idx].copy()
            best_fitness = current_best_fitness
        generation += 1
    print('GA generation:', generation)
    print('Final evaluations:', evaluations)
    return best_solution

def GA_optimized(in_population, upper, lower, objfunction_input, crossover_rate=0.9, mutation_rate=0.1,
                 max_evaluations=30000, tol=1e-6, patience=500, tournament_size=3):
    """
    优化后的遗传算法（Genetic Algorithm, GA）
    选择操作：使用轮盘赌选择（Roulette Wheel Selection）和锦标赛选择（Tournament Selection）两种方式。
    交叉操作：实现单点交叉（Single-Point Crossover）和模拟二进制交叉（Simulated Binary Crossover, SBX）。
    变异操作：实现多种变异方法，包括高斯变异（Gaussian Mutation）和均匀变异（Uniform Mutation）。
    适应度归一化：防止适应度值过大或过小对选择操作的影响。
    精英保留策略：保留当前种群中适应度最优的个体，防止最优解丢失。
    早停机制：当适应度值在一定代数内没有显著改进时提前终止。

    Parameters:
    - in_population: 初始种群
    - upper: 每个解的上界
    - lower: 每个解的下界
    - objfunction_input: 目标函数
    - crossover_rate: 交叉概率，默认0.9
    - mutation_rate: 变异概率，默认0.1
    - max_evaluations: 最大目标函数评估次数，默认30000
    - tol: 收敛容忍度，用于早停机制，默认1e-6
    - patience: 连续多少代没有显著改进则早停，默认100
    - tournament_size: 锦标赛选择的参与者数量，默认3

    Returns:
    - best_solution: 最优解
    - best_fitness: 最优解的适应度值
    """
    # 获取种群和解的维度
    population_size, dim = in_population.shape
    upper = np.array(upper)
    lower = np.array(lower)

    # 初始化适应度
    fitnesses = np.array([objfunction_input(ind) for ind in in_population])
    evaluations = len(fitnesses)

    # 初始化全局最优解
    best_idx = np.argmin(fitnesses)
    best_solution = in_population[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    # 记录最近几代的最优值变化，用于早停
    best_fitness_history = [best_fitness]
    no_improvement_count = 0

    generation = 0
    while evaluations < max_evaluations:
        # 选择操作（锦标赛选择）
        selected_population = []
        for _ in range(population_size):
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitnesses = fitnesses[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
            selected_population.append(in_population[winner_idx])
        selected_population = np.array(selected_population)

        # 交叉操作（模拟二进制交叉，SBX）
        offspring_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            if i + 1 < population_size:
                parent2 = selected_population[i + 1]
            else:
                parent2 = selected_population[np.random.randint(population_size)]

            if np.random.rand() < crossover_rate:
                # Simulated Binary Crossover (SBX)
                eta = 2  # Crossover distribution index
                beta = np.zeros(dim)
                for j in range(dim):
                    u = np.random.rand()
                    if u <= 0.5:
                        beta[j] = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta[j] = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
                offspring_population.append(np.clip(child1, lower, upper))
                offspring_population.append(np.clip(child2, lower, upper))
            else:
                offspring_population.append(parent1)
                offspring_population.append(parent2)
        offspring_population = np.array(offspring_population[:population_size])

        # 变异操作（高斯变异）
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_vector = np.random.normal(0, 0.1, dim)  # 高斯噪声
                offspring_population[i] += mutation_vector
                offspring_population[i] = np.clip(offspring_population[i], lower, upper)

        # 计算新种群的适应度
        offspring_fitnesses = np.array([objfunction_input(ind) for ind in offspring_population])
        evaluations += len(offspring_fitnesses)

        # 精英保留策略（保留上一代的最优解）
        worst_idx = np.argmax(offspring_fitnesses)
        if best_fitness < offspring_fitnesses[worst_idx]:
            offspring_population[worst_idx] = best_solution
            offspring_fitnesses[worst_idx] = best_fitness

        # 更新种群
        in_population = offspring_population
        fitnesses = offspring_fitnesses

        # 更新全局最优解
        current_best_idx = np.argmin(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        if current_best_fitness < best_fitness:
            best_solution = in_population[current_best_idx].copy()
            best_fitness = current_best_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # 记录当前代的最优值
        best_fitness_history.append(best_fitness)

        # # 检查早停条件
        # if no_improvement_count >= patience:
        #     print(f"Early stopping at generation {generation} with fitness {best_fitness}")
        #     break

        generation += 1

    print('GA generation:', generation)
    print('Final evaluations:', evaluations)
    return best_solution

def tamp(initial_population: np.ndarray, individual_upper: np.ndarray, individual_lower: np.ndarray,
         objective_function: callable) -> np.ndarray:
    """
    An algorithm for solving a minimization optimization problem. It allows up to 30,000 calls to the objective_function to evaluate solutions.

    Args:
    initial_population: np.ndarray
        A 2D array where each row represents an individual solution in the initial population.
    individual_upper: np.ndarray
        A 1D array representing the upper bounds for each dimension of the solution.
    individual_lower: np.ndarray
        A 1D array representing the lower bounds for each dimension of the solution.
    objective_function: callable
        A Python function used to compute the objective value of candidate solutions. Lower objective values indicate better solutions.

    Returns:
    np.ndarray
        The best solution found by the algorithm after 30,000 evaluations.
    """
    max_evaluations = 30000
    evaluations = 0
    population_size, dimensions = initial_population.shape
    current_population = initial_population.copy()
    best_solution = current_population[np.argmin([objective_function(ind) for ind in current_population])]

    # Evaluate initial population
    fitness = np.array([objective_function(ind) for ind in current_population])
    evaluations += population_size

    while evaluations < max_evaluations:
        for i in range(population_size):
            if evaluations >= max_evaluations:
                break

            # Mutation and Crossover
            indices = np.arange(population_size)
            np.random.shuffle(indices)
            a, b, c = indices[:3]
            mutant_solution = current_population[a] + np.random.uniform(-0.5, 0.5) * (
                        current_population[b] - current_population[c])
            mutant_solution = np.clip(mutant_solution, individual_lower, individual_upper)

            # Crossover
            trial_solution = np.where(np.random.rand(dimensions) < 0.5, mutant_solution, current_population[i])
            trial_solution = np.clip(trial_solution, individual_lower, individual_upper)
            new_value = objective_function(trial_solution)
            evaluations += 1

            # Selection
            if new_value < fitness[i]:
                current_population[i] = trial_solution
                fitness[i] = new_value
                if new_value < objective_function(best_solution):
                    best_solution = trial_solution

    return best_solution

def f1(x):
    return np.sum(x**2)


def terrable_2792(initial_population, individual_upper, individual_lower, objective_function):
    population = np.copy(initial_population)
    population_size, dimensions = population.shape
    eval_count = 0
    best_solution = None
    best_fitness = float('inf')
    max_evaluations = 30000
    stagnation_count = 0
    max_stagnation = 150
    elite_count = max(1, population_size // 10)
    initial_temp = 1.0
    cooling_rate = 0.95
    diversity_threshold = 0.1
    elite_memory = []

    def latin_hypercube_sampling(size, dimensions, lower, upper):
        intervals = np.linspace(lower, upper, size + 1)
        points = np.random.rand(size, dimensions)
        sample = intervals[:-1] + (intervals[1:] - intervals[:-1]) * points
        return np.clip(sample, lower, upper)

    population = latin_hypercube_sampling(population_size, dimensions, individual_lower, individual_upper)
    fitness_values = np.array([objective_function(ind) for ind in population])
    eval_count += population_size
    best_solution = population[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)
    elite_memory.append(best_solution)

    while eval_count < max_evaluations:
        selected_indices = np.argsort(fitness_values)[:elite_count]
        parents = population[selected_indices]
        offspring = []
        temp = initial_temp * (cooling_rate ** (eval_count // population_size))
        diversity_metric = np.std(fitness_values)

        for _ in range(population_size // 2):
            parent1, parent2 = parents[np.random.choice(parents.shape[0], 2, replace=False)]
            crossover_point = np.random.randint(1, dimensions)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            for child in [child1, child2]:
                if np.random.rand() < 0.5:
                    mutation_idx = np.random.randint(0, dimensions)
                    child[mutation_idx] = np.clip(
                        np.random.uniform(individual_lower[mutation_idx], individual_upper[mutation_idx]) * temp,
                        individual_lower[mutation_idx], individual_upper[mutation_idx])
                offspring.append(np.clip(child, individual_lower, individual_upper))

        offspring = np.array(offspring)
        local_search_steps = 10
        for ind in offspring:
            for _ in range(local_search_steps):
                perturb_idx = np.random.randint(dimensions)
                trial_solution = np.copy(ind)
                trial_solution[perturb_idx] = np.clip(
                    np.random.uniform(individual_lower[perturb_idx], individual_upper[perturb_idx]),
                    individual_lower[perturb_idx], individual_upper[perturb_idx])
                if objective_function(trial_solution) < objective_function(ind):
                    ind[:] = trial_solution

        fitness_values = np.append(fitness_values, [objective_function(ind) for ind in offspring])
        eval_count += offspring.shape[0]

        combined_population = np.vstack((population, offspring))
        combined_fitness = fitness_values.argsort()[:population_size]
        population = combined_population[combined_fitness]
        fitness_values = fitness_values[combined_fitness]

        current_best_fitness = np.min(fitness_values)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmin(fitness_values)]
            elite_memory.append(best_solution)
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= max_stagnation:
            perturb_idx = np.random.randint(0, population_size)
            if diversity_metric < diversity_threshold:
                population[perturb_idx] = np.clip(
                    np.random.rand(dimensions) * (individual_upper - individual_lower) + individual_lower,
                    individual_lower, individual_upper)
            if elite_memory:
                population[perturb_idx] = np.copy(elite_memory[np.random.choice(len(elite_memory))])
            stagnation_count = 0

    return best_solution


if __name__ == '__main__':
    from src.HIE.problems.optimization.mec_task_offloading_blackbox import MECENV, mec_instance
    from src.HIE.problems.optimization.single_mode_blackbox import Baseline
    from src.HIE.problems.optimization.machine_level_scheduling import MLSENV, Environment
    from src.HIE.problems.optimization.multi_mode_blackbox import Baseline_multi
    import time

    dim = 30
    popsize = 20
    upper = np.full(dim, 30)
    lower = np.full(dim, -30)

    inited = init_pop(lower, upper, popsize, dim)
    # result = SBAGO(inited, upper, lower, f1)
    # print(f1(result))
    #
    # # result2 = GWO(inited, upper, lower, f1, 1000)
    # print(f1(result2))

    s1 = time.time()
    result4 = terrable_2792(copy.deepcopy(inited), upper, lower, f1)
    print(f1(result4))
    print('time:', time.time() - s1)
    print()

    # s1 = time.time()
    # result4 = DE_optimized(copy.deepcopy(inited), upper, lower, f1)
    # print(f1(result4))
    # print('time:', time.time() - s1)
    # print()
    #
    # s1 = time.time()
    # result4 = GA_optimized(copy.deepcopy(inited), upper, lower, f1)
    # print(f1(result4))
    # print('time:', time.time()-s1)
    # print()
    #
    # s2 = time.time()
    # result5 = DE(copy.deepcopy(inited), upper, lower, f1)
    # print(f1(result5))
    # print('time:', time.time() - s2)
    # print()
    #
    # # s4 = time.time()
    # # result3 = GSPSO_generation_terminal(copy.deepcopy(inited), upper, lower, f1)
    # # print(f1(result3))
    # # print('time:', time.time() - s4)
    # # print()
    #
    # s3 = time.time()
    # result6 = GSPSO(copy.deepcopy(inited), upper, lower, f1)
    # print(f1(result6))
    # print('time:', time.time() - s3)
    # print()

