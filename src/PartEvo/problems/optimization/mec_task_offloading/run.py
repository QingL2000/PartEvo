import numpy as np
import pickle
import sys
import types
import warnings
import copy
from joblib import Parallel, delayed

from .prompts import GetPrompts


class MECENV():
    def __init__(self) -> None:
        self.prompts = GetPrompts()
        self.can_visualize = False
        # 固定参数
        self.K = 45
        self.J = 5
        self.ik_set = 6500
        # self.pUEk = 9.1 / 3600000  # 终端单位电价 yuan/J
        # self.pENj = 4.5 / 3600000  # 边缘服务器单位电价 yuan/J
        # self.pUEk = 18.2 / 3600000 * 100  # 终端单位电价 yuan/J
        # self.pENj = 9 / 3600000 * 100  # 边缘服务器单位电价 yuan/J
        self.pUEk = 18.2 / 3600000 * 1000000  # 终端单位电价 yuan/J
        self.pENj = 9 / 3600000 * 1000000  # 边缘服务器单位电价 yuan/J
        self.qLk = 10 ** -26  # 终端芯片架构系数
        self.qENj = 10 ** -27  # 边缘服务器架构系数
        self.flk = 4 * 10 ** 8  # 终端计算速度 cycles/s
        self.flk_max = 4 * 10 ** 8  # 终端最大计算速度
        self.fENj = 8 * 10 ** 8  # 边缘服务器计算速度 cycles/s
        self.fCk = 3.2 * 10 ** 9  # 云服务器计算速度 cycles/s
        self.rc = 1.142 * 10 ** -9  # 云服务器中每个CPU周期的价格
        self.phi1j_max = 8 * 10 ** 9  # 边缘服务器的最大CPU数
        self.phi2j_max = 2048 * 1024 * 1024  # 边缘服务器的最大内存数 2 GB
        self.wk = 8  # 终端执行每一个bit所需要的内存  byte/bit
        self.W = 10000000  # 终端与边缘服务器的信道带宽 10Mhz
        self.Ptm = 0.1  # 终端的传输功率 0.1W
        self.v = 4  # 路径损耗的指数
        self.h1 = 0.98  # 一个圆对称复高斯随机变量
        self.N0 = 1.6 * 10 ** -11  # 白高斯噪声的功率
        self.Mj = 10 ** 8  # bit/s  100Mbps 边缘服务器与云之间有线回程链路的传输速率 12.5 MB/s
        self.d_allow = 800  # 可以连接的距离
        self.T_k_max = 2.5  # 最大时间延迟
        self.alpha_max = 1
        self.alpha_min = 0.01
        self.beta_max = 1
        self.gamma_max = 1
        self.safe_number = 88
        self.zk = 100  # 终端UE执行每一个bit需要的CPU周期数 cpu cycles/bit
        self.Ek_max = 6
        self.EEN_max = 20
        self.Sj = 10  # 每个EN的通道数
        self.dist = np.random.randint(1, 1000, (self.K, self.J)).astype(np.float32)  # 存放每个UEk和ENj之间的距离
        self.index_allow = np.zeros((self.K, self.J))  # 标记哪些UE可以和哪些EN连接
        self.d_allow_to_connect = {}  # 记录可连接的EN的距离
        self.allow_No = {}  # 记录某个UEk可以和哪些EN连接
        # self.mean_door = 0.1
        self.mean_door = -1
        self.penalty_param = 100
        self.Ik = np.full((self.K, 1), self.ik_set)
        for k in range(self.K):
            self.allow_No.update({'UE' + str(k): []})
            self.d_allow_to_connect.update({'UE' + str(k): []})

        for k in range(self.K):
            for j in range(self.J):
                if self.dist[k, j] <= self.d_allow:
                    self.allow_No['UE' + str(k)].append(j)
                    self.d_allow_to_connect['UE' + str(k)].append(self.dist[k, j])

        self.partical_nums = 20
        self.dimension_num = 3 * self.K

        self.upper = np.zeros(self.dimension_num)
        self.lower = np.zeros(self.dimension_num)

        for k_ind in range(self.K):
            self.upper[k_ind] = 1
            self.lower[k_ind] = 0
        for k_ind in range(self.K, 2 * self.K):
            self.upper[k_ind] = 1
            self.lower[k_ind] = 0
        for k_ind in range(2 * self.K, 3 * self.K):
            self.upper[k_ind] = self.J * 0.999
            self.lower[k_ind] = 0

        self.inited_positions = self.init_positions()

        def compute_human_grade():
            return self.objfunction(
                self.human_design_algo(copy.deepcopy(self.inited_positions), self.upper, self.lower, self.objfunction))

        # 并行运行5次
        grades = Parallel(n_jobs=5)(delayed(compute_human_grade)() for _ in range(5))

        # 计算最小值和平均值
        min_grade = min(grades)
        average_grade = sum(grades) / len(grades)

        print('#### Human grade are ', grades)
        print('#### Minimum human grade is ', min_grade)
        print('#### Average human grade is ', average_grade)

    def init_positions(self):
        """
        每一个position如下：
        [alpha_1, alpha_2, alpha_3, ..., alpha_k,
        b_1, b_2, b_3, ..., b_k,
        x_1, x_2, x_3, ..., x_k]

        alhpa是0-1的float
        b是0-1的float
        x是0-J的float
        """
        # positions = np.zeros((self.partical_nums, self.dimension_num + 1))
        # for ind in range(self.partical_nums):
        #     for d in range(self.dimension_num):
        #         positions[ind, d] = np.random.uniform(self.lower[d], self.upper[d])
        #     objvalue = self.objfunction(positions[ind, :self.dimension_num])
        #     positions[ind, self.dimension_num] = objvalue

        positions = np.zeros((self.partical_nums, self.dimension_num))
        for ind in range(self.partical_nums):
            for d in range(self.dimension_num):
                positions[ind, d] = np.random.uniform(self.lower[d], self.upper[d])
        return positions

    def human_design_algo(self, in_population, upper, lower, objfunction_input):
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
        position_global_fitness = self.objfunction(position_global)


        population_fitnesses = np.zeros(Number_of_partical_gspso)
        # 初始化全局最优
        for i in range(Number_of_partical_gspso):
            population_fitnesses[i] = self.objfunction(in_population[i])
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

    def transform_position_to_matrix(self, position):
        alpha_tran = np.zeros((self.K, 1))
        a_tran = np.zeros((self.K, 1))
        x_tran = {}
        for k in range(self.K):
            alpha_tran[k, 0] = position[k] / 1
            a_tran[k, 0] = position[k + self.K] / 1
            x_tran.update({'UE' + str(k): [int(position[k + 2 * self.K])]})
        beta_tran = (1 - alpha_tran) * a_tran
        gamma_tran = 1 - alpha_tran - beta_tran

        alpha_matrix = np.zeros((self.K, 1))
        beta_matrix = np.zeros((self.K, self.J))
        gamma_matrix = np.zeros((self.K, self.J))
        x_k_j_matrix = np.zeros((self.K, self.J))
        for k in range(self.K):
            y_index = int(x_tran['UE' + str(k)][0])
            if y_index != self.safe_number:
                alpha_matrix[k, 0] = alpha_tran[k, 0]
                beta_matrix[k, y_index] = beta_tran[k, 0]
                gamma_matrix[k, y_index] = gamma_tran[k, 0]
                x_k_j_matrix[k, y_index] = 1
            else:
                alpha_matrix[k, 0] = 1
                beta_matrix[k, y_index] = 0
                gamma_matrix[k, y_index] = 0
        return alpha_matrix, beta_matrix, gamma_matrix, x_k_j_matrix

    def cal_t_k(self, alpha_matrix, beta_matrix, gamma_matrix, xkj_matrix):
        t_L_comp_k = alpha_matrix * self.Ik * self.zk / self.flk
        t_E_comp_k_j = beta_matrix * self.Ik * self.zk * np.sum(xkj_matrix, 0) / self.fENj
        t_C_comp_k_j = gamma_matrix * self.Ik * self.zk / self.fCk
        t_E_off_k_j = (beta_matrix + gamma_matrix) * self.Ik * np.sum(xkj_matrix, 0) / (
                self.W * np.log(1 + (self.Ptm * (self.dist ** -self.v) * self.h1 ** 2) / self.N0))
        t_B_off_k_j = gamma_matrix * self.Ik * np.sum(xkj_matrix, 0) / self.Mj
        t_L_k = np.maximum(t_L_comp_k, np.sum(t_E_off_k_j, 1).reshape(self.K, 1))
        t_E_k_j = np.maximum(t_E_comp_k_j, t_B_off_k_j)
        t_C_k_j = t_C_comp_k_j
        T_k_cal = t_L_k + np.sum(xkj_matrix * t_E_k_j, 1).reshape(self.K, 1) + np.sum(xkj_matrix * t_C_k_j, 1).reshape(
            self.K, 1)
        return T_k_cal

    def particle_normalization(self, particle_i):
        """
        规范粒子，主要是使粒子满足 alpha + beta + gamma = 1
        """
        alpha_list = particle_i[0:self.K]
        beta_list = particle_i[self.K:2 * self.K]
        gamma_list = particle_i[2 * self.K:3 * self.K]
        nor_particle = np.zeros(np.shape(particle_i))
        for k in range(self.K):
            alpha_k = alpha_list[k]
            beta_k = beta_list[k]
            gamma_k = gamma_list[k]
            sum_abg = alpha_k + beta_k + gamma_k + 0.0001

            beta_k = beta_k / sum_abg
            gamma_k = gamma_k / sum_abg
            alpha_k = 1 - beta_k - gamma_k

            nor_particle[k] = alpha_k
            nor_particle[k + self.K] = beta_k
            nor_particle[k + self.K * 2] = gamma_k

        nor_particle[3 * self.K:4 * self.K] = particle_i[3 * self.K:4 * self.K]
        return nor_particle

    def objfunction(self, particle):
        # particle_norma = self.particle_normalization(particle)
        # print(type(particle), np.shape(particle))
        alpha_m, beta_m, gamma_m, xkj_m = self.transform_position_to_matrix(particle)
        # F
        f_ue = np.sum(self.pUEk * self.zk * self.Ik * self.qLk * alpha_m * self.flk ** 2)
        f_en = np.sum(self.pENj * self.Ik * self.zk * self.qENj * beta_m * self.fENj ** 2)
        f_cloud = np.sum(self.rc * self.Ik * self.zk * gamma_m)
        F = f_ue + f_en + f_cloud
        T_k = self.cal_t_k(alpha_m, beta_m, gamma_m, xkj_m)
        # Constraint
        inequality = 0
        # C 26
        Ek = self.Ik * self.zk * alpha_m * self.qLk * self.flk ** 2
        inequality += np.sum(np.maximum(Ek - self.Ek_max, 0))
        cont_1 = np.sum(np.maximum(Ek - self.Ek_max, 0))
        # C 27
        EENj = np.sum(self.Ik * self.zk * beta_m * self.qENj * self.fENj ** 2, 0)
        inequality += np.sum(np.maximum(EENj - self.EEN_max, 0))
        cont_2 = np.sum(np.maximum(EENj - self.EEN_max, 0))
        # C 28
        inequality += np.sum(np.maximum(T_k - self.T_k_max, 0))
        # cont_3 = np.sum(np.maximum(T_k - self.T_k_max, 0))
        cont_3 = np.maximum(T_k - self.T_k_max, 0)
        # C 29
        phi1j = np.sum(self.Ik * beta_m * self.zk, 0)
        inequality += np.sum(np.maximum(phi1j - self.phi1j_max, 0))
        cont_4 = np.sum(np.maximum(phi1j - self.phi1j_max, 0))
        # C 30
        phi2j = np.sum(self.Ik * beta_m * self.wk, 0)
        inequality += np.sum(np.maximum(phi2j - self.phi2j_max, 0))
        cont_5 = np.sum(np.maximum(phi2j - self.phi2j_max, 0))
        # C 31
        abg_sum = alpha_m + np.sum(beta_m, 1).reshape(self.K, 1) + np.sum(gamma_m, 1).reshape(self.K, 1)
        if np.abs(np.sum(abg_sum) - self.K) >= 0.01:
            print('There is a grievous mistake!, Sum of alpha, beta and gamma ERROR', abg_sum)
        # C 34
        inequality += np.sum(np.maximum(np.sum(xkj_m, 0) - self.Sj, 0))
        cont_6 = np.sum(np.maximum(np.sum(xkj_m, 0) - self.Sj, 0))
        penalty = inequality * self.penalty_param
        objvalue = F + penalty
        infos = {'UE能量': cont_1,
                 'EN能量': cont_2,
                 '时间': cont_3,
                 'phi1': cont_4,
                 'phi2': cont_5,
                 'Sj': cont_6,
                 'penalty': penalty,
                 'profit': F}
        return objvalue

        # return objvalue, infos

    def evaluate(self, code_string):
        try:
            # Suppress warnings # 抑制警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Create a new module object # 创建一个新的模块对象
                heuristic_module = types.ModuleType("heuristic_module")

                # Execute the code string in the new module's namespace
                # 在新模块的命名空间中执行代码字符串
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                # 将模块添加到sys.modules中以便可以导入
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Now you can use the module as you would any other
                # 现在可以像使用其他模块一样使用它
                eva_instance = copy.deepcopy(self.inited_positions)
                # fitness = heuristic_module.algo(eva_instance, self.upper, self.lower,
                #                                 self.objfunction)  # 输入初始化种群和目标函数即可
                final_solution = heuristic_module.algo(eva_instance, self.upper, self.lower,
                                                       self.objfunction)  # 输入初始化种群和目标函数即可
                fitness = self.objfunction(final_solution)
                return fitness
        except Exception as e:
            print("Error:", str(e))
            return None
        # try:
        #     heuristic_module = importlib.import_module("ael_alg")
        #     eva = importlib.reload(heuristic_module)
        #     fitness = self.greedy(eva)
        #     return fitness
        # except Exception as e:
        #     print("Error:",str(e))
        #     return None


if __name__ == "__main__":
    # file_path = "D:\\00_Work\\00_CityU\\04_AEL_MEC\\test_code\\algocode_gpt.txt"
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     function_code = file.read()
    mecenv = MECENV()
    # print(mecenv.evaluate(function_code))
    # print(np.shape(mecenv.inited_positions))

    # qw-plus2
    import numpy as np


    def algo(initial_population, individual_upper, individual_lower, objective_function):
        n_particles = len(initial_population)
        dim = len(initial_population[0])
        K = dim // 3
        max_iter = 1000
        w = 0.7  # inertia weight
        c1 = 2  # cognitive acceleration coefficient
        c2 = 2  # social acceleration coefficient

        population = initial_population.copy()
        velocities = np.zeros_like(population)

        p_best = population.copy()
        g_best = population[np.argmin([objective_function(p) for p in population])]

        for t in range(max_iter):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities = w * velocities + c1 * r1 * (p_best - population) + c2 * r2 * (g_best - population)
            population += velocities

            for i in range(n_particles):
                population[i] = np.clip(population[i], individual_lower, individual_upper)

            for i in range(n_particles):
                fitness_i = objective_function(population[i])
                if fitness_i < objective_function(p_best[i]):
                    p_best[i] = population[i]

                    if fitness_i < objective_function(g_best):
                        g_best = population[i]

                        # Apply VNS to escape local optima
                        for k in range(K):
                            neighbor = population[i].copy()
                            neighbor[k] = np.random.uniform(individual_lower[k], individual_upper[k])
                            fitness_neighbor = objective_function(neighbor)

                            if fitness_neighbor < objective_function(population[i]):
                                population[i] = neighbor
                                break

        return g_best

    # qw-plus1
    # import numpy as np
    #
    #
    # def algo(initial_population, individual_upper, individual_lower, objective_function):
    #     num_harmonies, num_variables = initial_population.shape
    #     harmony_memory = initial_population
    #     best_solution = harmony_memory[0]
    #     hmcr = 0.8  # harmony memory consideration rate
    #     par = 0.3  # pitch adjustment rate
    #     bw = (individual_upper - individual_lower) / 100  # bandwidth for pitch adjustment
    #
    #     for _ in range(1000):
    #         new_harmony = np.zeros(num_variables)
    #
    #         for i in range(num_variables):
    #             if np.random.rand() < hmcr:
    #                 new_harmony[i] = harmony_memory[np.random.randint(num_harmonies), i]
    #
    #                 if np.random.rand() < par:
    #                     if np.random.rand() < 0.5:
    #                         new_harmony[i] += bw[i]
    #                     else:
    #                         new_harmony[i] -= bw[i]
    #
    #                     new_harmony[i] = np.clip(new_harmony[i], individual_lower[i], individual_upper[i])
    #             else:
    #                 new_harmony[i] = np.random.uniform(individual_lower[i], individual_upper[i])
    #
    #         new_harmony_fitness = objective_function(new_harmony)
    #
    #         worst_index = np.argmax([objective_function(harmony) for harmony in harmony_memory])
    #         if new_harmony_fitness < objective_function(harmony_memory[worst_index]):
    #             harmony_memory[worst_index] = new_harmony
    #
    #         best_solution = harmony_memory[np.argmin([objective_function(harmony) for harmony in harmony_memory])]
    #
    #     return best_solution

    # gpt4o
    # import numpy as np
    # def algo(initial_population, individual_upper, individual_lower, objective_function):
    #     max_iter = 1000
    #     population_size, num_vars = initial_population.shape
    #     K = num_vars // 3
    #
    #     def update_solution(solution, leader, diversity, t):
    #         alpha = 2 - 2 * t / max_iter
    #         r = np.random.random(size=solution.shape)
    #         C = 2 * r
    #         A = alpha * (2 * r - 1)
    #         new_solution = leader - A * np.abs(C * leader - solution)
    #         return np.clip(new_solution, individual_lower, individual_upper)
    #
    #     best_solution = initial_population[0]
    #     best_fitness = objective_function(best_solution)
    #
    #     for t in range(max_iter):
    #         fitness_values = np.apply_along_axis(objective_function, 1, initial_population)
    #         sorted_indices = np.argsort(fitness_values)
    #         population = initial_population[sorted_indices]
    #
    #         top_n = population_size // 5
    #         leaders = population[:top_n]
    #         leader_fitnesses = fitness_values[sorted_indices[:top_n]]
    #         diversity = np.mean(np.std(leaders, axis=0))
    #
    #         for i in range(population_size):
    #             leader_idx = np.random.randint(0, top_n)
    #             leader = leaders[leader_idx]
    #             initial_population[i] = update_solution(initial_population[i], leader, diversity, t)
    #
    #         current_best_fitness = fitness_values[sorted_indices[0]]
    #         if current_best_fitness < best_fitness:
    #             best_fitness = current_best_fitness
    #             best_solution = population[0]
    #
    #     return best_solution


    # print(mecenv.human_design_algo(mecenv.inited_positions, mecenv.upper, mecenv.lower, mecenv.objfunction))
    # print(algo(mecenv.inited_positions, mecenv.upper, mecenv.lower, mecenv.objfunction))
    testinstance_gspso = copy.deepcopy(mecenv.inited_positions)
    testinstance_llm = copy.deepcopy(mecenv.inited_positions)

    llmsolution = algo(testinstance_llm, mecenv.upper, mecenv.lower, mecenv.objfunction)
    gspsosolution = mecenv.human_design_algo(testinstance_gspso, mecenv.upper, mecenv.lower, mecenv.objfunction)


    print('gspso result:', mecenv.objfunction(gspsosolution))
    print('llm result:', mecenv.objfunction(llmsolution))
