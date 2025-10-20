import numpy as np
import pickle
import sys
import types
import warnings
import copy
from joblib import Parallel, delayed
import random
from scipy.stats import qmc  # 用于拉丁超立方采样
from .util_mec import initialize_mec_environment

from .prompts import GetPrompts

class mec_instance():
    def __init__(self, S=5, J=16, _Is_mean=150, granularity=1, simple_mode=False):
        """
        :param S:
        :param J:
        :param _Is_mean: GB 的 data
        :param granularity:
        """
        # random.seed(2024)  # 设置种子
        # np.random.seed(2024)  # 设置种子
        # 固定参数
        self.S = S
        self.J = J
        self.Is_mean = _Is_mean
        self.partical_nums = 50
        self.simple_mode = simple_mode
        self.build_system_model()

        self.granularity = granularity
        self.set_boundary(self.granularity)

    def build_system_model(self):
        # 初始化用户的请求计算量
        np.random.seed(2024)
        self.I_s_GB = np.random.normal(self.Is_mean, 30, self.S) if not self.simple_mode else np.full(self.S,
                                                                                                      self.Is_mean)  # 单位 GB
        # print('用户请求(GB)', self.I_s_GB)
        min_I_s = 10
        # print('Mean of I_s', np.mean(self.I_s_GB), ' | needed means', self.Is_mean)
        self.I_s_GB = np.clip(self.I_s_GB, min_I_s, None)
        self.I_s = self.I_s_GB * (1024 ** 3)  # 转换为byte
        # 初始化用户的应用类型以及对应的
        self.z_S = np.array([3000, 3500, 4000, 1000])  # FLOPs/byte
        self.O_S = np.array([1 / 50, 1 / 80, 1 / 150, 1 / 6])  # 返回的数据大小占输入数据大小的比例 ratio
        self.z_s = np.array([random.randint(0, len(self.z_S) - 1) for _ in range(self.S)])
        z_s = self.z_S[self.z_s]
        self.d_s = self.I_s * z_s

        self.f1_s = np.random.normal(5.6e11, 1e10, self.S)  # 终端计算速度 FLOPs/s
        self.f2_j = 10.1e12  # 边缘服务器计算速度 FLOPs/s
        self.f3 = 48.6e12  # 云端计算速度 FLOPs/s

        # unit_flop = 9/3600  # 0.01美元每亿FLOPs - 云端
        # self.pai_2 = unit_flop / self.f2_j  # 云端每FLOP成本 $/FLOP
        unit_flop = 0.0001  # 0.0002美元每亿FLOPs - 云端
        self.pai_2 = unit_flop / 1e8  # 云端每FLOP成本 $/FLOP
        unit_flop_en = self.pai_2 * 0.6
        self.psi_1_j = np.random.normal(unit_flop_en, unit_flop_en / 5, self.J)  # 边缘端每FLOP成本 $/FLOP
        min_psi_1_j = unit_flop_en * 0.9
        self.psi_1_j = np.clip(self.psi_1_j, min_psi_1_j, None)

        self.qLk = 1e-10  # 终端芯片能耗  J/FLOP

        self.phi1j_max = 60 * self.f1_s  # 每次调度间隙，边缘服务器的最大FLOPs
        self.phi2j_max = np.full(self.J, 1 * 1024 * 1024 * 1024 * 1024)  # bytes -> 边缘服务器的最大内存数 1 TB

        self.B_j = np.full(self.J, 8e8)  # 终端与边缘服务器的信道带宽 800Mhz 毫米波频段
        self.P_s = np.full(self.S, 0.25)  # 终端的传输功率 0.25W
        self.P_j = np.full(self.J, 20)
        self.v = 3  # 路径损耗的指数
        self.h1 = 0.99  # 一个圆对称复高斯随机变量
        self.N0 = 10 ** (-20.4)  # 噪声功率谱密度 W/Hz
        self.Mj = np.full(self.J, (1e11) / 8)  # bytes/s=Bps  光纤 100 Gbps = 100 * 10 ** 9 边缘服务器与云之间有线回程链路的传输速率
        self.connectiong_radius = 300  # 可以连接的距离

        self.Es_max = 324000 / 2  # J -> 能量焦耳 45Wh笔记本电脑
        self.Sj = 1000000  # 每个EN的可连接设备数量

        # self.T_s_max = self.d_s / ((self.f1_s + self.f2_j) / 1.5)  # 给顾客承诺的时间
        self.T_s_max = self.d_s / self.f1_s * 0.5

        _, _, self.distances_stj, self.distances_jtj, self.distance_sts = initialize_mec_environment(self.S, self.J,
                                                                                                     1000, 1000,
                                                                                                     radiu=self.connectiong_radius,
                                                                                                     plot_flag=False)
        self.pi_S = []
        for s in range(self.S):
            self.pi_S.append([])
            for j in range(self.J):
                if self.distances_stj[s, j] <= self.connectiong_radius:
                    self.pi_S[-1].append(j)
            if len(self.pi_S[-1]) == 0:
                print(self.pi_S)
                raise Exception("初始位置中有未被覆盖的移动设备")

        self.len_pi_S = [len(self.pi_S[i]) for i in range(self.S)]
        self.c_s_j = np.zeros((self.S, self.J))
        # Randomly assign base stations to mobile devices
        for s in range(self.S):
            self.c_s_j[s, random.choice(self.pi_S[s])] = 1

        # P_matrix = self.P_s[:, np.newaxis]  # 形状 (S, 1)
        # distance_matrix = self.distance_sts ** -self.v  # 形状 (S, S)
        # np.fill_diagonal(distance_matrix, 0)  # 自身干扰为 0
        # self.interference_matrix = P_matrix * distance_matrix  # 广播乘法

        self.penalty_param = 1000

    def set_boundary(self, granularity):
        """
        :param granularity: 优化问题的形态
        granularity == 0 : 优化两种决策变量 a (迁移的任务的比例) 和 b (迁移的任务在ENs上执行的比例)
        granularity == 1 : 优化三种决策变量 a, b, 和 e (实际执行计算的ENj, Ratio)
        granularity == 2 : 优化三种决策变量 a, b, c (Ratio) 和 e (Ratio)
        granularity == 3 : 优化两个决策变量 a (所有移动设备的平均迁移任务的比例), b (所有移动设备的迁移任务在ENs上执行的比例)
        """
        self.granularity = granularity
        if self.granularity == 0:
            """
            只优化a和b
            """
            self.dimension_num = 2 * self.S
            self.upper = np.zeros(self.dimension_num)
            self.lower = np.zeros(self.dimension_num)

            for k_ind in range(self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0
            for k_ind in range(self.S, 2 * self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0

        elif self.granularity == 1:
            self.dimension_num = 3 * self.S
            self.upper = np.zeros(self.dimension_num)
            self.lower = np.zeros(self.dimension_num)
            for k_ind in range(self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0
            for k_ind in range(self.S, 2 * self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0
            for k_ind in range(2 * self.S, 3 * self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0

        elif self.granularity == 2:
            self.dimension_num = 4 * self.S
            self.upper = np.zeros(self.dimension_num)
            self.lower = np.zeros(self.dimension_num)
            for k_ind in range(self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0
            for k_ind in range(self.S, 2 * self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0
            for k_ind in range(2 * self.S, 3 * self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0
            for k_ind in range(3 * self.S, 4 * self.S):
                self.upper[k_ind] = 1
                self.lower[k_ind] = 0
        elif self.granularity == 3:
            self.dimension_num = 2
            self.upper = np.ones(self.dimension_num)
            self.lower = np.zeros(self.dimension_num)
        else:
            raise Exception("Granularity wrong")

        self.inited_positions = self.init_positions()

    # def init_positions(self):
    #     positions = np.zeros((self.partical_nums, self.dimension_num))
    #     for ind in range(self.partical_nums):
    #         for d in range(self.dimension_num):
    #             positions[ind, d] = np.random.uniform(self.lower[d], self.upper[d])
    #     return positions

    def init_positions(self):
        # 计算随机初始化和LHS初始化的粒子数量
        random_num = int(self.partical_nums * 0.5)
        lhs_num = self.partical_nums - random_num

        # 随机初始化部分
        random_positions = np.random.uniform(
            low=self.lower, high=self.upper, size=(random_num, self.dimension_num)
        )

        # 拉丁超立方采样初始化部分
        sampler = qmc.LatinHypercube(d=self.dimension_num)
        lhs_sample = sampler.random(n=lhs_num)
        lhs_positions = qmc.scale(lhs_sample, self.lower, self.upper)

        # 合并两部分初始化结果
        positions = np.vstack((random_positions, lhs_positions))
        return positions

    def transform_granularity(self, o, d, ori_solution):
        if o != 3:
            raise Exception('只能从粗转换到细')

        if d == 0:
            dimension_num = 2 * self.S

        elif d == 1:
            dimension_num = 3 * self.S

        elif d == 2:
            dimension_num = 4 * self.S
        else:
            raise Exception('目标粒度错误')

        target_solution = np.random.rand(dimension_num)
        target_solution[:self.S] = np.full(self.S, ori_solution[0])[:]
        target_solution[self.S:2 * self.S] = np.full(self.S, ori_solution[1])[:]
        return target_solution

    def transform_position_to_matrix(self, position):
        """
        granularity = 0 : 优化两种决策变量 a (迁移的任务的比例) 和 b (迁移的任务在ENs上执行的比例)
        granularity = 1 : 优化三种决策变量 a, b, 和 e (实际执行计算的ENj, Ratio)
        granularity = 2 : 优化三种决策变量 a, b, c (Ratio) 和 e (No.)
        """
        alpha_array, beta_array, gamma_array = np.zeros(self.S), np.zeros(self.S), np.zeros(self.S)
        c_matrix = copy.deepcopy(self.c_s_j)
        e_matrix = np.zeros((self.S, self.J))
        if self.granularity == 0:
            alpha_array[:] = position[:self.S]
            beta_array[:] = (1 - alpha_array[:]) * position[self.S:2 * self.S]
            gamma_array[:] = 1 - alpha_array[:] - beta_array[:]
            e_matrix[:, :] = c_matrix[:, :]
        elif self.granularity == 1:
            alpha_array[:] = position[:self.S]
            beta_array[:] = (1 - alpha_array[:]) * position[self.S:2 * self.S]
            gamma_array[:] = 1 - alpha_array[:] - beta_array[:]
            for index, ratio in enumerate(position[2 * self.S:3 * self.S]):
                e_matrix[index, np.clip(np.floor(self.J * ratio).astype(int), 0, self.J - 1)] = 1
        elif self.granularity == 2:
            alpha_array[:] = position[:self.S]
            beta_array[:] = (1 - alpha_array[:]) * position[self.S:2 * self.S]
            gamma_array[:] = 1 - alpha_array[:] - beta_array[:]
            for index, ratio in enumerate(position[2 * self.S:3 * self.S]):
                e_matrix[index, np.clip(np.floor(self.J * ratio).astype(int), 0, self.J - 1)] = 1
            for index, ratio in enumerate(position[3 * self.S:4 * self.S]):
                target_index = np.clip(np.floor(self.len_pi_S[index] * ratio).astype(int), 0, self.len_pi_S[index] - 1)
                c_matrix[index, self.pi_S[index][target_index]] = 1
        elif self.granularity == 3:
            alpha_array[:] = np.full(self.S, position[0])
            beta_array[:] = (1 - alpha_array[:]) * position[1]
            gamma_array[:] = 1 - alpha_array[:] - beta_array[:]
            e_matrix[:, :] = c_matrix[:, :]

        return alpha_array, beta_array, gamma_array, c_matrix, e_matrix

    def cal_transmission_speed(self, c_matrix):
        # 计算干扰项矩阵
        # common_base_station_matrix = c_matrix @ c_matrix.T
        # common_base_station_matrix = (common_base_station_matrix > 0).astype(int)
        # total_interference_matrix = common_base_station_matrix * self.interference_matrix
        # total_interference = np.sum(total_interference_matrix, axis=1)
        total_interference = np.zeros(self.S)

        # 分母的基础项矩阵
        denominator_base = self.N0 * self.B_j / np.clip(np.sum(c_matrix, axis=0), 1, None)  # 形状 (J,)
        # 计算分子
        numerator = self.P_s[:, np.newaxis] * (self.distances_stj ** -self.v) * abs(self.h1) ** 2  # 形状 (S, J)
        # 计算最终的 R 矩阵
        Rup = (self.B_j / np.clip(np.sum(c_matrix, axis=0), 1, None)) * np.log2(
            1 + numerator / (denominator_base + total_interference[:, np.newaxis]))  # 单位 bps
        Rup = Rup / 8  # 单位Bps

        # 下载速率
        numerator_down = self.P_j * (self.distances_stj ** -self.v) * abs(self.h1) ** 2  # 形状 (S, J)
        Rdown = (self.B_j / np.clip(np.sum(c_matrix, axis=0), 1, None)) * np.log2(
            1 + numerator_down / (denominator_base + total_interference[:, np.newaxis]))  # 单位 bps

        Rdown = Rdown / 8
        return Rup, Rdown

    def cal_t_k(self, alpha_array, beta_array, gamma_array, c_matrix, e_matrix):
        """
        :param alpha_array: np.array([]) len=S
        :param beta_array: np.array([]) len=S
        :param gamma_array: np.array([]) len=S
        :param c_matrix: np.array([[]]) size=S*J
        :param e_matrix: np.array([[]]) size=S*J
        :return:
        """
        # 三端执行的bytes
        en_Is = beta_array * self.I_s
        cdc_Is = gamma_array * self.I_s
        # 三端执行的FLOPs
        local_ds = alpha_array * self.d_s
        en_ds = beta_array * self.d_s
        cdc_ds = gamma_array * self.d_s
        # 执行时间
        tau1_s = local_ds / self.f1_s
        # sum_e_sj = np.sum(e_matrix, axis=0)  # 对每列求和，结果为长度为 j 的数组
        sum_e_sj = np.clip(np.sum(e_matrix, axis=0), 1, None)
        tau2_sj = en_ds[:, np.newaxis] / (self.f2_j / sum_e_sj) * e_matrix  # 广播实现矩阵运算
        tau3_sj = (cdc_ds / self.f3)[:, np.newaxis] * c_matrix

        # 无线传输速度
        R_s_j_up, R_s_j_down = self.cal_transmission_speed(c_matrix)
        Oa_s = en_Is * self.O_S[self.z_s]
        Ob_s = cdc_Is * self.O_S[self.z_s]
        # 基站执行任务的上传时间
        tau1up = en_Is[:, np.newaxis] / R_s_j_up * c_matrix
        # 基站与基站之间的传输时间
        diff = np.abs(c_matrix - e_matrix)
        sum_term = np.sum(diff, axis=1)
        selected_M = self.Mj[np.argmax(c_matrix, axis=1)]  # 每行 1 所在的列索引，选取对应的 M 值
        # 基站上传至基站
        tau2right_s = (en_Is * sum_term) / selected_M
        # 基站下载结果至基站
        tau2left_s = (Oa_s * sum_term) / selected_M

        # 云端执行任务的上传时间
        tau3up_s = (1 / R_s_j_up + 1 / selected_M[:, np.newaxis]) * cdc_Is[:, np.newaxis] * c_matrix
        tau3down_s = (Ob_s / selected_M)[:, np.newaxis] * c_matrix

        # 边缘和云端任务全部执行完毕后统一传回的时间
        tau_down_s = (Oa_s + Ob_s)[:, np.newaxis] / R_s_j_down * c_matrix

        # 边缘和云谁先返回
        en_time_up_and_execute = tau1up + tau2_sj
        temp1 = np.sum(en_time_up_and_execute, axis=1)
        en_time_up_execute_and_wait_to_back = temp1 + tau2left_s + tau2right_s
        cdc_time_up_execute_backen_and_wait_to_back = np.sum(tau3_sj + tau3up_s + tau3down_s, axis=1)
        max_1 = np.maximum(en_time_up_execute_and_wait_to_back, cdc_time_up_execute_backen_and_wait_to_back)
        temp2 = max_1 + np.sum(tau_down_s, axis=1)
        T_s = np.maximum(temp2, tau1_s)

        return T_s

    def objfunction(self, particle):
        """
        alpha_s, beta_s, gamma_s是三个np.array([0.1, 0.2, ...., 0.2])
        c_s_j, e_s_j 是两个大小为 S * J 的np.array矩阵
        """
        alpha_s, beta_s, gamma_s, c_s_j, e_s_j = self.transform_position_to_matrix(particle)

        # F
        b_1 = np.sum(self.psi_1_j * self.d_s[:, np.newaxis] * beta_s[:, np.newaxis] * e_s_j)
        b_2 = np.sum(self.pai_2 * self.d_s[:, np.newaxis] * gamma_s[:, np.newaxis] * c_s_j)

        F_Cost = b_1 + b_2

        T_s = self.cal_t_k(alpha_s, beta_s, gamma_s, c_s_j, e_s_j)
        # Constraint
        # 能量消耗
        E_s = self.d_s * alpha_s * self.qLk
        cont_1 = np.sum(np.maximum(E_s - self.Es_max, 0))
        # 基站内存消耗
        Phi2 = np.sum((self.I_s * beta_s)[:, np.newaxis] * e_s_j, 0)
        cont_2 = np.sum(np.maximum(Phi2 - self.phi2j_max, 0))
        # 时间延迟
        cont_3 = np.sum(np.maximum(T_s - self.T_s_max, 0))
        # C 31
        abg_sum = alpha_s + beta_s + gamma_s
        bias = np.sum(abg_sum - np.full(self.S, 1))
        if bias > 1e-10:
            print('There is a grievous mistake!, Sum of alpha, beta and gamma ERROR', abg_sum)

        inequality = cont_1 + cont_2 + cont_3 + bias

        penalty = inequality * self.penalty_param
        objvalue = F_Cost + penalty
        return objvalue

    def objfunction_observe(self, particle):
        """
        alpha_s, beta_s, gamma_s是三个np.array([0.1, 0.2, ...., 0.2])
        c_s_j, e_s_j 是两个大小为 S * J 的np.array矩阵
        """
        alpha_s, beta_s, gamma_s, c_s_j, e_s_j = self.transform_position_to_matrix(particle)

        # F
        b_1 = np.sum(self.psi_1_j * self.d_s[:, np.newaxis] * beta_s[:, np.newaxis] * e_s_j)
        b_2 = np.sum(self.pai_2 * self.d_s[:, np.newaxis] * gamma_s[:, np.newaxis] * c_s_j)

        F_Cost = b_1 + b_2

        T_s = self.cal_t_k(alpha_s, beta_s, gamma_s, c_s_j, e_s_j)
        # Constraint
        # 能量消耗
        E_s = self.d_s * alpha_s * self.qLk
        cont_1 = np.sum(np.maximum(E_s - self.Es_max, 0))
        # 基站内存消耗
        Phi2 = np.sum((self.I_s * beta_s)[:, np.newaxis] * e_s_j, 0)
        cont_2 = np.sum(np.maximum(Phi2 - self.phi2j_max, 0))
        # 时间延迟
        cont_3 = np.sum(np.maximum(T_s - self.T_s_max, 0))
        # C 31
        abg_sum = alpha_s + beta_s + gamma_s
        bias = np.sum(abg_sum - np.full(self.S, 1))
        if bias > 1e-10:
            print('There is a grievous mistake!, Sum of alpha, beta and gamma ERROR', abg_sum)

        inequality = cont_1 + cont_2 + cont_3 + bias

        penalty = inequality * self.penalty_param
        objvalue = F_Cost + penalty
        info = {'objvalue': objvalue,
                'penalty': inequality,
                'cost': F_Cost,
                'Time constrain': cont_3,
                'energy constrain': cont_1,
                'memory constrain': cont_2
                }
        return objvalue, info

class MECENV():
    def __init__(self) -> None:
        self.prompts = GetPrompts()
        self.can_visualize = False
        self.taskname='task_offloading'
        self.instance_num = 3
        self.instances = [
            # mec_instance(5, 16, 150),
                         mec_instance(15, 16, 150),
                         # mec_instance(30, 16, 150)
                          ]

        def compute_human_grade(instance):

            return instance.objfunction(
                self.human_design_algo(copy.deepcopy(instance.inited_positions), instance.upper, instance.lower,
                                       instance.objfunction)
            )

        # 每个instance运行两次，保存结果
        results = Parallel(n_jobs=3)(
            delayed(compute_human_grade)(instance) for instance in self.instances for _ in range(2))

        # 计算每个instance的最小值和平均值
        min_grades = []
        average_grades = []

        # 对于每个instance，取出两次运行的结果
        for i, instance in enumerate(self.instances):
            instance_results = results[i * 2:i * 2 + 2]  # 取出该instance的两次结果
            min_grades.append(min(instance_results))  # 取最小值
            average_grades.append(sum(instance_results) / len(instance_results))  # 取平均值

        # 计算最终的min_grade和average_grade
        min_grade = sum(min_grades)
        average_grade = sum(average_grades)

        # 输出结果
        print('#### Human grades are ', results)
        print('#### Minimum human grade is ', min_grade)
        print('#### Average human grade is ', average_grade)

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
        position_global_fitness = objfunction_input(position_global)


        population_fitnesses = np.zeros(Number_of_partical_gspso)
        # 初始化全局最优
        for i in range(Number_of_partical_gspso):
            population_fitnesses[i] = objfunction_input(in_population[i])
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

                def compute_fitness(instance):
                    eva_instance_init = copy.deepcopy(instance.inited_positions)
                    final_solution = heuristic_module.algo(eva_instance_init, instance.upper, instance.lower,
                                                           instance.objfunction)
                    return instance.objfunction(final_solution)

                # Now you can use the module as you would any other
                # 使用Parallel并行化for循环
                fitnesses = [compute_fitness(instance) for instance in self.instances]

                # 计算fitnesses的平均值
                return np.sum(fitnesses)
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
    file_path = "D:\\00_Work\\00_CityU\\04_AEL_MEC\\test_code\\algocode_gpt.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        function_code = file.read()
    mecenv = MECENV()
    print(mecenv.evaluate(function_code))


    # qw-plus2
    # import numpy as np
    #
    #
    # def algo(initial_population, individual_upper, individual_lower, objective_function):
    #     n_particles = len(initial_population)
    #     dim = len(initial_population[0])
    #     K = dim // 3
    #     max_iter = 1000
    #     w = 0.7  # inertia weight
    #     c1 = 2  # cognitive acceleration coefficient
    #     c2 = 2  # social acceleration coefficient
    #
    #     population = initial_population.copy()
    #     velocities = np.zeros_like(population)
    #
    #     p_best = population.copy()
    #     g_best = population[np.argmin([objective_function(p) for p in population])]
    #
    #     for t in range(max_iter):
    #         r1, r2 = np.random.rand(dim), np.random.rand(dim)
    #         velocities = w * velocities + c1 * r1 * (p_best - population) + c2 * r2 * (g_best - population)
    #         population += velocities
    #
    #         for i in range(n_particles):
    #             population[i] = np.clip(population[i], individual_lower, individual_upper)
    #
    #         for i in range(n_particles):
    #             fitness_i = objective_function(population[i])
    #             if fitness_i < objective_function(p_best[i]):
    #                 p_best[i] = population[i]
    #
    #                 if fitness_i < objective_function(g_best):
    #                     g_best = population[i]
    #
    #                     # Apply VNS to escape local optima
    #                     for k in range(K):
    #                         neighbor = population[i].copy()
    #                         neighbor[k] = np.random.uniform(individual_lower[k], individual_upper[k])
    #                         fitness_neighbor = objective_function(neighbor)
    #
    #                         if fitness_neighbor < objective_function(population[i]):
    #                             population[i] = neighbor
    #                             break
    #
    #     return g_best


