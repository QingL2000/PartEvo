import numpy as np

import sys
import types
import warnings
import copy
from joblib import Parallel, delayed

import os
import time
import pickle

from .prompts import GetPrompts

class MLSENV:
    def __init__(self, test_mode=False):
        self.prompts = GetPrompts()
        self.can_visualize = False
        self.taskname = 'machine_level_scheduling'
        Data_init = Dataenv()

        if test_mode:
            self.instance_params = [
                {'I_set': 9, 'P_set': 3, 'np_set': 6, 'T_set': 7, 'data_init': Data_init},
                {'I_set': 9, 'P_set': 5, 'np_set': 6, 'T_set': 7, 'data_init': Data_init},
                {'I_set': 10, 'P_set': 3, 'np_set': 6, 'T_set': 7, 'data_init': Data_init}
            ]
            self.instance_save_path = "D:/00_Work/00_CityU/04_AEL_MEC/hle/test_instances"
        else:
            # 定义实例参数
            self.instance_params = [
                # {"S": 5, "J": 16, "Is_mean": 150},
                {'I_set':9, 'P_set':3, 'np_set':6, 'T_set':7, 'data_init':Data_init},
            ]
            self.instance_save_path = "D:/00_Work/00_CityU/04_AEL_MEC/hle/instances"  # 保存实例的文件夹路径

            # 确保保存路径存在
            if not os.path.exists(self.instance_save_path):
                os.makedirs(self.instance_save_path)

        self.instances =self.load_or_initialize_instances()

        # def compute_human_grade(instance):
        #     return instance.objfunction(
        #         self.human_design_algo(copy.deepcopy(instance.inited_positions), instance.upper, instance.lower,
        #                                instance.objfunction)
        #     )
        # s1= time.time()
        # # 每个instance运行两次，保存结果
        # results = Parallel(n_jobs=8)(
        #     delayed(compute_human_grade)(instance) for instance in self.instances for _ in range(1))
        #
        # # 输出结果
        # print(f'Time consumptiong: {time.time()-s1} sec.')
        # print('#### Human grades are ', results)
        # print('#### Human grades are ', np.sum(results))

    def load_or_initialize_instances(self):
        """
        加载或初始化 MEC 实例。
        如果保存的实例文件存在，则加载；否则初始化并保存。
        """
        instances = []
        for i, params in enumerate(self.instance_params):
            file_name = os.path.join(self.instance_save_path, f"mls_instance_{i}.pkl")
            if os.path.exists(file_name):
                # 如果文件存在，加载实例
                with open(file_name, "rb") as f:
                    instance = pickle.load(f)
                    print(f"Loaded instance from {file_name}")
            else:
                # 如果文件不存在，初始化并保存实例
                instance = Environment(I_set=params["I_set"],
                                       P_set=params["P_set"],
                                       np_set=params["np_set"],
                                       T_set=params["T_set"],
                                       data_init=params["data_init"])
                with open(file_name, "wb") as f:
                    pickle.dump(instance, f)
                    print(f"Initialized and saved instance to {file_name}")
            instances.append(instance)
        return instances

    def human_design_algo(self, in_population, upper, lower, objfunction_input):
        np.random.seed(2025)
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

    def test_evaluate(self, code_string):
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

                def compute_fitness(instance_idx, instance):
                    ss1 = time.time()
                    eva_instance_init = copy.deepcopy(instance.inited_positions)
                    final_solution = heuristic_module.algo(eva_instance_init, instance.upper, instance.lower,
                                                           instance.objfunction)
                    print(f'Instance {instance_idx} finished, spend {time.time() - ss1} sec.')
                    return instance.objfunction(final_solution)

                # Now you can use the module as you would any other
                # 使用Parallel并行化for循环
                fitnesses = [compute_fitness(instance_idx, instance) for instance_idx, instance in
                             enumerate(self.instances)]

                # 计算fitnesses的平均值
                return fitnesses
        except Exception as e:
            print("Error:", str(e))
            return None

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

    def evaluate_solution(self, code_string):
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
                    return final_solution

                # Now you can use the module as you would any other
                # 使用Parallel并行化for循环
                fitnesses = [compute_fitness(instance) for instance in self.instances]

                # 计算fitnesses的平均值
                return fitnesses
        except Exception as e:
            print("Error:", str(e))
            return None


from copy import deepcopy
class Environment:
    def __init__(self, I_set, P_set, np_set, T_set, data_init):
        self.I = I_set
        self.P = P_set
        self.n_p = np_set
        self.T = T_set
        self.hat_fsat = 1
        self.M_i = np.zeros((1, self.I))
        self.delta = 0.5
        self.U_i_p = deepcopy(data_init.U_i_p[:self.P, :self.I])
        self.CAP_p = deepcopy(data_init.CAP_p[:, :self.P])
        self.D_i_t = deepcopy(data_init.D_i_t[:self.T, :self.I])
        self.PAIR_i_i_p = deepcopy(data_init.PAIR_i_i_p[:self.I, :self.I])
        self.B_i_i_p_t = deepcopy(data_init.B_i_i_p_t[:self.I, :self.I])
        self.R_i_p_t = deepcopy(data_init.R_i_p_t[:self.P, :self.I])
        self.a_i_j_p = deepcopy(data_init.a_i_j_p[:self.P, :self.n_p, :self.I])
        self.b_i_j_p = deepcopy(data_init.b_i_j_p[:self.P, :self.n_p, :self.I])
        self.alpha_i_j_p = deepcopy(data_init.alpha_i_j_p[:self.P, :self.n_p, :self.I])
        self.TC_i_p_p = deepcopy(data_init.TC_i_p_p[:self.I, :self.P, :self.P])
        self.H2C_ip = deepcopy(data_init.H2C_ip[:self.P, :self.I])
        self.MLS_ipt = deepcopy(data_init.MLS_ipt[:self.P, :self.I])
        self.PMi = deepcopy(data_init.PMi[:, :self.I])
        self.Original_ip = deepcopy(data_init.Ori_ip[:self.P, :self.I])
        self.PCijpt = self.a_i_j_p + self.alpha_i_j_p * self.b_i_j_p
        self.boundary_1 = self.T * self.P * self.I
        self.boundary_2 = self.boundary_1 + self.T * self.P * (self.I - 8)
        self.boundary_3 = self.boundary_2 + self.I * self.P * self.T * (self.P - 1)
        self.boundary_4 = self.boundary_3 + self.I * self.P * self.T
        self.boundary_5 = self.boundary_4 + self.P * self.T * (self.I - 12) * (self.I - 13)
        print('粒子维度为：', self.boundary_5)
        # 上下限
        self.lower = np.zeros(self.boundary_5)
        self.upper = np.zeros(self.boundary_5)
        self.up_wijpt = self.n_p + 0.999  # 如果大于n_p，则说明不制造
        self.up_zipt = 60
        self.up_sippt = 30
        self.up_xipt = 30
        self.up_riipt = 1.1
        self.upper[:self.boundary_1] = self.up_wijpt
        self.upper[self.boundary_1: self.boundary_2] = self.up_zipt
        self.upper[self.boundary_2: self.boundary_3] = self.up_sippt
        self.upper[self.boundary_3: self.boundary_4] = self.up_xipt
        self.upper[self.boundary_4: self.boundary_5] = self.up_riipt

        #
        # self.upper[: self.boundary_1] = np.full((1, self.boundary_1), self.up_wijpt)[0]
        # # print('---', self.boundary_1, self.boundary_2, self.boundary_2 - self.boundary_1, self.up_zipt)
        # self.upper[self.boundary_1: self.boundary_2] = \
        #     np.full((1, self.boundary_2 - self.boundary_1), self.up_zipt)[0]
        # self.upper[self.boundary_2: self.boundary_3] = \
        #     np.full((1, self.boundary_3 - self.boundary_2), self.up_sippt)[0]
        # self.upper[self.boundary_3: self.boundary_4] = \
        #     np.full((1, self.boundary_4 - self.boundary_3), self.up_xipt)[0]
        # self.upper[self.boundary_4: self.boundary_5] = \
        #     np.full((1, self.boundary_5 - self.boundary_4), self.up_riipt)[0]
        self.penalty_param = 100000

        self.inited_positions = self.position_init(30)


    def position_to_matrix(self, input_position):
        position_to_matrix = np.trunc(input_position).copy()
        wijpt = np.zeros((self.T, self.P, self.n_p, self.I))
        zipt = np.zeros((self.T, self.P, self.I))
        sippt = np.zeros((self.T, self.I, self.P, self.P))
        xipt = np.zeros((self.T, self.P, self.I))
        riipt = np.zeros((self.T, self.P, self.I, self.I))
        # load wijpt
        position_con = 0
        for wijpt_t in range(self.T):
            for wijpt_p in range(self.P):
                for wijpt_i in range(self.I):
                    if position_to_matrix[position_con] < self.n_p:
                        wijpt[wijpt_t, wijpt_p, int(position_to_matrix[position_con]), wijpt_i] = 1
                    position_con += 1
        # load zipt
        for zipt_t in range(self.T):
            for zipt_p in range(self.P):
                for zipt_i in range(8, self.I):
                    zipt[zipt_t, zipt_p, zipt_i] = position_to_matrix[position_con]
                    position_con += 1
        # load sippt
        for sippt_t in range(self.T):
            for sippt_i in range(self.I):
                for sippt_p1 in range(self.P):
                    for sippt_p2 in range(self.P):
                        if sippt_p1 == sippt_p2:
                            continue
                        sippt[sippt_t, sippt_i, sippt_p2, sippt_p1] = position_to_matrix[position_con]
                        position_con += 1
        # load xipt
        for xipt_t in range(self.T):
            for xipt_p in range(self.P):
                for xipt_i in range(self.I):
                    xipt[xipt_t, xipt_p, xipt_i] = position_to_matrix[position_con]
                    position_con += 1
        # load riipt
        for riipt_t in range(self.T):
            for riipt_p in range(self.P):
                for riipt_i1 in range(12, self.I):
                    for riipt_i2 in range(12, self.I):
                        if riipt_i1 == riipt_i2:
                            continue
                        riipt[riipt_t, riipt_p, riipt_i2, riipt_i1] = position_to_matrix[position_con]
                        position_con += 1
        return wijpt, zipt, sippt, xipt, riipt

    def objfunction(self, position_cal):
        # print('shape_position', np.shape(position_cal))
        inequality = 0
        cost_write = np.zeros((3, 1))
        wijpt_obj, zipt_obj, sippt_obj, xipt_obj, riipt_obj = self.position_to_matrix(position_cal)
        Pcost, hxipt_paired = self.cal_Pcost(wijpt_obj, xipt_obj)
        Tcost = self.cal_Tcost(sippt_obj)
        Hcost = self.cal_Hcost(zipt_obj)
        cost_write[0, 0] = Pcost
        cost_write[1, 0] = Tcost
        cost_write[2, 0] = Hcost
        Totalcost = Pcost + Tcost + Hcost
        Store_ipt, wipt_obj = self.cal_storeipt(wijpt_obj, hxipt_paired, sippt_obj, riipt_obj, zipt_obj)
        # con 1
        fsati_obj, mit_obj, uit_obj = self.constraint_1(zipt_obj)
        # inequality += np.sum(np.maximum(self.hat_fsat - fsati_obj, 0))
        inequality += np.sum(np.maximum(self.hat_fsat - uit_obj, 0))
        # con 2
        inequality += np.sum(np.maximum(np.sum(Store_ipt * self.U_i_p, axis=2) - self.CAP_p[0], 0))
        # con 3
        # con 4
        inequality += np.sum(np.maximum(np.sum(riipt_obj, axis=2) - (np.dot(hxipt_paired * wipt_obj, self.B_i_i_p_t.T) + zipt_obj), 0))
        # con 5
        inequality += np.sum(np.maximum(np.sum(riipt_obj, axis=2) - self.R_i_p_t, 0))
        # con 6
        # con 7
        inequality += np.sum(np.maximum(-Store_ipt, 0))
        inequality += self.cal_boundpenalty(position_cal)
        penalty = inequality * self.penalty_param
        obj = penalty + Totalcost
        return obj

    def why_fail(self, position_cal):
        # print('shape_position', np.shape(position_cal))
        inequality = 0
        cost_write = np.zeros((3, 1))
        wijpt_obj, zipt_obj, sippt_obj, xipt_obj, riipt_obj = self.position_to_matrix(position_cal)
        Pcost, hxipt_paired = self.cal_Pcost(wijpt_obj, xipt_obj)
        Tcost = self.cal_Tcost(sippt_obj)
        Hcost = self.cal_Hcost(zipt_obj)
        cost_write[0, 0] = Pcost
        cost_write[1, 0] = Tcost
        cost_write[2, 0] = Hcost
        Totalcost = Pcost + Tcost + Hcost
        Store_ipt, wipt_obj = self.cal_storeipt(wijpt_obj, hxipt_paired, sippt_obj, riipt_obj, zipt_obj)
        # con 1
        fsati_obj, mit_obj, uit_obj = self.constraint_1(zipt_obj)
        # inequality += np.sum(np.maximum(self.hat_fsat - fsati_obj, 0))
        inequality += np.sum(np.maximum(self.hat_fsat - uit_obj, 0))
        # con 2
        inequality += np.sum(np.maximum(np.sum(Store_ipt * self.U_i_p, axis=2) - self.CAP_p[0], 0))
        # con 3
        # con 4
        inequality += np.sum(np.maximum(np.sum(riipt_obj, axis=2) - (np.dot(hxipt_paired * wipt_obj, self.B_i_i_p_t.T) + zipt_obj), 0))
        # con 5
        inequality += np.sum(np.maximum(np.sum(riipt_obj, axis=2) - self.R_i_p_t, 0))
        # con 6
        # con 7
        inequality += np.sum(np.maximum(-Store_ipt, 0))
        inequality += self.cal_boundpenalty(position_cal)
        penalty = inequality * self.penalty_param
        obj = penalty + Totalcost
        return obj

    def objfunction_final(self, position_cal):
        inequality = 0
        cost_write = np.zeros((3, 1))
        wijpt_obj, zipt_obj, sippt_obj, xipt_obj, riipt_obj = self.position_to_matrix(position_cal)
        Pcost, hxipt_paired = self.cal_Pcost(wijpt_obj, xipt_obj)
        Tcost = self.cal_Tcost(sippt_obj)
        Hcost = self.cal_Hcost(zipt_obj)
        cost_write[0, 0] = Pcost
        cost_write[1, 0] = Tcost
        cost_write[2, 0] = Hcost
        Totalcost = Pcost + Tcost + Hcost
        Store_ipt, wipt_obj = self.cal_storeipt(wijpt_obj, hxipt_paired, sippt_obj, riipt_obj, zipt_obj)
        # con 1
        fsati_obj, mit_obj, uit_obj = self.constraint_1(zipt_obj)
        # con_1 = np.sum(np.maximum(self.hat_fsat - fsati_obj, 0))
        con_1 = np.sum(np.maximum(self.hat_fsat - uit_obj, 0))
        inequality += con_1
        # con 2
        con_2 = np.sum(np.maximum(np.sum(Store_ipt * self.U_i_p, axis=2) - self.CAP_p[0], 0))
        inequality += con_2
        # con 3
        # con 4
        con_4 = np.sum(np.maximum(np.sum(riipt_obj, axis=2) - (np.dot(hxipt_paired * wipt_obj, self.B_i_i_p_t.T) + zipt_obj), 0))
        inequality += con_4
        # con 5
        con_5 = np.sum(np.maximum(np.sum(riipt_obj, axis=2) - self.R_i_p_t, 0))
        inequality += con_5
        # con 6
        # con 7
        con_7 = np.sum(np.maximum(-Store_ipt, 0))
        inequality += con_7
        inequality += self.cal_boundpenalty(position_cal)
        penalty = inequality * self.penalty_param
        obj = penalty + Totalcost
        con_sum = np.array([[con_1, con_2, con_4, con_5, con_7]])
        return obj, Totalcost, penalty, fsati_obj, con_sum, cost_write, mit_obj, uit_obj

    def cal_boundpenalty(self, position_cal):
        """
        计算位置 `position_cal` 相对于上下限的惩罚。
        位置在上限以上或下限以下都会被惩罚。
        """
        # 计算超出上下限的情况
        penalty_upper = np.maximum(0, position_cal - self.upper)  # 超过上限的部分
        penalty_lower = np.maximum(0, self.lower - position_cal)  # 低于下限的部分

        # 综合惩罚是超出部分的总和
        penalty = np.sum(penalty_upper + penalty_lower)

        return penalty

    def cal_Pcost(self, wijpt_pcost, xipt_pcost):
        PCipt = np.sum(self.PCijpt * wijpt_pcost, axis=2)
        hxipt = xipt_pcost * self.PMi
        hxipt_pair = self.constrain_3(hxipt)
        # print('test_1', hxipt_pair[0])
        main_hxipt = hxipt_pair > self.MLS_ipt
        hxipt_pair = hxipt_pair * main_hxipt
        pcost = np.sum(PCipt * hxipt_pair)
        return pcost, hxipt_pair

    def cal_Tcost(self, sippt_tcost):
        sipp = np.sum(sippt_tcost, axis=0)
        tcost = np.sum(self.TC_i_p_p * sipp)
        return tcost

    def cal_Hcost(self, zipt_hcost):
        zip_hcost = np.sum(zipt_hcost, axis=0)
        hcost = np.sum(zip_hcost * self.H2C_ip)
        return hcost

    def constrain_3(self, hxipt_input):
        hxipt_pair = deepcopy(hxipt_input)
        if np.shape(hxipt_pair)[2] >= 13:
            hxipt_pair[:, :, 12] = hxipt_pair[:, :, 10] * 4
        if np.shape(hxipt_pair)[2] >= 19:
            hxipt_pair[:, :, 18] = hxipt_pair[:, :, 19]
        return hxipt_pair

    def cal_storeipt(self, wijpt_store, hxipt_paired, sippt_store, riipt_store, zipt_store):
        wipt_store = np.sum(wijpt_store, axis=2)
        increment_ipt = hxipt_paired * wipt_store + np.swapaxes(np.sum(sippt_store, axis=2), 1, 2) + np.sum(riipt_store, axis=2)
        decrement_ipt = zipt_store + np.swapaxes(np.sum(sippt_store, axis=3), 1, 2) + np.sum(riipt_store,
                                                                                             axis=3) + np.dot(
            hxipt_paired * wipt_store, self.B_i_i_p_t.T)
        store_ipt = np.zeros((self.T, self.P, self.I))
        store_ipt[0, :, :] = self.Original_ip[:, :] + increment_ipt[0, :, :] - decrement_ipt[0, :, :]
        for t_store in range(1, self.T):
            store_ipt[t_store, :, :] = store_ipt[t_store - 1, :, :] + increment_ipt[t_store, :, :] - decrement_ipt[
                                                                                                     t_store, :, :]
        return store_ipt, wipt_store

    def constraint_1(self, zipt_c1):
        mit_before = np.zeros((self.T, self.I))
        zit = np.sum(zipt_c1, axis=1)
        mit_before[0, :] = self.M_i[0, :] + self.D_i_t[0, :] - zit[0, :]
        for t_c1 in range(1, self.T):
            mit_before[t_c1, :] = mit_before[t_c1 - 1, :] + self.D_i_t[t_c1, :] - zit[t_c1, :]
        mit = np.maximum(mit_before, 0)
        uit_before = np.zeros((self.T, self.I))
        uit_before[0, :] = (zit[0, :] - self.M_i[0] + self.delta) / (self.D_i_t[0] + self.delta)
        for t_c1 in range(1, self.T):
            uit_before[t_c1, :] = (zit[t_c1, :] - mit[t_c1 - 1, :] + self.delta) / (self.D_i_t[t_c1, :] + self.delta)
        uit = np.maximum(uit_before, 0)
        fsati = np.sum(uit, axis=0) / self.T
        # print(fsati)
        # print(np.shape(fsati))
        return fsati, mit, uit

    def position_init(self, numofparticle):
        position_init = np.zeros((numofparticle, self.boundary_5))
        for num_init in range(numofparticle):
            position_init[num_init, 0:self.boundary_1] = np.random.randint(0, self.up_wijpt, (1, self.boundary_1))[0, :]
            position_init[num_init, self.boundary_1: self.boundary_2] = np.random.randint(0, self.up_zipt, (
                1, self.boundary_2 - self.boundary_1))[0, :]
            position_init[num_init, self.boundary_2: self.boundary_3] = np.random.randint(0, self.up_sippt, (
                1, self.boundary_3 - self.boundary_2))[0, :]
            position_init[num_init, self.boundary_3: self.boundary_4] = np.random.randint(0, self.up_xipt, (
                1, self.boundary_4 - self.boundary_3))[0, :]
            position_init[num_init, self.boundary_4: self.boundary_5] = np.random.randint(0, self.up_riipt, (
                1, self.boundary_5 - self.boundary_4))[0, :]
        return position_init

class Dataenv:
    def __init__(self):
        self.CAP_p = 30 * np.array([[25000, 15000, 10000, 25000, 15000, 10000, 25000, 15000, 10000, 20000]])
        self.U_i_p = np.array([
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 2, 2, 5, 8, 6, 3, 4, 1, 3, 8, 3, 8, 8, 8, 8, 10, 10, 10, 10, 10]])
        self.D_i_t = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10],
                               [0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 10, 30, 10, 20, 40, 10, 20, 30, 10, 10, 10, 10, 20, 20,
                                30, 45, 30, 40, 30, 10]])
        self.PAIR_i_i_p = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.B_i_i_p_t = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.R_i_p_t = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ])
        self.a_i_j_p = np.zeros((10, 10, 30))
        self.a_i_j_p[0, :, :] = np.array([
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        ])
        self.a_i_j_p[1, :, :] = np.array([
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 40, 10, 10, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        ])
        self.a_i_j_p[2, :, :] = np.array([
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 15, 50, 10, 12, 11, 1, 15, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        ])
        self.a_i_j_p[3, :, :] = np.array([
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [7, 5, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        ])
        self.a_i_j_p[4, :, :] = np.array([
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [14, 12, 20, 50, 10, 12, 7, 1, 24, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        ])
        self.a_i_j_p[5, :, :] = np.array([
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 1, 2, 2, 2, 2, 3, 3, 1, 3, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        ])
        self.a_i_j_p[6, :, :] = np.array([
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10,
             10],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 3, 10, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        ])
        self.a_i_j_p[7, :, :] = np.array([
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15,
             15],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 1, 15, 1, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        ])
        self.a_i_j_p[8, :, :] = np.array([
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13,
             13],
            [10, 8, 20, 50, 10, 12, 11, 1, 20, 13, 3, 4, 4, 2, 2, 5, 5, 1, 2, 13, 2, 13, 13, 13, 13, 13, 13, 13, 13, 13]
        ])
        self.a_i_j_p[9, :, :] = np.array([
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14,
             14],
            [10, 8, 20, 45, 10, 10, 11, 1, 20, 13, 3, 2, 4, 2, 2, 5, 5, 1, 3, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14]
        ])
        self.b_i_j_p = np.zeros((10, 10, 30))
        self.b_i_j_p[0, :, :] = np.array([
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.b_i_j_p[1, :, :] = np.array([
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.b_i_j_p[2, :, :] = np.array([
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 0.5, 3, 1, 2, 1, 1, 0.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.b_i_j_p[3, :, :] = np.array([
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0.5, 0.5, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.b_i_j_p[4, :, :] = np.array([
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 0.5, 1, 1.5, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.b_i_j_p[5, :, :] = np.array([
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 3, 4, 1, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.b_i_j_p[6, :, :] = np.array([
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8.5, 1, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
             8.5]])
        self.b_i_j_p[7, :, :] = np.array([
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 0.5, 8, 0.5, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.b_i_j_p[8, :, :] = np.array([
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.b_i_j_p[9, :, :] = np.array([
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [1, 1, 1, 2.5, 1, 1.5, 1, 1, 1, 1, 2, 2, 3, 5, 3, 5, 2, 0.5, 1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
        self.alpha_i_j_p = deepcopy(self.b_i_j_p)
        self.TC_i_p_p = np.zeros((30, 10, 10))
        self.TC_i_p_p[0, :, :] = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                           [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                                           [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                                           [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                                           [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                           [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                                           [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                                           [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
        for i in range(1, 30):
            self.TC_i_p_p[i, :, :] = self.TC_i_p_p[0, :, :] * self.U_i_p[0, i]

        self.H2C_ip = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 4, 0.5, 4, 4, 4, 4, 4, 0.2, 0.5, 4, 0.5, 4, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1, 3, 1, 3, 4, 4, 4, 4, 0.2, 1, 3, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1, 4, 1, 4, 3, 3, 4, 4, 0.2, 1, 4, 1, 4, 3, 3, 4, 4, 4, 4, 4, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1, 4, 1, 4, 4, 4, 3, 3, 0.2, 1, 4, 1, 4, 4, 4, 3, 3, 4, 4, 4, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 3, 3, 4, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 3, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 3],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 0.2, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.7, 4, 0.7, 3.5, 3.5, 3.5, 3.5, 3.5, 0.2, 0.7, 4, 0.7, 3.5, 3.5, 3.5, 3.5,
              3.5, 3.5, 3.5, 3.5, 3.5]
             ])
        self.H2C_ip = self.H2C_ip * 3
        self.MLS_ipt = np.array(
            [[8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [8, 5, 3, 2, 8, 3, 2, 5, 3, 8, 5, 5, 5, 5, 5, 5, 20, 10, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 4, 3, 2, 2, 2, 4, 5, 5, 5, 5, 5, 5, 20, 5, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.PMi = np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # self.Ori_ip = np.zeros((10, 30))
        self.Ori_ip = np.array([[200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                                [200, 200, 200, 200, 200, 50, 50, 50, 30, 20, 50, 50, 50, 50, 50, 50, 100, 30, 20, 30,
                                 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]])
        self.Ori_ip = self.Ori_ip * 0.2

if __name__ == "__main__":
    file_path = "D:\\00_Work\\00_CityU\\04_AEL_MEC\\test_code\\a_amazing.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        function_code = file.read()
    mecenv = MLSENV()
    ss1= time.time()
    whatit = mecenv.evaluate_solution(function_code)
    print(whatit)
    returnit = mecenv.instances[0].objfunction_final(whatit[0])
    print(returnit)
    print(mecenv.evaluate(function_code))
    whyit = mecenv.instances[0].why_fail(whatit[0])
    print()
    print('time consumption', time.time()-ss1)

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
