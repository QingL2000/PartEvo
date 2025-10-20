import numpy as np
import pickle
import sys
import types
import warnings
import copy
from joblib import Parallel, delayed
import concurrent.futures
from .prompts import GetPrompts
import time

class baseline_instance():
    def __init__(self, obj_index, dim):
        """
        根据输入为self.objective填装目标函数，注意目标函数需要有对应的上下限之类的，你都可以自己定
        :param obj_index: 选择目标函数的代号
        :param dim: 向量的维度
        """
        self.dim = dim  # 向量维度
        self.pop_size = 30  # 种群大小

        # 根据目标函数代号选择目标函数，并设置上下限
        if obj_index == 1:
            self.objfunction = self.f1
            self.lower = np.full(self.dim, -20)  # 对应的上下限
            self.upper = np.full(self.dim, 20)
        elif obj_index == 2:
            self.objfunction = self.f2
            self.lower = np.full(self.dim, -20)  # 对应的上下限
            self.upper = np.full(self.dim, 20)
        elif obj_index == 3:
            self.objfunction = self.f3
            self.lower = np.full(self.dim, -20)  # 对应的上下限
            self.upper = np.full(self.dim, 20)
        elif obj_index == 4:
            self.objfunction = self.f4
            self.lower = np.full(self.dim, -10)  # 对应的上下限
            self.upper = np.full(self.dim, 10)
        elif obj_index == 5:
            self.objfunction = self.f5
            self.lower = np.full(self.dim, -10)  # 对应的上下限
            self.upper = np.full(self.dim, 10)
        elif obj_index == 6:
            self.objfunction = self.f6
            self.lower = np.full(self.dim, -10)  # 对应的上下限
            self.upper = np.full(self.dim, 10)
        elif obj_index == 7:
            self.objfunction = self.f7
            self.lower = np.full(self.dim, -5)  # 对应的上下限
            self.upper = np.full(self.dim, 5)
        elif obj_index == 8:
            self.objfunction = self.f8
            self.lower = np.full(self.dim, -30)  # 对应的上下限
            self.upper = np.full(self.dim, 30)
        elif obj_index == 9:
            self.objfunction = self.f9
            self.lower = np.full(self.dim, -100)  # 对应的上下限
            self.upper = np.full(self.dim, 100)
        elif obj_index == 10:
            self.objfunction = self.f10
            self.lower = np.full(self.dim, -100)  # 对应的上下限
            self.upper = np.full(self.dim, 100)

        self.inited_positions = self.init_pop()

    def f1(self, solution):
        # Sphere Function (球形函数)
        return np.sum(np.square(solution))  # 计算每个维度的平方和

    def f2(self, solution):
        # Shifted Sphere Function (平移后的球形函数)
        # 通过平移原始球形函数，增加优化问题的难度
        shift = np.ones(self.dim) * 2.5  # 平移的常数向量
        return np.sum(np.square(solution - shift))  # 计算每个维度的平方和，平移后的优化问题

    def f3(self, solution):
        # Rosenbrock Function (罗森布鲁克函数)
        # 该函数是一个典型的单峰函数，具有一个全球最小值
        # 目标函数最小值位于 (1,1,...,1) 处
        return np.sum(100.0 * np.square(solution[1:] - solution[:-1] ** 2) + (1 - solution[:-1]) ** 2)

    def f4(self, solution):
        # Shifted Ackley Function (平移后的Ackley函数)
        # 通过平移原始的Ackley函数来增加优化问题的难度
        shift = np.ones(self.dim) * 2.5  # 平移的常数向量
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(solution - shift)) / self.dim))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * (solution - shift))) / self.dim)
        return term1 + term2 + 20 + np.e

    def f5(self, solution):
        # Shifted Sphere Function (平移后的球形函数)
        shift = np.ones(self.dim) * 3.0  # 平移的常数向量
        return np.sum(np.square(solution - shift))  # 计算每个维度的平方和，平移后的优化问题

    def f6(self, solution):
        # Unimodal Quadratic Function (单峰二次函数)
        return np.sum(np.square(solution - 5))  # 计算平方和，偏移值为5，单峰

    def f7(self, solution):
        # Unimodal Sinusoidal Function (单峰正弦函数)
        return np.sum(np.sin(solution) ** 2)  # 正弦的平方，确保只有一个峰

    def f8(self, solution):
        # Unimodal Gaussian Function (单峰高斯函数)
        return np.sum(np.exp(-np.square(solution - 1)))  # 高斯分布，单峰在solution=1

    def f9(self, solution):
        # Unimodal Logistic Function (单峰逻辑函数)
        return np.sum(1 / (1 + np.exp(-solution)))  # 逻辑函数，单峰

    def f10(self, solution):
        # Unimodal Exponential Function (单峰指数衰减函数)
        return np.sum(np.exp(-solution))  # 指数衰减，单峰

    def init_pop(self):
        # 使用均匀分布随机初始化种群
        return np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))


class Baseline:
    def __init__(self, test_mode=False):
        self.prompts = GetPrompts()
        self.can_visualize = False
        self.taskname = 'single_peak'
        if test_mode:
            objinds = [2, 4, 6, 7, 8, 9, 10]  # 选择目标函数代号
            self.instance_num = len(objinds)
            dim = 20  # 向量维度，假设是5维，你可以根据需要调整
            self.instances = [baseline_instance(objinds[index], dim) for index in range(self.instance_num)]
        else:
            objinds = [1, 3, 5]  # 选择目标函数代号
            self.instance_num = len(objinds)
            dim = 20  # 向量维度，假设是5维，你可以根据需要调整
            self.instances = [baseline_instance(objinds[index], dim) for index in range(self.instance_num)]


        def compute_human_grade(instance):
            return instance.objfunction(
                self.human_design_algo(copy.deepcopy(instance.inited_positions), instance.upper, instance.lower,
                                       instance.objfunction))

        s1 = time.time()
        # 每个instance运行两次，保存结果
        results = Parallel(n_jobs=8)(
            delayed(compute_human_grade)(instance) for instance in self.instances for _ in range(1))

        # # 计算每个instance的最小值和平均值
        # min_grades = []
        # average_grades = []
        #
        # # 对于每个instance，取出两次运行的结果
        # for i, instance in enumerate(self.instances):
        #     instance_results = results[i * 2:i * 2 + 2]  # 取出该instance的两次结果
        #     min_grades.append(min(instance_results))  # 取最小值
        #     average_grades.append(sum(instance_results) / len(instance_results))  # 取平均值
        #
        # # 计算最终的min_grade和average_grade
        # min_grade = sum(min_grades)
        # average_grade = sum(average_grades)

        # 输出结果
        print('Time consumption:', time.time() - s1)
        print('#### Human grades are ', results)
        print('#### Human grades are ', np.sum(results))
        # print('#### Minimum human grade is ', min_grade)
        # print('#### Average human grade is ', average_grade)

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
                    print(f'Instance {instance_idx} finished, spend {time.time()-ss1} sec.')
                    return instance.objfunction(final_solution)

                # Now you can use the module as you would any other
                # 使用Parallel并行化for循环
                fitnesses = [compute_fitness(instance_idx, instance) for instance_idx, instance in enumerate(self.instances)]

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

from .cec2017 import functions

class baseline_instancecec():
    def __init__(self, obj_index, dim):
        """
        根据输入为self.objective填装目标函数，注意目标函数需要有对应的上下限之类的，你都可以自己定
        :param obj_index: 选择目标函数的代号
        :param dim: 向量的维度
        """
        self.dim = dim  # 向量维度
        self.pop_size = 30  # 种群大小

        # 根据目标函数代号选择目标函数，并设置上下限
        if obj_index == 1:
            self.objfunction = self.f1
            self.lower = np.full(self.dim, -100)  # 对应的上下限
            self.upper = np.full(self.dim, 100)
        elif obj_index == 3:
            self.objfunction = self.f3
            self.lower = np.full(self.dim, -100)  # 对应的上下限
            self.upper = np.full(self.dim, 100)

        self.inited_positions = self.init_pop()

    def f1(self, solution):
        # Sphere Function (球形函数)
        f = functions.f5
        result = f([solution])[0]
        return result

    def f2(self, solution):
        return functions.f3([solution])[0]

    def f3(self, solution):
        # Rosenbrock Function (罗森布鲁克函数)
        # 该函数是一个典型的单峰函数，具有一个全球最小值
        # 目标函数最小值位于 (1,1,...,1) 处
        return np.sum(100.0 * np.square(solution[1:] - solution[:-1] ** 2) + (1 - solution[:-1]) ** 2)

    def f4(self, solution):
        # Shifted Ackley Function (平移后的Ackley函数)
        # 通过平移原始的Ackley函数来增加优化问题的难度
        shift = np.ones(self.dim) * 2.5  # 平移的常数向量
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(solution - shift)) / self.dim))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * (solution - shift))) / self.dim)
        return term1 + term2 + 20 + np.e

    def f5(self, solution):
        # Shifted Sphere Function (平移后的球形函数)
        shift = np.ones(self.dim) * 3.0  # 平移的常数向量
        return np.sum(np.square(solution - shift))  # 计算每个维度的平方和，平移后的优化问题

    def f6(self, solution):
        # Unimodal Quadratic Function (单峰二次函数)
        return np.sum(np.square(solution - 5))  # 计算平方和，偏移值为5，单峰

    def f7(self, solution):
        # Unimodal Sinusoidal Function (单峰正弦函数)
        return np.sum(np.sin(solution) ** 2)  # 正弦的平方，确保只有一个峰

    def f8(self, solution):
        # Unimodal Gaussian Function (单峰高斯函数)
        return np.sum(np.exp(-np.square(solution - 1)))  # 高斯分布，单峰在solution=1

    def f9(self, solution):
        # Unimodal Logistic Function (单峰逻辑函数)
        return np.sum(1 / (1 + np.exp(-solution)))  # 逻辑函数，单峰

    def f10(self, solution):
        # Unimodal Exponential Function (单峰指数衰减函数)
        return np.sum(np.exp(-solution))  # 指数衰减，单峰

    def init_pop(self):
        # 使用均匀分布随机初始化种群
        return np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))


class BaselineCEC:
    def __init__(self, test_mode=False):
        self.prompts = GetPrompts()
        self.can_visualize = False
        self.taskname = 'single_peak'
        if test_mode:
            objinds = [1, 3]  # 选择目标函数代号
            self.instance_num = len(objinds)
            dim = 30  # 向量维度，假设是5维，你可以根据需要调整
            self.instances = [baseline_instancecec(objinds[index], dim) for index in range(self.instance_num)]
        else:
            objinds = [1, 3, 5]  # 选择目标函数代号
            self.instance_num = len(objinds)
            dim = 20  # 向量维度，假设是5维，你可以根据需要调整
            self.instances = [baseline_instance(objinds[index], dim) for index in range(self.instance_num)]

        # # 计算每个instance的最小值和平均值
        # min_grades = []
        # average_grades = []
        #
        # # 对于每个instance，取出两次运行的结果
        # for i, instance in enumerate(self.instances):
        #     instance_results = results[i * 2:i * 2 + 2]  # 取出该instance的两次结果
        #     min_grades.append(min(instance_results))  # 取最小值
        #     average_grades.append(sum(instance_results) / len(instance_results))  # 取平均值
        #
        # # 计算最终的min_grade和average_grade
        # min_grade = sum(min_grades)
        # average_grade = sum(average_grades)


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
                    print(f'Instance {instance_idx} finished, spend {time.time()-ss1} sec.')
                    return instance.objfunction(final_solution)

                # Now you can use the module as you would any other
                # 使用Parallel并行化for循环
                fitnesses = [compute_fitness(instance_idx, instance) for instance_idx, instance in enumerate(self.instances)]

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

if __name__ == "__main__":
    # file_path = "D:\\00_Work\\00_CityU\\04_AEL_MEC\\test_code\\algocode_gpt.txt"
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     function_code = file.read()
    # mecenv = Baseline()
    # print(mecenv.evaluate(function_code))
    f = functions.f5
    dimension = 30
    for i in range(0, 10):
        x = np.random.uniform(low=-100, high=100, size=dimension)
        y = f([x])[0]
        print(f"f5({x[0]:.2f}, {x[1]:.2f}, ...) = {y:.2f}")
