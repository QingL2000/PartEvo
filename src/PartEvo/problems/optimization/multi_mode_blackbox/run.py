
import sys
import types
import warnings
import copy
from joblib import Parallel, delayed
import time

from .prompts import GetPrompts

import numpy as np

class baseline_instance():
    def __init__(self, obj_index, dim):
        """
        根据输入为self.objfunction填装目标函数，注意目标函数需要有对应的上下限之类的。
        :param obj_index: 选择目标函数的代号
        :param dim: 向量的维度
        """
        self.dim = dim  # 向量维度
        self.pop_size = 30  # 种群大小

        # 根据目标函数代号选择目标函数，并设置上下限
        if obj_index == 1:
            self.objfunction = self.modified_rastrigin
            self.lower = np.full(self.dim, -5.12)
            self.upper = np.full(self.dim, 5.12)
            self.optimal_value = 100  # 最优目标函数值
            self.optimal_solution = np.full(self.dim, 2)  # 对应的最优解
        elif obj_index == 2:
            self.objfunction = self.modified_ackley
            self.lower = np.full(self.dim, -32.768)
            self.upper = np.full(self.dim, 32.768)
            self.optimal_value = 50  # 最优目标函数值
            self.optimal_solution = np.full(self.dim, -10)  # 对应的最优解
        elif obj_index == 3:
            self.objfunction = self.modified_griewank
            self.lower = np.full(self.dim, -600)
            self.upper = np.full(self.dim, 600)
            self.optimal_value = 200  # 最优目标函数值
            self.optimal_solution = np.full(self.dim, 50)  # 对应的最优解
        elif obj_index == 4:
            self.objfunction = self.modified_levy
            self.lower = np.full(self.dim, -10)
            self.upper = np.full(self.dim, 10)
            self.optimal_value = 500  # 最优目标函数值
            self.optimal_solution = np.full(self.dim, 5)  # 对应的最优解
        elif obj_index == 5:
            self.objfunction = self.modified_schaffer_n2
            self.lower = np.full(self.dim, -100)
            self.upper = np.full(self.dim, 100)
            self.optimal_value = 300  # 最优目标函数值
            self.optimal_solution = np.array([50, -50])  # 对应的最优解 (二维问题)
        elif obj_index == 6:
            self.objfunction = self.modified_rosenbrock
            self.lower = np.full(self.dim, -5)
            self.upper = np.full(self.dim, 5)
            self.optimal_value = 0  # 最优目标函数值
            self.optimal_solution = np.full(self.dim, 1)  # 对应的最优解
        elif obj_index == 7:
            self.objfunction = self.modified_weierstrass
            self.lower = np.full(self.dim, -0.5)
            self.upper = np.full(self.dim, 0.5)
            self.optimal_value = 10  # 最优目标函数值
            self.optimal_solution = np.full(self.dim, 0)  # 对应的最优解
        elif obj_index == 8:
            self.objfunction = self.modified_alpine
            self.lower = np.full(self.dim, -10)
            self.upper = np.full(self.dim, 10)
            self.optimal_value = 0  # 最优目标函数值
            self.optimal_solution = np.full(self.dim, 0)  # 对应的最优解
        elif obj_index == 9:
            self.objfunction = self.modified_michalewicz
            self.lower = np.full(self.dim, 0)
            self.upper = np.full(self.dim, np.pi)
            self.optimal_value = -1  # 最优目标函数值 (具体值依赖维度)
            self.optimal_solution = None  # 最优解未知
        elif obj_index == 10:
            self.objfunction = self.modified_schwefel
            self.lower = np.full(self.dim, -500)
            self.upper = np.full(self.dim, 500)
            self.optimal_value = 0  # 最优目标函数值
            self.optimal_solution = np.full(self.dim, 420.9687)  # 对应的最优解

        self.inited_positions = self.init_pop()

    def modified_rastrigin(self, solution):
        """
        Modified Rastrigin Function (修改后的拉斯特里金函数)
        最优目标函数值: 100
        最优解: [2, 2, ..., 2]
        """
        A = 10
        shifted_solution = solution - 2  # 平移到以 2 为中心
        return A * self.dim + np.sum(np.square(shifted_solution) - A * np.cos(2 * np.pi * shifted_solution)) + 100

    def modified_ackley(self, solution):
        """
        Modified Ackley Function (修改后的阿克雷函数)
        最优目标函数值: 50
        最优解: [-10, -10, ..., -10]
        """
        shifted_solution = solution + 10  # 平移到以 -10 为中心
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(shifted_solution)) / self.dim))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * shifted_solution)) / self.dim)
        return term1 + term2 + 20 + np.e + 50  # 确保值始终非负

    def modified_griewank(self, solution):
        """
        Modified Griewank Function (修改后的格里温克函数)
        最优目标函数值: 200
        最优解: [50, 50, ..., 50]
        """
        shifted_solution = solution - 50  # 平移到以 50 为中心
        sum_term = np.sum(np.square(shifted_solution)) / 4000
        prod_term = np.prod(np.cos(shifted_solution / np.sqrt(np.arange(1, self.dim + 1))))
        return 1 + sum_term - prod_term + 200

    def modified_levy(self, solution):
        """
        Modified Levy Function (修改后的莱维函数)
        最优目标函数值: 500
        最优解: [5, 5, ..., 5]
        """
        shifted_solution = solution - 5  # 平移到以 5 为中心
        w = 1 + (shifted_solution - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return term1 + term2 + term3 + 500

    def modified_schaffer_n2(self, solution):
        """
        Modified Schaffer Function N.2 (修改后的谢弗函数N.2)
        最优目标函数值: 300
        最优解: [50, -50]
        """
        x, y = solution[0] - 50, solution[1] + 50  # 平移到以 (50, -50) 为中心
        return 0.5 + (np.sin(x**2 - y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2 + 300

    def modified_rosenbrock(self, solution):
        shifted_solution = solution - 1  # 平移到以 1 为中心
        return np.sum(100 * (shifted_solution[1:] - shifted_solution[:-1] ** 2) ** 2 + (shifted_solution[:-1] - 1) ** 2)

    def modified_weierstrass(self, solution):
        a, b, k_max = 0.5, 3, 20
        shifted_solution = solution  # 不做平移
        term1 = np.sum([np.sum(a ** k * np.cos(2 * np.pi * b ** k * (shifted_solution + 0.5))) for k in range(k_max)])
        term2 = self.dim * np.sum([a ** k * np.cos(2 * np.pi * b ** k * 0.5) for k in range(k_max)])
        return term1 - term2 + 10

    def modified_alpine(self, solution):
        shifted_solution = solution  # 不做平移
        return np.sum(np.abs(shifted_solution * np.sin(shifted_solution) + 0.1 * shifted_solution))

    def modified_michalewicz(self, solution):
        m = 10
        shifted_solution = solution  # 不做平移
        return -np.sum(
            np.sin(shifted_solution) * np.sin((np.arange(1, self.dim + 1) * shifted_solution ** 2) / np.pi) ** (2 * m))

    def modified_schwefel(self, solution):
        shifted_solution = solution  # 不做平移
        return 418.9829 * self.dim - np.sum(shifted_solution * np.sin(np.sqrt(np.abs(shifted_solution))))

    def init_pop(self):
        # 使用均匀分布随机初始化种群
        return np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))

class Baseline_multi:
    def __init__(self, test_mode=False):
        self.prompts = GetPrompts()
        self.can_visualize = False
        self.taskname='multi_peak'

        if test_mode:
            objinds = [2, 5,6, 7, 8,9]  # 选择目标函数代号  # 9是负的
            self.instance_num = len(objinds)
            dim = 40  # 向量维度，假设是5维，你可以根据需要调整
            self.instances = [baseline_instance(objinds[index], dim) for index in range(self.instance_num)]
        else:
            objinds = [1, 3, 4]  # 选择目标函数代号
            self.instance_num = len(objinds)
            dim = 50  # 向量维度，假设是5维，你可以根据需要调整
            self.instances = [baseline_instance(objinds[index], dim) for index in range(self.instance_num)]

        def compute_human_grade(instance):
            return instance.objfunction(self.human_design_algo(copy.deepcopy(instance.inited_positions), instance.upper, instance.lower,
                                       instance.objfunction))

        s1 = time.time()
        # 每个instance运行两次，保存结果
        results = Parallel(n_jobs=8)(
            delayed(compute_human_grade)(instance) for instance in self.instances for _ in range(1))


        # 输出结果
        print('Time consumption:', time.time() - s1)
        print('#### Human grades are ', results)
        print('#### Human grades are ', np.sum(results))

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
    file_path = "D:\\00_Work\\00_CityU\\04_AEL_MEC\\test_code\\algocode_gpt.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        function_code = file.read()
    mecenv = Baseline_multi()
    print(mecenv.evaluate(function_code))


