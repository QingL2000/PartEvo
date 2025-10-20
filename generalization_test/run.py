import numpy as np
import pickle
import sys
import types
import warnings
import copy
from joblib import Parallel, delayed
import concurrent.futures
import time

import numpy as np
from scipy.stats import ortho_group


class baseline_instance_generation():
    def __init__(self, obj_index, dim):
        """
        根据输入为self.objective填装目标函数，注意目标函数需要有对应的上下限之类的，你都可以自己定
        :param obj_index: 选择目标函数的代号(1-20)
        :param dim: 向量的维度
        """
        self.dim = dim  # 向量维度
        self.pop_size = 30  # 种群大小

        # 初始化随机偏移量和旋转矩阵（固定随机种子保证可复现）
        self.obj_index = obj_index
        np.random.seed(42)
        self.shifts = [np.random.uniform(-5, 5, dim) for _ in range(21)]
        self.rotation_matrices = [ortho_group.rvs(dim) for _ in range(21)]

        # 根据目标函数代号选择目标函数，并设置上下限
        if obj_index == 1:
            self.objfunction = self.f1
            self.lower = np.full(self.dim, -20)
            self.upper = np.full(self.dim, 20)
        elif obj_index == 2:
            self.objfunction = self.f2
            self.lower = np.full(self.dim, -20)
            self.upper = np.full(self.dim, 20)
        elif obj_index == 3:
            self.objfunction = self.f3
            self.lower = np.full(self.dim, -20)
            self.upper = np.full(self.dim, 20)
        elif obj_index == 4:
            self.objfunction = self.f4
            self.lower = np.full(self.dim, -10)
            self.upper = np.full(self.dim, 10)
        elif obj_index == 5:
            self.objfunction = self.f5
            self.lower = np.full(self.dim, -10)
            self.upper = np.full(self.dim, 10)
        elif obj_index == 6:
            self.objfunction = self.f6
            self.lower = np.full(self.dim, -10)
            self.upper = np.full(self.dim, 10)
        elif obj_index == 7:
            self.objfunction = self.f7
            self.lower = np.full(self.dim, -5)
            self.upper = np.full(self.dim, 5)
        elif obj_index == 8:
            self.objfunction = self.f8
            self.lower = np.full(self.dim, -30)
            self.upper = np.full(self.dim, 30)
        elif obj_index == 9:
            self.objfunction = self.f9
            self.lower = np.full(self.dim, -100)
            self.upper = np.full(self.dim, 100)
        elif obj_index == 10:
            self.objfunction = self.f10
            self.lower = np.full(self.dim, -100)
            self.upper = np.full(self.dim, 100)
        elif obj_index == 11:
            self.objfunction = self.f11
            self.lower = np.full(self.dim, -50)
            self.upper = np.full(self.dim, 50)
        elif obj_index == 12:
            self.objfunction = self.f12
            self.lower = np.full(self.dim, -20)
            self.upper = np.full(self.dim, 20)
        elif obj_index == 13:
            self.objfunction = self.f13
            self.lower = np.full(self.dim, -100)
            self.upper = np.full(self.dim, 100)
        elif obj_index == 14:
            self.objfunction = self.f14
            self.lower = np.full(self.dim, -50)
            self.upper = np.full(self.dim, 50)
        elif obj_index == 15:
            self.objfunction = self.f15
            self.lower = np.full(self.dim, -10)
            self.upper = np.full(self.dim, 10)
        elif obj_index == 16:
            self.objfunction = self.f16
            self.lower = np.full(self.dim, -5)
            self.upper = np.full(self.dim, 5)
        elif obj_index == 17:
            self.objfunction = self.f17
            self.lower = np.full(self.dim, -5)
            self.upper = np.full(self.dim, 5)
        elif obj_index == 18:
            self.objfunction = self.f18
            self.lower = np.full(self.dim, -100)
            self.upper = np.full(self.dim, 100)
        elif obj_index == 19:
            self.objfunction = self.f19
            self.lower = np.full(self.dim, -10)
            self.upper = np.full(self.dim, 10)
        elif obj_index == 20:
            self.objfunction = self.f20
            self.lower = np.full(self.dim, -10)
            self.upper = np.full(self.dim, 10)
        else:
            raise ValueError("obj_index must be between 1 and 20")

        self.inited_positions = self.init_pop()

    # --- 目标函数实现 ---
    def f1(self, solution):
        # Shifted Elliptic Function
        shift = self.shifts[1]
        weights = np.array([10 ** (6 * (i / (self.dim - 1))) for i in range(self.dim)])
        return np.sum(weights * np.square(solution - shift))

    def f2(self, solution):
        # Shifted Elliptic Function
        shift = self.shifts[2]
        weights = np.array([10 ** (6 * (i / (self.dim - 1))) for i in range(self.dim)])
        return np.sum(weights * np.square(solution - shift))

    def f3(self, solution):
        # Shifted Elliptic Function
        shift = self.shifts[3]
        weights = np.array([10 ** (6 * (i / (self.dim - 1))) for i in range(self.dim)])
        return np.sum(weights * np.square(solution - shift))

    def f4(self, solution):
        # Shifted Elliptic Function
        shift = self.shifts[4]
        weights = np.array([10 ** (6 * (i / (self.dim - 1))) for i in range(self.dim)])
        return np.sum(weights * np.square(solution - shift))

    def f5(self, solution):
        # Rotated Shifted Sphere
        R = self.rotation_matrices[5]
        shift = self.shifts[5]
        rotated_shifted_x = R.dot(solution - shift)
        return np.sum(np.square(rotated_shifted_x))

    def f6(self, solution):
        # Shifted Sum of Powers
        shift = self.shifts[6]
        return sum(abs(x - s) ** (i + 2) for i, (x, s) in enumerate(zip(solution, shift)))

    def f7(self, solution):
        # Shifted Exponential Sum (修正后)
        shift = self.shifts[7]
        return np.sum(np.exp(abs(solution - shift)) - np.sum(np.exp(0)))  # 最优时=0

    def f8(self, solution):
        # Shifted Logarithmic Valley
        shift = self.shifts[8]
        return np.sum(np.log(1 + np.square(solution - shift)))

    def f9(self, solution):
        # Shifted Absolute Penalized
        shift = self.shifts[9]
        diff = solution - shift
        return np.sum(abs(diff) + 0.5 * np.sin(10 * abs(diff)))

    def f10(self, solution):
        # Shifted Absolute Penalized
        shift = self.shifts[10]
        diff = solution - shift
        return np.sum(abs(diff) + 0.5 * np.sin(10 * abs(diff)))

    def f11(self, solution):
        # Randomly Scaled Shifted Sphere
        shift = self.shifts[11]
        scales = np.random.RandomState(42).uniform(1, 100, self.dim)
        return np.sum(scales * np.square(solution - shift))

    def f12(self, solution):
        # Linear + Quadratic Combo
        shift = self.shifts[12]
        diff = solution - shift
        return np.sum(diff[:self.dim // 2] ** 2) + np.sum(abs(diff[self.dim // 2:]))

    def f13(self, solution):
        # Ill-Conditioned Shifted Sum
        shift = self.shifts[13]
        weights = np.array([10 ** (8 * (i / (self.dim - 1))) for i in range(self.dim)])
        return np.sum(weights * np.square(solution - shift))

    def f14(self, solution):
        # Exponential Decay Shifted Valley
        shift = self.shifts[14]
        weights = np.array([np.exp(i / self.dim) - 1 for i in range(self.dim)])
        return np.sum(weights * np.square(solution - shift))

    def f15(self, solution):
        # Noisy Shifted Sphere
        shift = self.shifts[15]
        noise = np.random.RandomState(42).normal(0, 0.01, self.dim)
        return np.sum(np.square(solution - shift) * (1 + noise))

    def f16(self, solution):
        # Quantized Shifted Sphere
        shift = self.shifts[16]
        return np.sum(np.floor(np.square(solution - shift)))

    def f17(self, solution):
        # Zero-Slope Plateau Function
        shift = self.shifts[17]
        return np.sum(np.where(np.abs(solution - shift) < 1e-10, 0, 1))

    def f18(self, solution):
        # Shifted Schwefel's Problem 1.2
        shift = self.shifts[18]
        diff = solution - shift
        return np.sum([np.sum(diff[:i + 1]) ** 2 for i in range(self.dim)])

    def f19(self, solution):
        # Shifted Sum of Powers
        shift = self.shifts[19]
        return sum(abs(x - s) ** (i + 2) for i, (x, s) in enumerate(zip(solution, shift)))

    def f20(self, solution):
        # Exponential Decay Shifted Valley
        shift = self.shifts[20]
        weights = np.array([np.exp(i / self.dim) - 1 for i in range(self.dim)])
        return np.sum(weights * np.square(solution - shift))

    def init_pop(self):
        """使用均匀分布随机初始化种群"""
        return np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))

    def get_optimal_solution(self):
        """获取当前目标函数的最优解"""
        if hasattr(self, 'objfunction'):
            solution = self.shifts[self.obj_index]
            return self.objfunction(solution), solution
        return None

class Baseline_for_generationtest:
    def __init__(self, test_mode=False):
        self.can_visualize = False
        self.taskname = 'single_peak'
        if test_mode:
            objinds = [i for i in range(1, 21)]
            self.instance_num = len(objinds)
            dim = 30  # 向量维度，假设是5维，你可以根据需要调整
            self.instances = [baseline_instance_generation(objinds[index], dim) for index in range(self.instance_num)]
        else:
            objinds = [1, 3, 5]  # 选择目标函数代号
            self.instance_num = len(objinds)
            dim = 20  # 向量维度，假设是5维，你可以根据需要调整
            self.instances = [baseline_instance_generation(objinds[index], dim) for index in range(self.instance_num)]


        # def compute_human_grade(instance):
        #     return instance.objfunction(
        #         self.human_design_algo(copy.deepcopy(instance.inited_positions), instance.upper, instance.lower,
        #                                instance.objfunction))
        #
        # s1 = time.time()
        # # 每个instance运行两次，保存结果
        # results = Parallel(n_jobs=8)(
        #     delayed(compute_human_grade)(instance) for instance in self.instances for _ in range(1))
        #
        # # # 计算每个instance的最小值和平均值
        # # min_grades = []
        # # average_grades = []
        # #
        # # # 对于每个instance，取出两次运行的结果
        # # for i, instance in enumerate(self.instances):
        # #     instance_results = results[i * 2:i * 2 + 2]  # 取出该instance的两次结果
        # #     min_grades.append(min(instance_results))  # 取最小值
        # #     average_grades.append(sum(instance_results) / len(instance_results))  # 取平均值
        # #
        # # # 计算最终的min_grade和average_grade
        # # min_grade = sum(min_grades)
        # # average_grade = sum(average_grades)
        #
        # # 输出结果
        # print('Time consumption:', time.time() - s1)
        # print('#### Human grades are ', results)
        # print('#### Human grades are ', np.sum(results))
        # # print('#### Minimum human grade is ', min_grade)
        # # print('#### Average human grade is ', average_grade)

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

    for i in range(1, 21):
        instances = baseline_instance_generation(i, 20)
        print(i, instances.get_optimal_solution())
