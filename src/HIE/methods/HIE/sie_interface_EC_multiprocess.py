import numpy as np
import time
import os
import json  # 用于记录数据到 JSON 文件中
import math

from .sie_evolution import Evolution  # 这里面进行实际的初始化、交叉、变异等过程

import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import sys
import random
from .individual_cluster import Individual
import copy
import time
import traceback
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import TimeoutError
from functools import partial


class InterfaceEC():
    """
    管理Evolution的交互类
    功能包括：
    1. code写成py文件  --  code2file(self,code)
    2. offspring加入到population中 --  add2pop(self,population,offspring)
    3. 检查代码是否重复 -- check_duplicate(self,population,code)
    4. 迭代的population生成 -- population_generation(self)  **初代种群生成
        内部调用 self.get_algorithm([], 'i1')

    5. 将已有的seed中的individual进行评估并放入种群 -- population_generation_seed(self,seeds,n_p)
    6. 根据operator对pop进行操作从而获得新的algorithm，返回parents和offspring -- _get_alg(self,pop,operator)
        内部与ael_evolution 中的Evolution进行交互

    7. 根据当前population和需要进行的operator生成offspring -- get_offspring(self, pop, operator)
        内部调用 _get_alg(self,pop,operator)

    8. 最外界调用该类的接口，用于生成算法 -- get_algorithm(self, pop, operator)
        内部并行调用 self.get_offspring

    最终包含关系为：
    get_algorithm(self, pop, operator) 并行调用 get_offspring 调用 _get_alg 调用 Evolution中的i1、crossover和mutation

    """

    def __init__(self, pop_size, api_endpoint, api_endpoint_url, api_key, llm_model, debug_mode, interface_prob, k, m,
                 n_p, timeout, use_numba, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------
        self.extra_params = kwargs
        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_endpoint_url, api_key, llm_model, debug_mode, prompts, **kwargs)
        self.m = m  # 每个cluster中有m个有机会
        self.k = k  # cc时的cluster个数
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        # self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba

        self.init_times_limit = self.extra_params.get('init_times_limit', 2 * self.pop_size)

        self.best_instruct_prob = self.extra_params.get('besta_instruct_prob', 1)
        self.local_instruct_prob = self.extra_params.get('locala_instruct_prob', 1)

        self.save_path = kwargs.get('logsave_path')
        self.iterative_init = kwargs.get('iterative_init')
        self.sample_times = 0
        self.fail_times = 0
        # 确保保存路径存在
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # 记录的文件路径
        self.log_file_path = os.path.join(self.save_path, "sample_log.json")
        self.log_best_file_path = os.path.join(self.save_path, "best_sample_log.json")

    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
            # Write the code to the file
            file.write(code)
        return

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate(self, population, code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation(self):
        population = []
        create_epoch = 0

        # 定义生成单个个体的并行任务
        def generate_individual(create_epoch, current_pop):
            try:
                print('Initialization request, ', create_epoch)
                indibyllm = self.get_algorithm_single(operator='init', current_pop=current_pop)
                if indibyllm['objective']:
                    individual = Individual()
                    individual.create_individual(ind_in_dict=indibyllm)
                    return individual
                return None
            except Exception as e:
                print(f"Error in generate_individual at epoch {create_epoch}: {e}")
                return None  # 如果出错，返回 None，确保不会中断其他任务

        # 使用并行化生成个体
        while len(population) < self.pop_size and create_epoch < self.init_times_limit:
            # 每次生成一批个体，最大为每次并行生成的个体数
            batch_size = 3  # 每次生成3个个体，可以根据实际需求调整
            s1 = time.time()
            try:
                if self.iterative_init:
                    results = Parallel(n_jobs=min(batch_size, 3), timeout=self.timeout)(
                        delayed(generate_individual)(create_epoch + i, population)
                        for i in range(batch_size)
                    )
                else:
                    results = Parallel(n_jobs=self.n_p, timeout=self.timeout)(
                        delayed(generate_individual)(create_epoch + i, [])
                        for i in range(self.n_p)
                    )
            except Exception as e:
                print(f'Error in parallel execution: {e}')
                results = []  # 如果发生异常，确保 results 是一个空列表，不影响后续代码
            if self.iterative_init:
                print(f'This {batch_size} samples used {time.time() - s1} seconds')
            else:
                print(f'This {self.n_p} samples used {time.time() - s1} seconds')
            # 过滤掉None值
            new_individuals = [indiv for indiv in results if indiv is not None]

            # 将新个体添加到种群中
            population.extend(new_individuals)
            create_epoch += batch_size

            # 如果种群大小超出了限制，进行裁剪
            if len(population) > self.pop_size:
                # 按照目标函数值排序，保留目标函数值较小的个体
                population.sort(key=lambda x: x.objective)  # 假设个体有一个objective属性
                excess_size = len(population) - self.pop_size
                print(f'Population size exceeded by {excess_size}, trimming...')
                # 裁剪掉目标函数值大的个体
                population = population[:-excess_size]

            # 打印当前种群状态
            print(f'Generated {len(new_individuals)} individuals, total population: {len(population)}')

        print('Initialization Done')

        if len(population) < self.pop_size:
            print(
                'Initialization Warn ---> There are no more branches available to meet the specified settings number.')

        #
        for ind in population:
            self.sample_times += 1
            record = {
                'sample_order': self.sample_times,  # 样本编号
                'algorithm': ind.algorithm,
                'code': ind.code,
                'objective': ind.objective,
            }

            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.append(record)
            else:
                data = [record]

            with open(self.log_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            if record['objective'] is not None:
                if os.path.exists(self.log_best_file_path):
                    with open(self.log_best_file_path, "r", encoding="utf-8") as f:
                        data_best = json.load(f)
                    if data_best[-1]['objective'] > record['objective']:
                        data_best.append(record)
                else:
                    data_best = [record]

                with open(self.log_best_file_path, "w", encoding="utf-8") as f:
                    json.dump(data_best, f, indent=4, ensure_ascii=False)

        return population

    def population_generation_seed(self, seeds, n_p):

        population = []

        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

        return population

    def _get_alg(self, operator, **kwargs):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "init":
            [offspring['code'], offspring['algorithm']] = self.evol.init_unit(**kwargs)
        elif operator == "re":
            [offspring['code'], offspring['algorithm']] = self.evol.independent_explore(**kwargs)
        elif operator == "cc":
            [offspring['code'], offspring['algorithm']] = self.evol.cooperation_explore(**kwargs)
        elif operator == "se":
            [offspring['code'], offspring['algorithm']] = self.evol.God_guide_explore(**kwargs)
        elif operator == "lge":
            [offspring['code'], offspring['algorithm']] = self.evol.PSO_explore(**kwargs)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        return offspring



    # def get_offspring(self, operator, **kwargs):
    #
    #     try:
    #         offspring = self._get_alg(operator, **kwargs)
    #
    #         if self.use_numba:  # numba是一个用来快速提取函数的库，但不好用，_get_alg中已经写好了直接得到code和algo
    #
    #             # Regular expression pattern to match function definitions
    #             pattern = r"def\s+(\w+)\s*\(.*\):"
    #
    #             # Search for function definitions in the code
    #             match = re.search(pattern, offspring['code'])
    #
    #             function_name = match.group(1)
    #
    #             code = add_numba_decorator(program=offspring['code'], function_name=function_name)
    #         else:
    #             code = offspring['code']
    #
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             future = executor.submit(self.interface_eval.evaluate, code)
    #
    #             try:
    #                 fitness = future.result(timeout=self.timeout)
    #                 if fitness is None:
    #                     print('Fitness equal to None, so np.round would raise error')
    #                 else:
    #                     offspring['objective'] = np.round(fitness, 8)
    #             except concurrent.futures.TimeoutError:
    #                 print("Timeout error in fitness calculation")
    #                 offspring['objective'] = None
    #
    #             future.cancel()
    #
    #     except Exception as e:
    #         print('Error at get offspring --->', e)
    #         traceback.print_exc()
    #         if 'offspring' not in locals():  # 检查是否还没有offspring存在
    #             offspring = self.default_offspring()
    #         else:
    #             offspring['objective'] = None
    #
    #     return offspring

    def get_algorithm_single(self, operator, **kwargs):

        try:
            offspring = self.get_offspring_with_evaluation(operator, **kwargs)
        except Exception as e:
            print('Error location -----> get_algorithm_single', operator)
            print('Error in get_algorithm_single:', e)
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }

        return offspring

    def get_offspring_with_evaluation(self, operator, **kwargs):
        offspring = None
        try:
            offspring = self._get_alg(operator, **kwargs)

            if self.use_numba:
                # Regular expression pattern to match function definitions
                # pattern = r"def\s+(\w+)\s*$.*$:"
                pattern = r"def\s+(\w+)\s*\("
                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])
                function_name = match.group(1)
                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code)
                try:
                    fitness = future.result(timeout=self.timeout)
                    if fitness is None:
                        print('Fitness equal to None, so np.round would raise error')
                    else:
                        offspring['objective'] = np.round(fitness, 8)
                except concurrent.futures.TimeoutError:
                    print("Timeout error in fitness calculation")
                    offspring['objective'] = None
                finally:
                    future.cancel()

        except Exception as e:
            print('Error at get offspring --->', e)
            traceback.print_exc()
            if offspring is None:  # If offspring was never initialized
                offspring = self.default_offspring()
            offspring['objective'] = None

        return offspring

    def default_offspring(self):
        return {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }

    # def evaluate_offspring_with_timeout(self, code):
    #     """
    #     Helper function to evaluate offspring with a timeout.
    #     """
    #     fitness = None
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.interface_eval.evaluate, code)
    #         try:
    #             fitness = future.result(timeout=self.timeout)
    #             if fitness is None:
    #                 print('Fitness equal to None, so np.round would raise error')
    #             else:
    #                 fitness = np.round(fitness, 8)
    #         except concurrent.futures.TimeoutError:
    #             print("Timeout error in fitness calculation")
    #             fitness = None
    #         finally:
    #             future.cancel()
    #     return fitness

    def evaluate_offspring_with_timeout(self, code):
        """
        Helper function to evaluate offspring with timeout.
        Uses the main process pool to handle timeout.
        """
        try:
            fitness = self.interface_eval.evaluate(code)
            if fitness is None:
                print(f"Fitness equal to None, due to code error.")
                return None
            return np.round(fitness, 8)
        except Exception as e:
            print(f"Error evaluating code {code}: {e}")
            return None

    def get_offspring(self, operator, parent, co_indivs, iterecneed):
        """
        Generate offspring without evaluating it.
        """
        offspring = None
        try:
            offspring = self._get_alg(operator, main_individual=parent, co_indivs=co_indivs, **iterecneed)
            offspring['objective'] = None
        except Exception as e:
            print('Error at get offspring --->', e)
            traceback.print_exc()
            if offspring is None:  # If offspring was never initialized
                offspring = self.default_offspring()
            offspring['objective'] = None
        return offspring

    # def get_algorithm(self, operator, **kwargs):
    #     """
    #     Generate offspring in parallel and evaluate them using multiprocessing.
    #     """
    #     results = []
    #     parents = kwargs.get('parent_in_clusters_ops')
    #     coor_individuals = kwargs.get('coorinds_for_each_cluster', None)
    #     flatten_parent = [item for sublist in parents[operator] for item in sublist]
    #     iterecneed = copy.deepcopy(kwargs)
    #     timeout = self.timeout
    #     if sys.gettrace() is not None:  # 检查是否在调试模式下
    #         timeout = None  # 取消超时限制
    #
    #     # Step 1: Generate offspring in parallel (without evaluating them)
    #     try:
    #         with multiprocessing.Pool(self.n_p) as pool:
    #             generate_args = [
    #                 (operator, parent, coor_individuals, iterecneed)
    #                 for parent in flatten_parent
    #             ]
    #             results = pool.starmap(self.get_offspring, generate_args)
    #     except Exception as e:
    #         print(f'EC Error in {operator}===>', e)
    #         print("Parallel time out .")
    #         traceback.print_exc()
    #
    #     # Step 2: Evaluate offspring using multiprocessing with timeout control
    #     try:
    #         codes = [res['code'] for res in results]
    #         evaluated_objectives = [None] * len(codes)  # 初始化结果列表
    #
    #         with multiprocessing.Pool(self.n_p) as pool:
    #             async_results = []
    #             # 提交任务到进程池
    #             for i, code in enumerate(codes):
    #                 async_result = pool.apply_async(self.evaluate_offspring_with_timeout, (code,))
    #                 async_results.append((i, async_result))
    #
    #             # 收集结果，带超时控制
    #             for i, async_result in async_results:
    #                 try:
    #                     # 尝试获取结果，超时则跳过
    #                     evaluated_objectives[i] = async_result.get(timeout=self.timeout)
    #                 except multiprocessing.TimeoutError:
    #                     print(f"Timeout for code at index {i}: {codes[i]}")
    #                     evaluated_objectives[i] = None
    #                 except Exception as e:
    #                     print(f"Error for code at index {i}: {codes[i]}, Error: {e}")
    #                     evaluated_objectives[i] = None
    #
    #     except Exception as e:
    #         print(f"Error during multiprocessing evaluation: {e}")
    #         traceback.print_exc()
    #         for res in results:
    #             res['objective'] = None
    #
    #     # 将评估结果回填到 results 中
    #     for res, obj in zip(results, evaluated_objectives):
    #         res['objective'] = None if obj is None else np.round(obj, 8)
    #
    #     print(f'There are {len(flatten_parent)} parent, we get {len(results)} offspring in results')
    #     for res in results:
    #         self.sample_times += 1
    #         record = {
    #             'sample_order': self.sample_times,  # 样本编号
    #             'algorithm': res['algorithm'],
    #             'code': res['code'],
    #             'objective': res['objective'],
    #         }
    #
    #         if os.path.exists(self.log_file_path):
    #             with open(self.log_file_path, "r", encoding="utf-8") as f:
    #                 data = json.load(f)
    #             data.append(record)
    #         else:
    #             data = [record]
    #
    #         with open(self.log_file_path, "w", encoding="utf-8") as f:
    #             json.dump(data, f, indent=4, ensure_ascii=False)
    #
    #         if record['objective'] is not None:
    #             if os.path.exists(self.log_best_file_path):
    #                 with open(self.log_best_file_path, "r", encoding="utf-8") as f:
    #                     data_best = json.load(f)
    #                 if data_best[-1]['objective'] >= record['objective']:
    #                     data_best.append(record)
    #             else:
    #                 data_best = [record]
    #
    #             with open(self.log_best_file_path, "w", encoding="utf-8") as f:
    #                 json.dump(data_best, f, indent=4, ensure_ascii=False)
    #
    #     clusters_new = kwargs.get('clusters')
    #     for cluster in clusters_new:
    #         cluster.clear_offspring()
    #     for resind, result in enumerate(results):
    #         if result['objective'] is not None and not math.isnan(result['objective']):
    #             clusters_new[flatten_parent[resind].whichcluster].add_individual(result)
    #             clusters_new[flatten_parent[resind].whichcluster].add_offspring(result)
    #             flatten_parent[resind].update_opresult_recorder(operator, result)

    def get_algorithm(self, operator, **kwargs):
        """
        Generate offspring in parallel and evaluate them using multiprocessing.
        """
        results = []
        parents = kwargs.get('parent_in_clusters_ops')
        coor_individuals = kwargs.get('coorinds_for_each_cluster', None)
        flatten_parent = [item for sublist in parents[operator] for item in sublist]
        iterecneed = copy.deepcopy(kwargs)
        timeout = self.timeout
        if sys.gettrace() is not None:  # 检查是否在调试模式下
            timeout = None  # 取消超时限制

        # Step 1: Generate offspring in parallel (without evaluating them)
        try:
            with multiprocessing.Pool(self.n_p) as pool:
                generate_args = [
                    (operator, parent, coor_individuals, iterecneed)
                    for parent in flatten_parent
                ]
                results = pool.starmap(self.get_offspring, generate_args)
        except Exception as e:
            print(f'EC Error in {operator}===>', e)
            print("Parallel time out .")
            traceback.print_exc()

        # Step 2: Evaluate offspring using multiprocessing with timeout control
        try:
            codes = [res['code'] for res in results]
            evaluated_objectives = [None] * len(codes)  # 初始化结果列表

            with multiprocessing.Pool(self.n_p) as pool:
                async_results = []
                for i, code in enumerate(codes):
                    async_result = pool.apply_async(self.evaluate_offspring_with_timeout, (code,))
                    async_results.append((i, async_result))

                for i, async_result in async_results:
                    try:
                        evaluated_objectives[i] = async_result.get(timeout=self.timeout)
                    except multiprocessing.TimeoutError:
                        print(f"Timeout for code at index {i}: {codes[i]}")
                        evaluated_objectives[i] = None
                    except Exception as e:
                        print(f"Error for code at index {i}: {codes[i]}, Error: {e}")
                        evaluated_objectives[i] = None
        except Exception as e:
            print(f"Error during multiprocessing evaluation: {e}")
            traceback.print_exc()
            for res in results:
                res['objective'] = None

        # 将评估结果回填到 results 中
        for res, obj in zip(results, evaluated_objectives):
            res['objective'] = None if obj is None else np.round(obj, 8)

        print(f'There are {len(flatten_parent)} parent, we get {len(results)} offspring in results')
        for res in results:
            self.sample_times += 1
            record = {
                'sample_order': self.sample_times,  # 样本编号
                'algorithm': res['algorithm'],
                'code': res['code'],
                'objective': res['objective'],
            }

            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.append(record)
            else:
                data = [record]

            with open(self.log_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            if record['objective'] is not None:
                if os.path.exists(self.log_best_file_path):
                    with open(self.log_best_file_path, "r", encoding="utf-8") as f:
                        data_best = json.load(f)
                    if data_best[-1]['objective'] >= record['objective']:
                        data_best.append(record)
                else:
                    data_best = [record]

                with open(self.log_best_file_path, "w", encoding="utf-8") as f:
                    json.dump(data_best, f, indent=4, ensure_ascii=False)

        clusters_new = kwargs.get('clusters')
        for cluster in clusters_new:
            cluster.clear_offspring()
        for resind, result in enumerate(results):
            if result['objective'] is not None and not math.isnan(result['objective']):
                clusters_new[flatten_parent[resind].whichcluster].add_individual(result)
                clusters_new[flatten_parent[resind].whichcluster].add_offspring(result)
                flatten_parent[resind].update_opresult_recorder(operator, result)