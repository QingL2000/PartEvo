import numpy as np
import time

from .sie_evolution import Evolution  # 这里面进行实际的初始化、交叉、变异等过程


import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import sys
import random
from .Branch import Branch, get_random_cooperator_branches
import copy

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

    def __init__(self, pop_size, m, api_endpoint, api_endpoint_url, api_key, llm_model, debug_mode, interface_prob,
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
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        # self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba

        self.init_times_limit=self.extra_params.get('init_times_limit', 2*self.pop_size)

        self.best_instruct_prob = self.extra_params.get('besta_instruct_prob', 1)
        self.local_instruct_prob = self.extra_params.get('locala_instruct_prob', 1)

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

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new

    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        population = []
        create_epoch = 0
        while len(population) < self.pop_size and create_epoch < self.init_times_limit:
            print('Initialzation request, ', create_epoch)
            indibyllm = self.get_algorithm_single(operator='init', current_pop=population)
            if indibyllm['objective']:
                population.append(Branch(len(population)+1))
                population[-1].init_branch(ind_in_dict=indibyllm)
                print(f'Algorithm {len(population)} is designed')
            create_epoch += 1
        print('Initialization Done')
        if len(population) < self.pop_size:
            print('Initialization Warn ---> There are no more branches available to meet the specified settings number.')
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
        elif operator == "ie":
            [offspring['code'], offspring['algorithm']] = self.evol.independent_explore(**kwargs)
        elif operator == "ce":
            [offspring['code'], offspring['algorithm']] = self.evol.cooperation_explore(**kwargs)
        elif operator == "ge":
            [offspring['code'], offspring['algorithm']] = self.evol.God_guide_explore(**kwargs)
        elif operator == "pe":
            [offspring['code'], offspring['algorithm']] = self.evol.PSO_explore(**kwargs)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        return offspring

    def get_offspring(self, operator, **kwargs):

        try:
            offspring = self._get_alg(operator, **kwargs)

            if self.use_numba:  # numba是一个用来快速提取函数的库，但不好用，_get_alg中已经写好了直接得到code和algo

                # Regular expression pattern to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"

                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)

                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            # n_retry = 1
            # while self.check_duplicate(pop, offspring['code']):
            #     # 重复了就retry
            #
            #     n_retry += 1
            #     if self.debug:
            #         print("duplicated code, wait 1 second and retrying ... ")
            #
            #     p, offspring = self._get_alg(pop, operator, bestone=bestone, locals=locals)
            #
            #     if self.use_numba:
            #         # Regular expression pattern to match function definitions
            #         pattern = r"def\s+(\w+)\s*\(.*\):"
            #
            #         # Search for function definitions in the code
            #         match = re.search(pattern, offspring['code'])
            #
            #         function_name = match.group(1)
            #
            #         code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            #     else:
            #         code = offspring['code']
            #
            #     if n_retry > 1:
            #         break

            # self.code2file(offspring['code'])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code)
                fitness = future.result(timeout=self.timeout)
                if fitness == None:
                    print('Fitness equal to None, so np.round would raise error')
                offspring['objective'] = np.round(fitness, 8)
                future.cancel()

            # fitness = self.interface_eval.evaluate(code)


        except Exception as e:
            print('Error at get offspring --->', e)
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }

        # Round the objective values
        return offspring

    # def process_task(self,pop, operator):
    #     result =  None, {
    #             'algorithm': None,
    #             'code': None,
    #             'objective': None,
    #             'other_inf': None
    #         }
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.get_offspring, pop, operator)
    #         try:
    #             result = future.result(timeout=self.timeout)
    #             future.cancel()
    #             #print(result)
    #         except:
    #             future.cancel()

    #     return result

    def get_algorithm_single(self, operator, **kwargs):

        try:
            offspring = self.get_offspring(operator, **kwargs)
        except Exception as e:
            print('Error location ----->', operator)
            print('Error:', e)
            offspring = None

        return offspring

    def get_algorithm(self, operator, **kwargs):
        """
        ecneed = {    'current_pop': [branch, branch],
                      'External_sorting_set': class .solutions=[],
                      'summary': summary str
                      'best_solution': self.external_set.get_best_solution()
                      }
        :param operator:
        :param kwargs:
        :return:
        """
        results = []
        iter_currentpop = kwargs.get('current_pop')
        iterecneed = copy.deepcopy(kwargs)
        timeout = self.timeout + 15
        if sys.gettrace() is not None:  # 检查是否在调试模式下
            timeout = None  # 取消超时限制

        try:
            results = Parallel(n_jobs=self.n_p, timeout=timeout)(
                delayed(self.get_offspring)(operator=operator, individual=individual, indivs=get_random_cooperator_branches(iter_currentpop, indidex, self.m), **iterecneed) for indidex, individual in
                enumerate(iter_currentpop))
        except Exception as e:
            print(f'EC Error in {operator}===>', e)
            print("Parallel time out .")

        for resind, result in enumerate(results):
            iter_currentpop[resind].update_opresult_recoder(operator, result)

        return None
    # def get_algorithm(self,pop,operator, pop_size, n_p):

    #     # perform it pop_size times with n_p processes in parallel
    #     p,offspring = self._get_alg(pop,operator)
    #     while self.check_duplicate(pop,offspring['code']):
    #         if self.debug:
    #             print("duplicated code, wait 1 second and retrying ... ")
    #         time.sleep(1)
    #         p,offspring = self._get_alg(pop,operator)
    #     self.code2file(offspring['code'])
    #     try:
    #         fitness= self.interface_eval.evaluate()
    #     except:
    #         fitness = None
    #     offspring['objective'] =  fitness
    #     #offspring['other_inf'] =  first_gap
    #     while (fitness == None):
    #         if self.debug:
    #             print("warning! error code, retrying ... ")
    #         p,offspring = self._get_alg(pop,operator)
    #         while self.check_duplicate(pop,offspring['code']):
    #             if self.debug:
    #                 print("duplicated code, wait 1 second and retrying ... ")
    #             time.sleep(1)
    #             p,offspring = self._get_alg(pop,operator)
    #         self.code2file(offspring['code'])
    #         try:
    #             fitness= self.interface_eval.evaluate()
    #         except:
    #             fitness = None
    #         offspring['objective'] =  fitness
    #         #offspring['other_inf'] =  first_gap
    #     offspring['objective'] = np.round(offspring['objective'],5) 
    #     #offspring['other_inf'] = np.round(offspring['other_inf'],3)
    #     return p,offspring
