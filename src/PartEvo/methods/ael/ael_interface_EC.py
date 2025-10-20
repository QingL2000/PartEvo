import numpy as np
import time

from .ael_evolution import Evolution  # 这里面进行实际的初始化、交叉、变异等过程

import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import sys
import random


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
                 select, n_p, timeout, use_numba, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_endpoint_url, api_key, llm_model, debug_mode, prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba

        self.extra_params = kwargs
        self.best_instruct_prob = self.extra_params.get('besta_instruct_prob', 0)
        self.local_instruct_prob = self.extra_params.get('locala_instruct_prob', 0)

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

        n_create = 2

        population = []

        # for i in range(n_create):
        #     _, pop = self.get_algorithm([], 'i1')
        #     for p in pop:
        #         population.append(p)

        create_epoch = 0
        while len(population) < self.pop_size and create_epoch < n_create:
            _, popbyllm = self.get_algorithm([], 'i1', done=len(population))
            for indi in popbyllm:
                if indi['objective']:
                    population.append(indi)
                if len(population) == self.pop_size:
                    break
            create_epoch += 1
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

    def best_local_judge(self, parents, bestone, locals, indexs, maxlen=4):
        """
        判定是否增加best和local
        """
        # ori_parents_len = len(parents)
        best_flag = False
        local_flag = False
        if len(parents) >= maxlen:
            return parents
        if self.best_instruct_prob:  # 如果有概率启用best algo指导
            if bestone:  # 且存在best algorithm
                passflag = False
                for parent in parents:  # 且所选的parents中没有best algorithm
                    if bestone['objective'] == parent['objective']:
                        passflag = True
                if random.random() < self.best_instruct_prob and not passflag:  # 且触发使用best algo指导
                    parents.append(bestone)
                    best_flag = True
        if len(parents) >= maxlen:
            return parents
        if self.local_instruct_prob:
            if locals:
                for index_decide in indexs:
                    passflag = False
                    for parent in parents:  # 且所选的parents中没有best algorithm
                        if locals[index_decide]['objective'] == parent['objective']:
                            passflag = True
                    if random.random() < self.local_instruct_prob and not passflag:
                        parents.append(locals[index_decide])
                        local_flag = True
                    if len(parents) >= maxlen:
                        return parents
        return parents, best_flag, local_flag

    def _get_alg(self, pop, operator, bestone=None, locals=None):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'], offspring['algorithm']] = self.evol.i1()
        elif operator == "crossover":
            parents, indexs = self.select.parent_selection(pop, self.m)
            parents, bestflag, localflag = self.best_local_judge(parents, bestone, locals, indexs)  # 判定是否加入best或local
            infos = {"best": bestflag, "local": localflag, "oriparnum": self.m}
            [offspring['code'], offspring['algorithm']] = self.evol.crossover_plus_pso(parents, infos)
            # [offspring['code'], offspring['algorithm']] = self.evol.crossover(parents)
        elif operator == "mutation":
            parents, indexs = self.select.parent_selection(pop, 1)
            [offspring['code'], offspring['algorithm']] = self.evol.mutation(parents[0])
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        return parents, offspring

    def get_offspring(self, pop, operator, bestone=None, locals=None):

        try:
            p, offspring = self._get_alg(pop, operator, bestone=bestone, locals=locals)

            if self.use_numba:  # numba是一个用来快速提取函数的库，但不好用，_get_alg中已经写好了直接得到code和algo

                # Regular expression pattern to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"

                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)

                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            n_retry = 1
            while self.check_duplicate(pop, offspring['code']):
                # 重复了就retry

                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")

                p, offspring = self._get_alg(pop, operator, bestone=bestone, locals=locals)

                if self.use_numba:
                    # Regular expression pattern to match function definitions
                    pattern = r"def\s+(\w+)\s*\(.*\):"

                    # Search for function definitions in the code
                    match = re.search(pattern, offspring['code'])

                    function_name = match.group(1)

                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']

                if n_retry > 1:
                    break

            # self.code2file(offspring['code'])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code)
                fitness = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5)
                future.cancel()

            # fitness = self.interface_eval.evaluate(code)


        except Exception as e:

            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        # Round the objective values
        return p, offspring

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

    def get_algorithm(self, pop, operator, bestone=None, locals=None, done=0):

        results = []

        timeout = self.timeout + 15
        if sys.gettrace() is not None:  # 检查是否在调试模式下
            timeout = None  # 取消超时限制

        try:
            results = Parallel(n_jobs=self.n_p, timeout=timeout)(
                delayed(self.get_offspring)(pop, operator, bestone=bestone, locals=locals) for _ in
                range(self.pop_size - done))
        except Exception as e:
            print('Error:', e)
            print("Parallel time out .")

        # time.sleep(2)

        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
        return out_p, out_off
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
