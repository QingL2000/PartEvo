import numpy as np
import json
import random
import time
import os
import copy

from .ael_interface_EC import InterfaceEC


# main class for AEL
class AEL:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage

        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_endpoint_url = paras.llm_api_endpoint_url
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations iteration count
        self.init_times_limit = self.pop_size

        self.operators = paras.ec_operators  # self.ec_operators = ['crossover', 'mutation']
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        self.bestalgo_in_iteration = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }

        self.local_algos = []

        self.extra_params = kwargs
        self.psoparams = self.extra_params.get('psoparams', {})
        self.besta_instruct_prob = self.psoparams.get('besta_instruct_prob', 0)
        self.locala_instruct_prob = self.psoparams.get('locala_instruct_prob', 0)

        print("- AEL parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)

    # run ael 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_endpoint_url, self.api_key,
                                   self.llm_model,
                                   self.debug_mode, interface_prob, use_local_llm=self.use_local_llm, url=self.url,
                                   select=self.select, n_p=self.exp_n_proc, timeout=self.timeout,
                                   use_numba=self.use_numba,
                                   besta_instruct_prob=self.besta_instruct_prob,
                                   locala_instruct_prob=self.locala_instruct_prob
                                   )


        # initialization
        population = []
        if self.use_seed:
            # print("当前路径是：", os.getcwd())
            with open(self.seed_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data, self.exp_n_proc)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
            create_epoch = 0
            while len(population) < self.pop_size and create_epoch < self.init_times_limit:
                _, popbyllm = interface_ec.get_algorithm([], 'i1', done=len(population))
                for indi in popbyllm:
                    if indi['objective']:
                        population.append(indi)
                    if len(population) == self.pop_size:
                        break
                create_epoch += 1
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)

                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "/results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0
        # local algorithm
        for individual_local in population:
            self.local_algos.append(copy.deepcopy(individual_local))
        if len(self.local_algos) != self.pop_size:
            self.local_algos.append(copy.deepcopy(self.local_algos[-1]))


        # best algorithm initialization
        for key in self.bestalgo_in_iteration.keys():
            self.bestalgo_in_iteration[key] = population[0][key]
        for individual_index in range(1, len(population)):
            if population[individual_index]['objective'] < self.bestalgo_in_iteration['objective']:
                for key in self.bestalgo_in_iteration.keys():
                    self.bestalgo_in_iteration[key] = population[individual_index][key]
        print('===> The best algorithm get a objective funciton value with {}'.format(
            self.bestalgo_in_iteration['objective']))

        # main loop
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):  # population iterating
            print(f" Iteration Count: [{pop + 1} / {self.pop_size}] ", end="|")
            for i in range(n_op):
                op = self.operators[i]
                print(f" Operation: {op}, [{i + 1} / {n_op}] ", end="|")
                op_w = self.operator_weights[i]
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op, bestone=self.bestalgo_in_iteration, locals=self.local_algos)
                self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                for offindex, off in enumerate(offsprings):
                    print(" Obj: ", off['objective'], end="|")
                    print('len(offsprings)', len(offsprings))
                    if off['objective']:
                        if off['objective'] < self.bestalgo_in_iteration['objective']:  # 时刻记录最优解
                            for key in self.bestalgo_in_iteration.keys():
                                self.bestalgo_in_iteration[key] = off[key]
                            print('---> Best algorithm updated <---')
                        if off['objective'] < self.local_algos[offindex]['objective']:  # 时刻记录local algorithm
                            for key in self.bestalgo_in_iteration.keys():
                                self.local_algos[offindex][key] = off[key]
                            print('---> Local algorithm updated <---')
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                # populatin management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()

            # Save population to a file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(self.bestalgo_in_iteration, f, indent=5)

            print(
                f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time() - time_start) / 60):.1f} min")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()
