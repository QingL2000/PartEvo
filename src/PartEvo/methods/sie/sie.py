import numpy as np
import json
import random
import time
import os
import copy

from .sie_interface_EC import InterfaceEC
from .sie_summarizer import Summarizer
from .sie_monitor import Monitor

from .Branch import Branch
from .util import ExternalSet
from joblib import Parallel, delayed


# main class for AEL
class SIE:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.manage = manage
        self.Algo_Name = paras.method

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
        self.branch_num = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.init_times_limit = 2 * self.branch_num  # initialization try times
        self.iteration_count = paras.ec_n_pop  # number of populations iteration count

        self.branch_novelty = paras.branch_novelty  # control diversity of initial branches
        self.stepbystep_flag = paras.stepbystep_flag  # whether LLM write chain of though

        self.operators = paras.ec_operators  # self.ec_operators = ['ie', 'ce', 'ge', 'pe']
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.branch_num or paras.ec_m == 1:  # ce 时参与的 branches 数量
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m  # ce 时参与的 branches 数量

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path

        # for sie
        self.ExternalSet_size = paras.ExternalSet_size
        self.external_set = ExternalSet(self.ExternalSet_size)
        self.summary = ""

        ### now no use ###
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc  # number of cpu core
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        self.images_root = paras.images_root

        self.multimodal = paras.multimodal and getattr(self.prob, 'can_visualize', False) and bool(self.images_root)
        if not self.multimodal:
            print("The multimodal module is not used because either the multimodal settings are incorrect, the problem does not implement can_visualize, or there is no path to the image knowledge base.")

        self.extra_params = kwargs  # if no, {}

        print("- PartEvo parameters loaded -")

        # Save population to a file
        self.population_dir = f"{self.output_path}/results/{self.Algo_Name}/pops"
        self.best_dir = f"{self.output_path}/results/{self.Algo_Name}/pops_best"
        os.makedirs(self.population_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)

        self.temp_path = './temp'

        # Set a random seed
        random.seed(2024)

    # run ael 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.branch_num, self.m, self.api_endpoint, self.api_endpoint_url, self.api_key,
                                   self.llm_model,
                                   self.debug_mode, interface_prob, use_local_llm=self.use_local_llm, url=self.url,
                                   n_p=self.exp_n_proc, timeout=self.timeout,
                                   use_numba=self.use_numba,
                                   branch_novelty=self.branch_novelty,
                                   stepbystep_flag=self.stepbystep_flag,
                                   init_times_limit=self.init_times_limit
                                   )

        minitor = Monitor(api_endpoint=self.api_endpoint,
                          api_endpoint_url=self.api_endpoint_url,
                          api_key=self.api_key,
                          model_LLM=self.llm_model,
                          debug_mode=self.debug_mode,
                          prob=interface_prob,
                          stepbystep_flag=self.stepbystep_flag,
                          n_p=self.exp_n_proc,
                          timeout=self.timeout,
                          use_numba=self.use_numba)
        if self.images_root:
            minitor.get_knowledge(images_root=self.images_root)

        summarizer = Summarizer(api_endpoint=self.api_endpoint,
                                api_endpoint_url=self.api_endpoint_url,
                                api_key=self.api_key,
                                model_LLM=self.llm_model,
                                debug_mode=self.debug_mode,
                                prob=interface_prob,
                                stepbystep_flag=self.stepbystep_flag,
                                n_p=self.exp_n_proc,
                                timeout=self.timeout,
                                use_numba=self.use_numba)

        # initialization
        population = []
        if self.use_seed:
            with open(self.seed_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            read_individuals = interface_ec.population_generation_seed(data, self.exp_n_proc)
            for read_index, read_ind in enumerate(read_individuals):
                population.append(Branch(len(population) + 1))
                population[-1].init_branch(ind_in_dict=read_ind)

            n_start = 0
            create_epoch = 0
            while len(population) < self.branch_num and create_epoch < self.init_times_limit:
                indibyllm = interface_ec.get_algorithm_single(operator='init', current_pop=population)
                if indibyllm['objective']:
                    population.append(Branch(len(population) + 1))
                    population[-1].init_branch(ind_in_dict=indibyllm)
                    read_individuals.append(indibyllm)
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

                print(f"Pop initial: ")
                for offid, off in enumerate(population):
                    print(" Obj: ", off.objective, end="|")
                print()

                n_start = 0
        print("----> initial population has been created! <----")
        print(f'There are total {len(population)} branch')
        # set tags and external set
        branches_tags = summarizer.get_tags(population)

        for index, individual in enumerate(population):
            individual.set_branch(branches_tags[index])
            self.external_set.add_solution(individual)

        # save generation 0
        filename = f"{self.population_dir}/population_generation_0.json"
        with open(filename, 'w') as f:
            temp_pop = [ind_class.branch_to_dict() for ind_class in population]
            json.dump(temp_pop, f, indent=5)
        filename = f"{self.best_dir}/population_generation_0.json"
        with open(filename, 'w') as f:
            temp_pop = self.external_set.get_best_solution()
            json.dump(temp_pop, f, indent=5)

        # main loop
        for iteration_count in range(n_start, self.iteration_count):  # population iterating
            print(f" -------->Iteration Count: [{iteration_count + 1} / {self.iteration_count}] ", end="|")
            self.summary = summarizer.get_summary(self.external_set, self.summary)
            ecneed = {'current_pop': population,
                      'External_sorting_set': self.external_set,
                      'summary': self.summary,
                      'best_solution': self.external_set.get_best_solution()
                      }
            for opid, op in enumerate(self.operators):
                # 并行reflection
                if hasattr(self.prob, 'visualize') and self.multimodal:
                    Parallel(n_jobs=self.exp_n_proc)(
                        delayed(self.prob.visualize)(branch_i.code, os.path.join(self.temp_path, "branch_" + str(branch_i.branchno)))
                        for
                        i, branch_i in enumerate(population))
                    print('png Done')

                # # reflection context
                # reflections = Parallel(n_jobs=self.exp_n_proc)(
                #     delayed(minitor.get_reflection)(branch_i, self.multimodal,
                #                                     os.path.join(self.temp_path, "branch_" + str(branch_i.branchno)))
                #     for i, branch_i in enumerate(population))
                # print('reflections Done')
                # for index, individual in enumerate(population):
                #     individual.set_reflection(reflections[index])

                print(f" Operation: {op}, [{opid + 1} / {len(self.operators)}] ", end="|")
                op_w = self.operator_weights[opid]
                if (np.random.rand() < op_w):
                    interface_ec.get_algorithm(operator=op, **ecneed)

                for see_id, branch_see in enumerate(population):
                    print(" Obj: ", branch_see.opresult_recorder[op]['objective'], end="|")
                print('\n')

                for branch_inside_selection in population:
                    branch_inside_selection.selection_in_branch()
                    self.external_set.add_solution(branch_inside_selection)

            # Save population to a file
            filename = f"{self.population_dir}/population_generation_{iteration_count + 1}.json"
            with open(filename, 'w') as f:
                temp_pop = [ind_class.branch_to_dict() for ind_class in population]
                json.dump(temp_pop, f, indent=5)

            # Save the best one to a file
            filename = f"{self.best_dir}/population_generation_{iteration_count + 1}.json"
            with open(filename, 'w') as f:
                temp_pop = self.external_set.get_best_solution()
                json.dump(temp_pop, f, indent=5)

            print(
                f"--- {iteration_count + 1} of {self.iteration_count} populations finished. Time Cost:  {((time.time() - time_start) / 60):.1f} min")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i].objective) + " ", end="")
            print()
