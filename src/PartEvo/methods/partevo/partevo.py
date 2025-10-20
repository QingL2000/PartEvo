import numpy as np
import json
import random
import time
import os
import datetime  # 新增，用于生成时间戳

from .partevo_interface_EC_multiprocess import InterfaceEC
from .partevo_summarizer import Summarizer
from .partevo_monitor import Monitor

from .individual_cluster import Individual, individual_feature, individual_cluster, \
    get_random_cooperator_clusters
from .util import ExternalSet
from joblib import Parallel, delayed


# main class for PartEvo
class PartEvo:

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
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.init_times_limit = 3 * self.pop_size  # initialization try times
        self.iteration_count = paras.ec_n_pop  # number of populations iteration count

        self.branch_novelty = paras.branch_novelty  # control diversity of initial branches
        self.stepbystep_flag = paras.stepbystep_flag  # whether LLM write chain of though

        self.operators = paras.ec_operators  # self.ec_operators = ['re', 'cc', 'se', 'lge']
        self.operator_weights = paras.ec_operator_weights

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path

        # for sie
        self.m = paras.ec_m  # 每个cluster中有self.m个individual可以进化
        self.cluster_num = paras.Cluster_number  # Clusters的总数
        self.k = paras.coor_cluster_num  # Crossover betw clusters时参与的clusters数量
        self.ExternalSet_size = paras.ExternalSet_size
        self.external_set = ExternalSet(self.ExternalSet_size)
        self.summary = ""
        self.reflect_flag = paras.reflect

        ### now no use ###
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc  # number of cpu core
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        self.images_root = paras.images_root
        self.feature_type = paras.feature_type
        self.iterative_init = paras.iterative_init

        self.multimodal = paras.multimodal and getattr(self.prob, 'can_visualize', False) and bool(self.images_root)
        if not self.multimodal:
            print(
                "The multimodal module is not used because either the multimodal settings are incorrect, the problem does not implement can_visualize, or there is no path to the image knowledge base.")

        self.extra_params = kwargs  # if no, {}
        self.threshold = paras.threshold
        self.clustering_algorithm = paras.clustering_algorithm

        print("- PartEvo parameters loaded -")
        self.addition_info_on_logtitle = paras.addition_info_on_logtitle

        # 生成时间戳目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.output_base_dir = os.path.join(self.output_path, "results", self.Algo_Name, timestamp + self.prob.taskname+ self.addition_info_on_logtitle)
        self.population_dir = os.path.join(self.output_base_dir, "pops_generation")
        self.best_dir = os.path.join(self.output_base_dir, "generation_best")
        os.makedirs(self.population_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)

        print(f"Results will be saved in: {self.output_base_dir}")
        # 配置保存到 JSON 文件
        self.configs = {
            'problem': self.prob.taskname,
            'm': self.m,
            'cluster_number': self.cluster_num,
            'iteration_count': self.iteration_count,
            'pop_size': self.pop_size,
            'feature_type': self.feature_type,
            'use_seed': self.use_seed,
            'seed_path': self.seed_path,
            'output_path': self.output_path,
            'exp_n_proc': self.exp_n_proc,
            'timeout': self.timeout,
            'use_numba': self.use_numba,
            'images_root': self.images_root,
            'multimodal': self.multimodal,
            'operators': self.operators,
            'reflect': self.reflect_flag,
            'iterative_init': self.iterative_init,
            'operator_weights': self.operator_weights,
            'clustering_method': self.clustering_algorithm,
            'llm_settings': {
                'use_local_llm': self.use_local_llm,
                'url': self.url,
                'api_endpoint': self.api_endpoint,
                'api_endpoint_url': self.api_endpoint_url,
                'api_key': self.api_key,
                'llm_model': self.llm_model
            },
        }

        self.config_save_path = os.path.join(self.output_base_dir, "config_log.json")
        self.save_config(self.configs)

        self.temp_path = './temp'

        # Set a random seed
        random.seed(2024)


    def save_config(self, configs):
        """
        Save the configuration dictionary to a JSON file.
        """
        try:
            with open(self.config_save_path, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=4, ensure_ascii=False)
            print(f"Configurations saved to {self.config_save_path}")
        except Exception as e:
            print(f"Failed to save configurations: {e}")

    # run ael 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.api_endpoint, self.api_endpoint_url, self.api_key,
                                   self.llm_model,
                                   self.debug_mode, interface_prob,
                                   k=self.k,
                                   m=self.m,
                                   use_local_llm=self.use_local_llm, url=self.url,
                                   n_p=self.exp_n_proc, timeout=self.timeout,
                                   use_numba=self.use_numba,
                                   branch_novelty=self.branch_novelty,
                                   stepbystep_flag=self.stepbystep_flag,
                                   init_times_limit=self.init_times_limit,
                                   logsave_path=self.output_base_dir,
                                   iterative_init=self.iterative_init
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
                population.append(Individual())
                population[-1].create_individual(ind_in_dict=read_ind)

            n_start = 0
            create_epoch = 0
            while len(population) < self.pop_size and create_epoch < self.init_times_limit:
                indibyllm = interface_ec.get_algorithm_single(operator='init', current_pop=population)
                if indibyllm['objective']:
                    population.append(Individual())
                    population[-1].create_individual(ind_in_dict=indibyllm)
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
            individual.set_thought(branches_tags[index])
            self.external_set.add_solution(individual)

        # 聚类
        # individual_feature(population, feature_type=['AST', 'objective'])
        individual_feature(population, self.configs, feature_type=self.feature_type)
        # individual_feature(population, feature_type=('language'))
        clusters = individual_cluster(population, self.cluster_num, clustering_algorithm=self.clustering_algorithm)
        for cluster in clusters:
            for ind in cluster:
                ind.set_cluster(cluster)

        # save generation 0
        filename = os.path.join(self.population_dir, "population_generation_0.json")
        with open(filename, 'w') as f:
            temp_pop = [ind_class.individual_to_dict() for ind_class in population]
            json.dump(temp_pop, f, indent=5)

        # save generation 0
        filename = os.path.join(self.population_dir, "cluster_information.json")
        with open(filename, 'w') as f:
            temp_pop = [ind_class.clusteranalysis_to_dict() for ind_class in population]
            json.dump(temp_pop, f, indent=5)

        filename = os.path.join(self.best_dir, "best_solution_generation_0.json")
        with open(filename, 'w') as f:
            temp_pop = self.external_set.get_best_solution()
            json.dump(temp_pop, f, indent=5)

        # main loop
        for iteration_count in range(n_start, self.iteration_count):  # population iterating
            print(f" -------->Iteration Count: [{iteration_count + 1} / {self.iteration_count}] ", end="|")
            self.summary = summarizer.get_summary(self.external_set, self.summary)

            parent_in_clusters_ops = {}
            ecneed = {'current_pop': [item for sublist in clusters for item in sublist],
                      'clusters': clusters,
                      'External_sorting_set': self.external_set,
                      'summary': self.summary,
                      'best_solution': self.external_set.get_best_solution(),
                      'parent_in_clusters_ops': parent_in_clusters_ops,
                      }

            for opid, op in enumerate(self.operators):
                parent_in_clusters_ops[op] = [cluster.choose_individual(self.m, option=op) for cluster in
                                              clusters]  # 放在option循环内，让新产生的soluiton也有机会成为parent
                if op == "cc":
                    if len(clusters) == 1:
                        continue
                if op == "re":
                    flatten_parent = [item for sublist in parent_in_clusters_ops[op] for item in sublist]
                    if self.reflect_flag:
                        # 并行reflection
                        if hasattr(self.prob, 'visualize') and self.multimodal:
                            Parallel(n_jobs=self.exp_n_proc)(
                                delayed(self.prob.visualize)(individual_i.code,
                                                             os.path.join(self.temp_path,
                                                                          "individual_" + str(i)))
                                for
                                i, individual_i in enumerate(flatten_parent))
                            print('png Done')
                            for i, individual_i in enumerate(flatten_parent):
                                individual_i.set_visual_path(os.path.join(self.temp_path, "individual_" + str(i)))

                        # reflection context
                        reflections = Parallel(n_jobs=self.exp_n_proc)(
                            delayed(minitor.get_reflection)(individual_i, self.multimodal,
                                                            os.path.join(self.temp_path,
                                                                         "individual_" + str(i)))
                            for i, individual_i in enumerate(flatten_parent))
                        print('reflections Done')
                        for index, ind in enumerate(flatten_parent):
                            ind.set_reflection(reflections[index])
                    else:
                        print('Ban reflections')
                        pass
                if op == 'cc':
                    coorclusters_for_each_cluster = [get_random_cooperator_clusters(clusters, cluster_id, self.k) for
                                                     cluster_id, cluster in enumerate(clusters)]
                    coorinds_for_each_cluster_temp = [
                        [cluster.choose_individual(1, option=op) for cluster in coorclusters] for
                        coorclusters in coorclusters_for_each_cluster]
                    coorinds_for_each_cluster = [
                        [ind for sublist in coorinds_for_each_cluster_temp[cluster_id] for ind in sublist]
                        for cluster_id, _ in enumerate(clusters)
                    ]
                    ecneed['coorinds_for_each_cluster'] = coorinds_for_each_cluster

                print(f" Operation: {op}, [{opid + 1} / {len(self.operators)}] ", end="|")
                op_w = self.operator_weights[opid]
                if (np.random.rand() < op_w):
                    interface_ec.get_algorithm(operator=op, **ecneed)

                population = [item for sublist in clusters for item in sublist]
                population_mean_objective = []
                for see_id, branch_see in enumerate(population):
                    print(" Obj: ", branch_see.objective, end="|")
                    population_mean_objective.append(branch_see.objective)
                print()
                print(f'Population Mean objective is: {np.mean(population_mean_objective)}')

                for cluster in clusters:
                    for ind in cluster.offspring:
                        self.external_set.add_solution(ind)
                    cluster.management()

                print('Best Obj is:', self.external_set.get_best_solution()['objective'])
                print('-------------------------------')

            # for cluster in clusters:
            #     cluster.management()
            # Save population to a file
            # 保存当前代的种群
            filename = os.path.join(self.population_dir, f"population_generation_{iteration_count + 1}.json")
            with open(filename, 'w') as f:
                temp_pop = [ind_class.individual_to_dict() for ind_class in population]
                json.dump(temp_pop, f, indent=5)

            # 保存当前代的最佳解
            filename = os.path.join(self.best_dir, f"best_solution_generation_{iteration_count + 1}.json")
            with open(filename, 'w') as f:
                temp_pop = self.external_set.get_best_solution()
                json.dump(temp_pop, f, indent=5)

            print(
                f"--- {iteration_count + 1} of {self.iteration_count} populations finished. Time Cost:  {((time.time() - time_start) / 60):.1f} min")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i].objective) + " ", end="")
            print()

            if self.threshold is not None:
                if self.external_set.get_best_solution()['objective'] <= self.threshold:
                    print(f'We got a good algorithm Now, Stop the search at iteration {iteration_count}')
                    break
