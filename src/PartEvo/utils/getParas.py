class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'partevo'
        self.problem = 'mec_task_offloading'
        self.selection = None
        self.management = None

        #####################
        ###  branch settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 5  # number of populations, default = 10
        self.ec_operators = None  # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]

        #####################
        ### LLM settings  ###
        #####################
        self.llm_use_local = False  # if use local model
        self.llm_local_url = None  # your local server 'http://127.0.0.1:11012/completions'
        self.llm_api_endpoint = None  # endpoint for remote LLM, e.g., api.deepseek.com 通义千问：dashscope.aliyuncs.com
        self.llm_api_endpoint_url = None  # 通义千问：/api/v1/services/aigc/text-generation/generation'
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat  通义千问：qwen-turbo

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seed0.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = 1

        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 60
        self.init_times_limit = 3
        self.eva_numba_decorator = False

        self.besta_instruct_prob = 0
        self.locala_instruct_prob = 0

        self.stepbystep_flag = False
        self.branch_novelty = 50
        self.ExternalSet_size = 30  # 外部解集的最大大小

        self.Cluster_number = 5
        self.coor_cluster_num = 2
        self.chances_evo = 2

        self.images_root = ""
        self.multimodal = False
        self.feature_type = ('AST',)
        self.addition_info_on_logtitle = ""
        self.reflect = True
        self.threshold = None
        self.iterative_init = True
        self.clustering_algorithm = 'KMeans'

    def set_parallel(self):
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_debug_mode:
            print(f'This computer has {num_processes} cores')
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")

    def set_ec(self):

        if self.management == None:
            if self.method in ['ael', 'eoh', 'sie', 'partevo']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'
            # elif self.method == 'sie':
            #     self.management = 'dont need'

        if self.selection == None:
            # self.selection = 'prob_rank'
            self.selection = 'equal'

        if self.ec_operators == None:
            if self.method == 'eoh':
                self.ec_operators = ['e1', 'e2', 'm1', 'm2']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1, 1, 1]
            elif self.method == 'partevo':
                self.ec_operators = ['re', 'cc', 'se', 'lge']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1, 1, 1]
            elif self.method == 'sie':
                self.ec_operators = ['ie', 'ce', 'ge', 'pe']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1, 1, 1]
            elif self.method == 'ael':
                self.ec_operators = ['crossover', 'mutation']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1]
            elif self.method == 'ls':
                self.ec_operators = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]
            elif self.method == 'sa':
                self.ec_operators = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]

        if self.method in ['ls', 'sa'] and self.ec_pop_size > 1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")

    def set_evaluation(self):
        # Initialize evaluation settings
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            self.eva_numba_decorator = False
            # self.eva_numba_decorator = True
        elif self.problem == 'tsp_construct':
            self.eva_timeout = 20

    def set_paras(self, *args, **kwargs):

        # Map paras
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Identify and set parallel
        self.set_parallel()

        # Initialize method and ec settings
        self.set_ec()

        # Initialize evaluation settings
        self.set_evaluation()

        # Identify and set pso related params
        self.psoparams = {'besta_instruct_prob': self.besta_instruct_prob,
                          'locala_instruct_prob': self.locala_instruct_prob}


if __name__ == "__main__":
    # Create an instance of the Paras class
    paras_instance = Paras()

    # Setting parameters using the set_paras method
    paras_instance.set_paras(llm_use_local=True, llm_local_url='http://example.com', ec_pop_size=8)

    # Accessing the updated parameters
    print(paras_instance.llm_use_local)  # Output: True
    print(paras_instance.llm_local_url)  # Output: http://example.com
    print(paras_instance.ec_pop_size)  # Output: 8
