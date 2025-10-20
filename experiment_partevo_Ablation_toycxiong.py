from src.HIE.utils.getParas import Paras
from src.HIE import HIE
from src.HIE.problems.optimization.mec_task_offloading_blackbox.run import mec_instance
from src.HIE.problems.optimization.machine_level_scheduling.run import Environment
from tqdm import tqdm

# from src.HIE.problems.optimization import machine_level_scheduling
# from src.HIE.problems.optimization.machine_level_scheduling.datainit_i30_p10_t30_np10 import Dataenv

# tasks = ["mec_task_offloading_new", "mec_task_offloading_blackbox", "single_mode", "multi_mode", "machine_level_scheduling"]
# tasks = ["mec_task_offloading_blackbox", 'mec_task_offloading_new', 'machine_level_scheduling']
# tasks = ['single_mode', 'multi_mode', "mec_task_offloading_blackbox", 'machine_level_scheduling']

if __name__ == "__main__":
    all_set = {
        # 'entire': {
        #     'ec_n_pop': 14,
        #     'ec_operators': ['re', 'cc', 'se', 'lge'],
        #     'addition_info_on_logtitle': '_entire',
        #     'reflect': True,
        # },
        # 'wo_re': {
        #     'ec_n_pop': 14,
        #     'ec_operators': ['re', 'cc', 'se', 'lge'],
        #     'addition_info_on_logtitle': '_wo_re',
        #     'reflect': False,
        # },
        # 'wo_cc': {
        #     'ec_n_pop': 17,
        #     'ec_operators': ['re', 'se', 'lge'],
        #     'addition_info_on_logtitle': '_wo_cc',
        #     'reflect': True,
        # },
        # 'wo_se': {
        #     'ec_n_pop': 17,
        #     'ec_operators': ['re', 'cc', 'lge'],
        #     'addition_info_on_logtitle': '_wo_se',
        #     'reflect': True,
        # },
        # 'wo_re_se': {
        #     'ec_n_pop': 17,
        #     'ec_operators': ['re', 'cc', 'lge'],
        #     'addition_info_on_logtitle': '_wo_re_se',
        #     'reflect': False,
        # },
        'wo_lge': {
            'ec_n_pop': 22,
            'ec_operators': ['re', 'cc', 'se'],
            'addition_info_on_logtitle': '_wo_lge',
            'reflect': True,
        },
        'wo_lge_cc': {
            'ec_n_pop': 33,
            'ec_operators': ['re', 'se'],
            'addition_info_on_logtitle': '_wo_lge_cc',
            'reflect': True,
        },
        # 'pure': {
        #     'ec_n_pop': 33,
        #     'ec_operators': ['re', 'cc'],
        #     'addition_info_on_logtitle': 'pure',
        #     'reflect': False,
        # },
    }

    # tasks = ['single_mode', 'multi_mode', "mec_task_offloading_blackbox", 'machine_level_scheduling']
    # seedpath = ['single', 'multi', 'task_offloading_2', 'machine_level']
    # thresholds = [0, 800, 6000, 3000]

    tasks = ['single_mode']
    seedpath = ['single']
    thresholds = [0]

    feature = ('AST',)

    for key, value in tqdm(all_set.items(), desc="Processing ablation"):
        for _ in range(2):
            for taskid, task in enumerate(tqdm(tasks, desc="Task Progress", leave=False)):
                paras = Paras()
                # Set parameters #
                paras.set_paras(method="hie",  # ['ael','eoh','sie']
                                problem=task,
                                # ['tsp_construct','bp_online', mec_task_offloading]single_mode, multi_mode
                                llm_api_endpoint="api.bltcy.ai",  # set your LLM endpoint
                                llm_api_endpoint_url='/v1/chat/completions',
                                llm_api_key="sk-0hCjhh3wBUP7H2TQF9B6D290Ee604cAc88633dDc5f68B0Ed",  # set your key
                                llm_model="gpt-4o-mini",
                                exp_use_seed=True,
                                exp_seed_path=f"./seeds/{seedpath[taskid]}.json",
                                ec_pop_size=16,  # number of samples in each population
                                ec_n_pop=value['ec_n_pop'],  # number of populations
                                exp_n_proc=10,  # multi-core parallel
                                exp_debug_mode=False,
                                stepbystep_flag=False,
                                branch_novelty=30,
                                ExternalSet_size=40,
                                # images_root="./get_instances_png/instances_solution_png",
                                images_root="",
                                multimodal=False,  # 是否观察中间解的可视化结果
                                Cluster_number=4,
                                eva_timeout=180,
                                feature_type=feature,  # ('AST','language', 'random'),
                                coor_cluster_num=2,
                                ec_operators=value['ec_operators'],
                                ec_operator_weights=[1, 1, 1, 1],
                                addition_info_on_logtitle=value['addition_info_on_logtitle'],
                                reflect=value['reflect'],
                                threshold=thresholds[taskid],
                                )

                evolution = HIE.EVOL(paras)

                evolution.run()
