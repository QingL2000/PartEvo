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
    tasks = ['machine_level_scheduling']
    seedpath = ['machine_level']
    thresholds = [2000]
    # tasks = ['single_mode', 'machine_level_scheduling']
    # seedpath = ['single', 'machine_level']
    # tasks = ['multi_mode']
    for i in tqdm(range(1), desc="Outer Loop Progress"):
        # Inner loop wrapped with tqdm
        for taskid, task in enumerate(tqdm(tasks, desc="Task Progress", leave=False)):
            paras = Paras()
            # Set parameters #
            paras.set_paras(method="hie",  # ['ael','eoh','sie']
                            problem=task,  # ['tsp_construct','bp_online', mec_task_offloading]single_mode, multi_mode
                            llm_api_endpoint="api.bltcy.ai",  # set your LLM endpoint
                            llm_api_endpoint_url='/v1/chat/completions',
                            llm_api_key="sk-0hCjhh3wBUP7H2TQF9B6D290Ee604cAc88633dDc5f68B0Ed",  # set your key
                            llm_model="gpt-4o-mini",
                            exp_use_seed=True,
                            exp_seed_path=f"./seeds/{seedpath[taskid]}.json",
                            ec_pop_size=16,  # number of samples in each population
                            ec_n_pop=15,  # number of populations
                            exp_n_proc=8,  # multi-core parallel
                            exp_debug_mode=False,
                            besta_instruct_prob=1,
                            locala_instruct_prob=1,
                            stepbystep_flag=False,
                            branch_novelty=30,
                            ExternalSet_size=40,
                            # images_root="./get_instances_png/instances_solution_png",
                            images_root="",
                            multimodal=False,  # 是否观察中间解的可视化结果
                            Cluster_number=4,
                            eva_timeout=180,
                            feature_type=('AST',),  # ('AST','language', 'random'),
                            coor_cluster_num=2,
                            ec_operators=['re', 'cc', 'se', 'lge'],
                            ec_operator_weights=[1, 1, 1, 1],
                            addition_info_on_logtitle='GMM',
                            reflect=True,
                            threshold=thresholds[taskid],
                            clustering_algorithm='GMM'
                            )

            evolution = HIE.EVOL(paras)

            evolution.run()
