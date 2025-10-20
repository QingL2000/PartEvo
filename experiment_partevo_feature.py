from src.PartEvo.utils.getParas import Paras
from src.PartEvo import HIE
from src.PartEvo.problems.optimization.mec_task_offloading_blackbox.run import mec_instance
from src.PartEvo.problems.optimization.machine_level_scheduling.run import Environment
from tqdm import tqdm

# from src.PartEvo.problems.optimization import machine_level_scheduling
# from src.PartEvo.problems.optimization.machine_level_scheduling.datainit_i30_p10_t30_np10 import Dataenv

# tasks = ["mec_task_offloading_new", "mec_task_offloading_blackbox", "single_mode", "multi_mode", "machine_level_scheduling"]
# tasks = ["mec_task_offloading_blackbox", 'mec_task_offloading_new', 'machine_level_scheduling']
# tasks = ['single_mode', 'multi_mode', "mec_task_offloading_blackbox", 'machine_level_scheduling']

if __name__ == "__main__":
    all_set = {
        'machine_level': {
            'ec_n_pop': 14,
            'ec_operators': ['re', 'cc', 'se', 'lge'],
            'addition_info_on_logtitle': "",
            'reflect': True,
            'repeat_times': 2,
            'seedpath': 'machine_level',
            'task': 'machine_level_scheduling',
            'threshold': 0
        },
    }
    # tasks = ['single_mode', 'multi_mode', "mec_task_offloading_blackbox", 'machine_level_scheduling']
    # seedpath = ['single', 'multi', 'task_offloading', 'machine_level']
    # thresholds = [0, 800, None, None]
    features = ('language', 'random')
    # features = ('language', 'AST', 'random')

    for key, value in tqdm(all_set.items(), desc="Processing Outloop"):
        for feature in tqdm(features, desc="Feature"):
            for i in tqdm(range(value['repeat_times']), desc="Repeat times"):
                paras = Paras()
                # Set parameters #
                paras.set_paras(method="hie",  # ['ael','eoh','sie']
                                problem=value['task'],
                                # ['tsp_construct','bp_online', mec_task_offloading]single_mode, multi_mode
                                llm_api_endpoint="api.bltcy.ai",  # set your LLM endpoint
                                llm_api_endpoint_url='/v1/chat/completions',
                                llm_api_key="sk-0hCjhh3wBUP7H2TQF9B6D290Ee604cAc88633dDc5f68B0Ed",  # set your key
                                llm_model="gpt-4o-mini",
                                exp_use_seed=True,
                                exp_seed_path=f"./seeds/{value['seedpath']}.json",
                                ec_pop_size=16,  # number of samples in each population
                                ec_n_pop=value['ec_n_pop'],  # number of populations
                                exp_n_proc=8,  # multi-core parallel
                                exp_debug_mode=False,
                                stepbystep_flag=False,
                                branch_novelty=30,
                                ExternalSet_size=40,
                                # images_root="./get_instances_png/instances_solution_png",
                                images_root="",
                                multimodal=False,  # 是否观察中间解的可视化结果
                                Cluster_number=4,
                                eva_timeout=180,
                                feature_type=(feature,),  # ('AST','language', 'random'),
                                coor_cluster_num=2,
                                ec_operators=value['ec_operators'],
                                ec_operator_weights=[1, 1, 1, 1],
                                addition_info_on_logtitle='_'+feature,
                                reflect=value['reflect'],
                                threshold=value['threshold']
                                )

                evolution = HIE.EVOL(paras)

                evolution.run()
