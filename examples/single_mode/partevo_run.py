from src.PartEvo.utils.getParas import Paras
from src.PartEvo import PartEvo
from tqdm import tqdm

if __name__ == "__main__":
    tasks = ['single_mode']
    seedpath = ['single']
    thresholds = [0]
    for i in tqdm(range(1), desc="Outer Loop Progress"):
        # Inner loop wrapped with tqdm
        for taskid, task in enumerate(tqdm(tasks, desc="Task Progress", leave=False)):
            paras = Paras()
            # Set parameters #
            paras.set_paras(method="partevo",  # ['ael','eoh','partevo']
                            problem=task,  # ['tsp_construct','bp_online', mec_task_offloading]single_mode, multi_mode
                            llm_api_endpoint="api.bltcy.ai",  # set your LLM endpoint
                            llm_api_endpoint_url='/v1/chat/completions',
                            llm_api_key="sk-0hCjhh3wBUP7H2TQF9B6D290Ee604cAc88633dDc5f68B0Ed",  # set your key
                            llm_model="gpt-4o-mini",
                            exp_use_seed=True,
                            exp_seed_path=f"./{seedpath[taskid]}.json",
                            ec_pop_size=16,  # number of samples in each population
                            ec_n_pop=15,  # number of populations
                            exp_n_proc=8,  # multi-core parallel
                            exp_debug_mode=False,
                            besta_instruct_prob=1,
                            locala_instruct_prob=1,
                            stepbystep_flag=False,
                            branch_novelty=30,
                            ExternalSet_size=40,
                            images_root="",
                            multimodal=False,
                            Cluster_number=4,
                            eva_timeout=180,
                            feature_type=('AST',),  # ('AST','language', 'random'),
                            coor_cluster_num=2,
                            ec_operators=['re', 'cc', 'se', 'lge'],
                            ec_operator_weights=[1, 1, 1, 1],
                            addition_info_on_logtitle='',
                            reflect=True,
                            threshold=thresholds[taskid]
                            )

            evolution = PartEvo.EVOL(paras)

            evolution.run()
