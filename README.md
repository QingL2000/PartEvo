

<div align=center>
<h1 align="center">
PartEvo: Partition to Evolution 
</h1>
<h3 align="center">
A Niching-enhanced Evolution with LLMs for Automated Algorithm Discovery
</h3>


[![Github][Github-image]][Github-url]
[![License][License-image]][License-url]
[![Releases][Releases-image]][Releases-url]
[![Wiki][Wiki-image]][Wiki-url]


[Github-image]: https://img.shields.io/badge/github-12100E.svg?style=flat-square
[License-image]: https://img.shields.io/badge/License-MIT-orange?style=flat-square
[Releases-image]: https://img.shields.io/badge/Release-Version_1.0-blue?style=flat-square
[Installation-image]: https://img.shields.io/badge/Web_Demo-Version_1.0-blue?style=flat-square
[Wiki-image]: https://img.shields.io/badge/Docs-参考文档-black?style=flat-square


[Github-url]: https://github.com/FeiLiu36/EOH
[License-url]: https://github.com/FeiLiu36/EOH/blob/main/LICENSE
[Releases-url]: https://github.com/FeiLiu36/EOH/releases
[Wiki-url]: https://github.com/FeiLiu36/EOH/tree/main/docs



</div>
<br>

PartEvo (Partition to Evolve) is a novel framework that deeply integrates Large Language Model-assisted Evolutionary Search
(LES) with Niching strategies. It is designed to significantly boost the efficiency and efficacy of Automated Algorithm Discovery (AAD) in abstract language search spaces.

<img src="./docs/figures/PartEvo.JPG" alt="partevo" width="600" height="280" div align=center>

The PartEvo project will be standardized and integrated into the LLM4AD Platform (https://github.com/Optima-CityU/LLM4AD) in the future for academic comparison and usage. Please stay tuned.

## Introduction 📖

While early LES methods (such as EoH, ReEvo) have demonstrated the feasibility and potential of LES in AAD, they often rely on oversimplified
search mechanisms (e.g., greedy selection), which limits their efficiency.

In Evolutionary Computation (EC), search efficiency is enhanced through better exploration-exploitation trade-offs, which
can be achieved via computational resource allocation techniques such as niching and search space
partitioning. It is natural to consider adapting these established EC techniques to the LES to 
enhance algorithm discovery. However, applying them to LES poses new challenges due to the fundamental shift in the nature
of the search space.

With the integration of LLMs, the search space extends beyond traditional numerical or manually designed discrete spaces 
into language spaces. Unlike traditional search spaces, language spaces lack explicit dimensionality
and well-defined structures. Instead, they are implicitly shaped by the interaction between the LLMs and the specific 
task context. This abstraction complicates the adoption of advanced EC techniques. For example, in numerical domains, 
niches can often be defined using distance thresholds. However, in language spaces, it is 
inherently challenging to compute distances between algorithms, which hinders the application of niche-based EC techniques.

To address these challenges, we present a practical pipeline for partitioning language search spaces and constructing 
niches, integrated into a general LES framework to improve search efficiency. This framework facilitates the seamless
incorporation of niche-based EC techniques into LES, enabling more effective allocation of sampling resources (i.e., 
queries to LLMs) during the search process. Critically, it also establishes a methodological blueprint for incorporating
a diverse range of advanced EC methods into future LES pipelines. Building on this foundation, we propose PartEvo
, an LES method that combines advanced prompting strategies with effective EC techniques. PartEvo significantly 
improves search efficiency and excels in AAD tasks, particularly under limited sampling budgets.

## PartEvo Example Usage 💻 

### 🎁 Step 1: Requirements & Installation

You can quickly set up the required Python environment using the provided `environment.yml` file.

1.  **Create the Conda environment**:
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the environment**:
    ```bash
    conda activate mles
    ```

### Step 2: Try Example: 

**<span style="color: red;">Setup your Endpoint and Key for remote LLM or Setup your local LLM before start !</span>** 

For example, set the llm_api_endpoint to "api.deepseek.com", set llm_api_key to "your key", and set llm_model to "deepseek-chat".

```python
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
                            ec_pop_size=8,  # number of samples in each population
                            ec_n_pop=16,  # number of populations
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
```

### Step 3: Use PartEvo solve your local problem 

```bash
cd examples/user_XXX

python partevo_run.py
```

If you encounter any difficulty using the code, you can contact us through the above or submit an [issue](https://github.com/FeiLiu36/EoH/issues)


## ✨Citation (Reference)

If you find MLES helpful please cite:

```bibtex
@inproceedings{
qlh2025partition,
title={Partition to Evolve: Niching-enhanced Evolution with {LLM}s for Automated Algorithm Discovery},
author={Qinglong Hu, Qingfu Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=OEawM2coNT}
}
```

