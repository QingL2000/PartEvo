import json
import numpy as np
from src.HIE.problems.optimization.mec_task_offloading_blackbox import MECENV, mec_instance
from src.HIE.problems.optimization.single_mode_blackbox import Baseline
from src.HIE.problems.optimization.machine_level_scheduling import MLSENV, Environment
from src.HIE.problems.optimization.multi_mode_blackbox import Baseline_multi
from run import Baseline_for_generationtest

import os
from human_algo import GSPSO, DE_optimized, GA_optimized, PSO, DE, GA, terrable_2792
import copy
from joblib import Parallel, delayed


def get_code(path):
    """
    从JSON文件中抽取最后一次采样的code
    :param path: JSON文件的路径
    :return: 最后一次采样的code
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 读取json文件

    # 获取最后一次采样记录（按sample_order排序）
    last_sample = sorted(data, key=lambda x: x['sample_order'], reverse=True)[0]
    if 'code' in last_sample:
        return last_sample['code']
    else:
        # 如果没有 'code'，提取 'function'，并添加 import numpy as np
        function = last_sample['function']  # function是一个Python代码的字符串形式（只有函数，没有import库）

        # 如果 function 中没有 'import numpy as np'，则添加它
        if 'import numpy as np' not in function:
            function = 'import numpy as np\n' + function

        return function


def get_paths_and_parameters(problem_type):
    """
    根据问题类型返回相应的路径和参数
    :param problem_type: 问题类型 ("single", "multi", "task_offloading", "machine_level")
    :return: 包含路径和参数的字典
    """
    config = {
        'single': {
            'paths': [
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\partevo\202501161648single_peakseed_language\best_sample_log.json',
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\funsearch\single_1\20250117_221632_Problem_Method\samples\samples_best.json',
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\eoh\single_3\20250118_134820_Problem_EoH\samples\samples_best.json'
            ],
            'legends': ['PartEvo', 'Funsearch', 'Eoh'],
        },
        'multi': {
            'paths': [
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\partevo\202501211602multi_peak_language\best_sample_log.json',
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\funsearch\multi_3\20250118_140453_Problem_Method\samples\samples_best.json',
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\eoh\multi_1\20250118_151856_Problem_EoH\samples\samples_best.json'  # 1best
            ],
            'legends': ['PartEvo', 'Funsearch', 'Eoh'],
            'GSPSOvalue': 860,
            'max_value': 1500
        },
        'task_offloading': {
            'paths': [
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\partevo\202501250233task_offloading_bb_AST\best_sample_log.json',
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\funsearch\task_offloading_6\20250124_113342_Problem_Method\samples\samples_best.json',
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\eoh\taskoffloading_6\20250124_113351_Problem_EoH\samples\samples_best.json'
            ],
            'legends': ['PartEvo', 'Funsearch', 'Eoh'],
            'GSPSOvalue': 6473.46,
            'max_value': 10000
        },
        'machine_level': {
            'paths': [
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\partevo\202501220403machine_level_scheduling_AST\best_sample_log.json',
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\funsearch\machine_level_2\20250119_011554_Problem_Method\samples\samples_best.json',
                r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\seed\eoh\machine_level_2\20250118_002603_Problem_EoH\samples\samples_best.json'
            ],
            'legends': ['PartEvo', 'Funsearch', 'Eoh'],
        }
    }

    return config.get(problem_type, None)


def compute_human_grade(algo, instance):
    return instance.objfunction(
        algo(copy.deepcopy(instance.inited_positions), instance.upper, instance.lower,
             instance.objfunction))


if __name__ == "__main__":
    import time
    jsonsave_directory = r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\test_instances'

    problem_type = "single"  # 选择问题类型（single, multi, task_offloading, machine_level）
    tast_mode = True

    # 创建测试环境
    if problem_type == 'single':
        test_env = Baseline_for_generationtest(test_mode=tast_mode)
    elif problem_type == 'multi':
        test_env = Baseline_multi(test_mode=tast_mode)
    elif problem_type == 'task_offloading':
        test_env = MECENV(test_mode=tast_mode)
    elif problem_type == 'machine_level':
        test_env = MLSENV(test_mode=tast_mode)
    else:
        raise ValueError(f"未知的 problem_type: {problem_type}")

    algos = [DE_optimized]  # 这里可以添加其他的算法
    # algos = [GA, GA_optimized, DE, DE_optimized, PSO, GSPSO]  # 这里可以添加其他的算法
    results_algos = []
    # algo_names = ['GA', 'GA-variant', 'DE', 'DE-variant', 'PSO', 'PSO-variant']  # 这里可以添加其他的算法
    algo_names = ["DE_optimized"]  # 保存算法名称的列表
    # algo_names = ["tamp"]  # 保存算法名称的列表
    for algo, algo_name in zip(algos, algo_names):
        s1 = time.time()
        # 人工算法测试
        results = Parallel(n_jobs=8)(
            delayed(compute_human_grade)(algo, instance) for instance in test_env.instances for _ in range(1))

        results_algos.append({
            'algorithm': algo_name,
            'results': results
        })
        print(f"Results for {algo_name}: {np.sum(results)}")
        for i, r in enumerate(results):
            print(i, r)
        print('use time:', time.time()-s1)

    # single_AST_language, multi_AST_language, task_offloading_AST_language, machine_level_AST_language
    # single_clusters, multi_clusters, task_offloading_clusters, machine_level_clusters
    config = get_paths_and_parameters(problem_type)

    if config:
        total_path = config['paths']
        legends = config['legends']
    else:
        raise ValueError(f"未知的 problem_type: {problem_type}")

    all_results = []
    for pathid, jsonpath in enumerate(total_path):
        # 获取最后一次采样的代码
        code = get_code(jsonpath)
        # print(code)

        # 评估代码的fitness
        fitnesses = test_env.test_evaluate(code)

        # 输出fitness结果
        print('============================')
        print(jsonpath)
        print('----------------------------')
        print(fitnesses)
        print('总的fitness为:', np.sum(fitnesses))
        print('============================')
        print()

        result_data = {
            'Method': legends[pathid],
            'paths': jsonpath,
            'code': code,
            'fitnesses': fitnesses,
            'total_fitness': np.sum(fitnesses)
        }
        all_results.append(result_data)

    # 将人工算法的结果也加入 all_results 中
    for result_algo in results_algos:
        result_data = {
            'Method': result_algo['algorithm'],
            'results': result_algo['results'],
            'total_fitness': np.sum(result_algo['results'])
        }
        all_results.append(result_data)

    if tast_mode:
        flag = 'Test'
    else:
        flag = 'Train'

    # 保存所有结果到一个总文件
    all_results_path = os.path.join(jsonsave_directory, f'{problem_type}_{flag}_results_seed_generalization.json')
    with open(all_results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("所有结果已保存。")

    # JSON文件路径
    # hie - 4
    # jsonpath = r'D:\00_Work\00_CityU\04_AEL_MEC\hle\results\hie\sussces\hie\202412222057\best_sample_log.json'      # single
    # jsonpath = r'D:\00_Work\00_CityU\04_AEL_MEC\hle\results\hie\sussces\hie\202412260923\best_sample_log.json'  # task offloading
    # jsonpath = r'D:\00_Work\00_CityU\04_AEL_MEC\hle\results\hie\sussces\hie\202412301325\best_sample_log.json'  # multi
