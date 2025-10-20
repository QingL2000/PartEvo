import os
import json
import numpy as np
import matplotlib.pyplot as plt
from codebleu import calc_codebleu
from tqdm import tqdm  # 导入 tqdm


# 读取hie样本
# 读取hie样本，并增加筛选功能
def read_hie_samples(hie_dir, max_order=None):
    hie_samples = []
    hie_file = os.path.join(hie_dir, 'sample_log.json')
    with open(hie_file, 'r', encoding='utf-8') as file:
        all_samples = json.load(file)

    if max_order is not None:
        # 筛选出 order 小于 max_order 的样本
        hie_samples = [sample for sample in all_samples if sample.get('sample_order') is not None and sample.get('sample_order') < max_order]
    else:
        hie_samples = all_samples

    return hie_samples


# 读取eoh样本，并增加筛选功能
def read_eoh_samples(eoh_dir, max_order=None):
    eoh_samples = []

    # 1. Separate filenames into two groups: numerical and special
    numerical_files = []
    special_files = []
    for f in os.listdir(eoh_dir):
        if f.startswith('samples_') and f.endswith('.json'):
            if '~' in f:
                numerical_files.append(f)
            elif f == 'samples_best.json':
                special_files.append(f)

    # 2. Sort the numerical files based on the first number in the range
    numerical_files.sort(key=lambda f: int(f.split('_')[1].split('~')[0]))

    # 3. Concatenate the sorted lists. We put numerical files first
    sorted_filenames = numerical_files + special_files

    # 4. Process the files in the correct order
    for filename in sorted_filenames:
        with open(os.path.join(eoh_dir, filename), 'r', encoding='utf-8') as file:
            all_samples = json.load(file)
            if max_order is not None:
                # Filter samples based on max_order
                filtered_samples = [sample for sample in all_samples if
                                    sample.get('sample_order') is not None and sample.get('sample_order') < max_order]
                eoh_samples.extend(filtered_samples)
            else:
                eoh_samples.extend(all_samples)

    return eoh_samples


# 计算相似度分数
def compute_similarity_score(similarity_matrix, include_diagonal=True):
    if not include_diagonal:
        np.fill_diagonal(similarity_matrix, 0)
    total_similarity = np.sum(similarity_matrix)
    num_elements = similarity_matrix.size
    return total_similarity / num_elements


# 计算代码相似度
def calculate_code_similarity(codes):
    code_no = len(codes)
    AST_codebleu = np.zeros((code_no, code_no))
    for i in range(code_no):
        for j in range(code_no):
            cal_result = calc_codebleu([codes[i]], [codes[j]], lang='python',
                                       weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
            AST_codebleu[i, j] = cal_result['codebleu']
    return compute_similarity_score(AST_codebleu)


# 滑动窗口计算相似度
def sliding_window_similarity(codes, window_size):
    similarity_scores = []
    # 使用 tqdm 来显示进度条
    for i in tqdm(range(len(codes) - window_size + 1), desc="Processing Sliding Window", unit="window"):
        window = codes[i:i + window_size]
        similarity_scores.append(calculate_code_similarity(window))
    return similarity_scores


# 可视化相似度曲线
def plot_similarity_curve(hie_similarity, eoh_similarity, funsearch_similarity, reevo_similarity):
    plt.figure(figsize=(10, 6))
    plt.plot(hie_similarity, label='PartEvo Similarity')
    plt.plot(eoh_similarity, label='EOH Similarity')
    plt.plot(funsearch_similarity, label='Funsearch Similarity')
    plt.plot(reevo_similarity, label='ReEvo Similarity')
    plt.xlabel('Sample Index')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.title('Code Similarity Over Time')
    plt.show()


# 无重叠滑动窗口计算相似度
# def sliding_window_similarity_no_overlap(codes, window_size):
#     similarity_scores = []
#     for i in tqdm(range(0, len(codes), window_size), desc="Processing Sliding Window (No Overlap)", unit="window"):
#         # 如果剩余的样本数量不足 window_size，就直接取剩余的部分
#         window = codes[i:i + window_size]
#         similarity_scores.append(calculate_code_similarity(window))
#     return similarity_scores


def sliding_window_similarity_no_overlap(codes, window_size, mode='all'):
    """
    使用无重叠滑动窗口计算代码相似度。

    Args:
        codes (list): 代码样本列表。
        window_size (int): 滑动窗口的大小。
        mode (str): 计算模式。可选值为 'all' 或 'first_last'。
                    'all'：计算所有窗口的相似度。
                    'first_last'：只计算第一个和最后一个窗口的相似度。

    Returns:
        list: 相似度分数列表。
    """
    similarity_scores = []

    if mode == 'first_last':
        # 确保代码样本数量足够
        if len(codes) < window_size:
            print("Warning: Not enough code samples for the specified window size.")
            return []

        # 计算第一个窗口
        first_window = codes[:window_size]
        similarity_scores.append(calculate_code_similarity(first_window))
        print('第一窗口计算完毕', similarity_scores)

        # 计算最后一个窗口
        # 从列表末尾向前取 window_size 个样本
        last_window = codes[-window_size:]
        similarity_scores.append(calculate_code_similarity(last_window))
        print('最后窗口计算完毕', similarity_scores)

        return similarity_scores

    elif mode == 'all':
        # 计算所有窗口
        for i in tqdm(range(0, len(codes), window_size), desc="Processing Sliding Window (No Overlap)", unit="window"):
            window = codes[i:i + window_size]
            similarity_scores.append(calculate_code_similarity(window))
        return similarity_scores

    else:
        # 如果模式参数无效，抛出错误
        raise ValueError("Invalid mode. Please choose 'all' or 'first_last'.")


# 主程序
def main(hie_dir, eoh_dir, funsearch_dir, reevo_dir):
    # single_peak

    hie_samples = read_hie_samples(hie_dir, max_order=500)
    eoh_samples = read_eoh_samples(eoh_dir, max_order=500)
    funsearch_samples = read_eoh_samples(funsearch_dir, max_order=500)
    reevo_samples = read_eoh_samples(reevo_dir, max_order=500)

    hie_code_sample = [sample['code'] for sample in hie_samples]
    eoh_code_sample = [sample['function'] for sample in eoh_samples]
    funsearch_code_sample = [sample['function'] for sample in funsearch_samples]
    reevo_code_sample = [sample['function'] for sample in reevo_samples]

    print('len(partevo_code_sample)', len(hie_code_sample))
    print('len(eoh_code_sample)', len(eoh_code_sample))
    print('len(funsearch_code_sample)', len(funsearch_code_sample))
    print('len(Reevo_code_sample)', len(reevo_code_sample))

    min_len = min(len(hie_code_sample), len(eoh_code_sample), len(funsearch_code_sample))

    hie_code_sample = hie_code_sample[:min_len]
    eoh_code_sample = eoh_code_sample[:min_len]
    funsearch_code_sample = funsearch_code_sample[:min_len]
    reevo_code_sample = reevo_code_sample[:min_len]

    # 设置滑动窗口大小
    window_size = 50  # 你可以调整这个值

    # 计算滑动窗口中的相似度（无重叠）
    print('partevo calculating')
    hie_similarity = sliding_window_similarity_no_overlap(hie_code_sample, window_size, mode='first_last')
    print('eoh calculating')
    eoh_similarity = sliding_window_similarity_no_overlap(eoh_code_sample, window_size, mode='first_last')
    print('funsearch calculating')
    funsearch_similarity = sliding_window_similarity_no_overlap(funsearch_code_sample, window_size, mode='first_last')
    print('reevo calculating')
    reevo_similarity = sliding_window_similarity_no_overlap(reevo_code_sample, window_size, mode='first_last')

    # 绘制相似度曲线
    plot_similarity_curve(hie_similarity, eoh_similarity, funsearch_similarity, reevo_similarity)


# 运行主程序
if __name__ == '__main__':
    tasks = ['single', 'multi', 'taskoffloading', 'machinclevel']
    id = 3
    directories = {'single':{'PartEvo':r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\partevo\p1\202501231723single_peak_free_AST_0.0',
                             'EoH': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\eoh_2\single_1\20250122_144129_Problem_EoH\samples',
                             'Funsearch': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\funsearch_2\single_1\20250122_144131_Problem_Method\samples',
                             'ReEvo':r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\reevo\single_1\20250807_235614\samples'},
                   'multi': {
                       'PartEvo': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\partevo\p2\202501230912multi_peak_free_AST_800.0000',
                       'EoH': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\eoh_2\multi_1\20250122_152818_Problem_EoH\samples',
                       'Funsearch': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\funsearch_2\multi_1\20250122_152819_Problem_Method\samples',
                       'ReEvo': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\reevo\multi_1\20250807_232029\samples'},
                   'taskoffloading': {
                       'PartEvo': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\partevo\p3\202501230108task_offloading_bb_free_AST_6471.3110',
                       'EoH': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\eoh_2\task_offloading_1\20250122_180928_Problem_EoH\samples',
                       'Funsearch': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\funsearch_2\task_offloading_1\20250122_180930_Problem_Method\samples',
                       'ReEvo': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\reevo\taskoffloading_1\20250808_012120\samples'},
                   'machinclevel': {
                       'PartEvo': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\partevo\p4\202501222140machine_level_scheduling_free_AST_2792.1',
                       'EoH': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\eoh_2\machine_level_1\20250122_224136_Problem_EoH\samples',
                       'Funsearch': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\funsearch_2\machine_level_1\20250122_205955_Problem_Method\samples',
                       'ReEvo': r'D:\00_Work\00_CityU\04_AEL_MEC\partevo_v2\results\hie\freeinit\reevo\machine_level_1\20250809_002355\samples'}
                   }
    print('任务是', tasks[id])
    hie_dir = directories[tasks[id]]['PartEvo']
    eoh_dir = directories[tasks[id]]['EoH']
    funsearch_dir = directories[tasks[id]]['Funsearch']
    reevo_dir = directories[tasks[id]]['ReEvo']
    main(hie_dir, eoh_dir, funsearch_dir, reevo_dir)
