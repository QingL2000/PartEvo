import random

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

import numpy as np
# from codebleu.syntax_match import calc_syntax_match
from codebleu import calc_codebleu
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from transformers import BertTokenizer, BertModel
import torch

"""
该py脚本实现了HIE算法中的种群个体、种群聚类、以及簇管理。能够支持HIE中的LLM+EC算法自动设计过程.
1.种群首先初始化
2.初始化后聚类成K类，聚类时使用某种距离来度量
3.聚类后将进行HIE框架下的进化。
"""


class Individual:
    """
    个体类，每个个体包含algorithm、code和objective等关键信息，然后会被总结者赋予thought,会被Minitor赋予reflection，所有与该个体相关的option都会被记录，
    """

    def __init__(self):
        self.algorithm = ""
        self.code = ""
        self.objective = None
        self.thought = ""
        self.history_option = ""
        self.history_thought = ""
        # self.selection_strategy = greedy
        self.reflection = ""
        self.feature = []
        self.visual_path = ""
        self.whichcluster = None

        options = ['Init', 're', 'cc', 'lge', 'se']
        contents = ['algorithm', 'code', 'objective']
        self.opresult_recorder = {option: {content: None for content in contents} for option in options}

    def set_cluster(self, cluster):
        self.whichcluster = cluster.cluster_no

    def set_feature(self, feature):
        self.feature = feature

    def set_visual_path(self, visual_path):
        self.visual_path = visual_path

    def update_opresult_recorder(self, op, offspring_dict):
        self.opresult_recorder[op]['algorithm'] = offspring_dict['algorithm']
        self.opresult_recorder[op]['code'] = offspring_dict['code']
        self.opresult_recorder[op]['objective'] = offspring_dict['objective']

    def create_individual(self, **kwargs):
        if kwargs.get('ind_in_dict'):
            ind_in_dict = kwargs.get('ind_in_dict')
            self.algorithm = ind_in_dict['algorithm']
            self.code = ind_in_dict['code']
            self.objective = ind_in_dict['objective']
        elif kwargs.get('algorithm') and kwargs.get('code') and kwargs.get('objective'):
            self.algorithm = kwargs.get('algorithm')
            self.code = kwargs.get('code')
            self.objective = kwargs.get('objective')
        else:
            print('Here is no suitable input to build a branch')
            exit()
        self.opresult_recorder['Init']['algorithm'] = self.algorithm
        self.opresult_recorder['Init']['code'] = self.code
        self.opresult_recorder['Init']['objective'] = self.objective

    def individual_to_dict(self):
        return {'algorithm': self.algorithm,
                'code': self.code,
                'objective': self.objective,
                'other_inf': None}

    def clusteranalysis_to_dict(self):
        return {'algorithm': self.algorithm,
                'code': self.code,
                'objective': self.objective,
                'cluster': self.whichcluster,
                'feature': self.feature.tolist(),
                'other_inf': None}

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def set_reflection(self, reflection):
        self.reflection = reflection

    def set_code(self, code):
        self.code = code

    def set_objective(self, objective):
        self.objective = objective

    def set_thought(self, thought):
        self.thought = thought

    def add_history_option(self, history_option):
        self.history_option += history_option
        self.history_option += '_'

    def add_history_thought(self, history_thought):
        self.history_thought += history_thought
        self.history_thought += '\n'


# def individual_feature(population, feature_type=('AST', 'objective'), save_path='hotpic.png'):
#     """
#     为每个individual构建特征，特征包括：
#     代码相似度矩阵的对应行
#     适应度值
#     """
#     # 确保feature_type不为空
#     if not feature_type:
#         raise ValueError("feature_type cannot be empty.")
#
#     population_size = len(population)
#     features = [[] for _ in range(population_size)]
#
#     if 'AST' in feature_type:
#         AST = np.zeros((population_size, population_size))
#         for i in range(population_size):
#             for j in range(population_size):
#                 cal_result = calc_codebleu([population[i].code], [population[j].code], lang='python', weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
#                 AST[i, j] = 0.4 * cal_result['syntax_match_score'] + 0.4 * cal_result['dataflow_match_score'] + 0.2 * cal_result['weighted_ngram_match_score']
#             features[i].extend(AST[i, :].tolist())  # 转换为列表后再扩展
#
#         print('相似度热图如下:\n', AST)
#         # 绘制热图
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(AST, annot=True, cmap='Blues', xticklabels=False, yticklabels=False)
#         plt.title('Code Similarity Heatmap (AST)')
#
#         # 如果提供了保存路径，则保存热图
#         if save_path:
#             # 确保目录存在
#             # os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             plt.savefig(save_path)
#             print(f"Heatmap saved to {save_path}")
#         else:
#             # 如果没有指定路径，则显示热图
#             plt.show()
#
#
#     if 'objective' in feature_type:
#         for i in range(population_size):
#             features[i].append(population[i].objective)
#
#     # 为每个个体设置特征
#     for i, ind in enumerate(population):
#         ind.set_feature(features[i])

# 使用BERT模型生成文本嵌入
def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained('D:/00_Work/00_CityU/04_AEL_MEC/transformers')
    model = BertModel.from_pretrained('D:/00_Work/00_CityU/04_AEL_MEC/transformers')

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        # 获取BERT输出的最后一层隐藏状态
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

        # 使用[CLS] token的嵌入表示整个句子
        cls_embedding = hidden_states[0, 0, :].numpy()  # [CLS] token的嵌入
        embeddings.append(cls_embedding)

    return np.array(embeddings)


def individual_feature(population, configs, feature_type=('AST', 'language'), save_path='hotpic.png'):
    print('The feature is |---->', feature_type)
    population_size = len(population)
    features = [[] for _ in range(population_size)]

    # 计算AST相似度并降维
    if 'AST' in feature_type:
        AST = np.zeros((population_size, population_size))

        # 计算两两算法代码之间的相似度
        for i in range(population_size):
            for j in range(population_size):
                # 使用CodeBLEU评估语法匹配的相似度
                cal_result = calc_codebleu([population[i].algorithm], [population[j].algorithm], lang='python',
                                           weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
                # 将不同的匹配得分组合成AST相似度，计算AST的语法匹配和数据流匹配
                AST[i, j] = 0.5 * cal_result['syntax_match_score'] + 0.5 * cal_result['dataflow_match_score']
                # AST[i, j] = cal_result['codebleu']

        AST = (AST + AST.T) / 2
        # 将AST相似度作为特征
        for i in range(population_size):
            features[i].extend(AST[i, :].tolist())

    # 计算自然语言嵌入特征
    if 'language' in feature_type:
        texts = [ind.algorithm for ind in population]  # 每个个体的算法描述
        embeddings = get_bert_embeddings(texts)

        for i in range(population_size):
            features[i].extend(embeddings[i, :].tolist())

    if 'random' in feature_type:
        random_features = np.random.normal(size=(population_size, 20))  # 生成与PCA降维后的维度一致的随机特征
        for i in range(population_size):
            features[i].extend(random_features[i, :].tolist())

    # 添加目标值（Objective）特征
    if 'objective' in feature_type:
        for i in range(population_size):
            features[i].append(population[i].objective)

    # 将所有特征拼接后进行统一的PCA降维
    all_features = np.array(features)  # 将所有特征拼接成一个大矩阵

    # 标准化所有特征
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)

    # 进行PCA降维，将特征维度减少到10
    pca = PCA(n_components=10)
    all_features_reduced = pca.fit_transform(all_features)

    # 为每个个体设置特征
    for i, ind in enumerate(population):
        ind.set_feature(all_features_reduced[i])

    # 绘制拼接后的特征热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(all_features, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('All Features')

    if save_path:
        plt.savefig(configs['problem'] + feature_type[0] + 'all_features_' + save_path)
        print(f"Heatmap of All Features saved to {'all_features_' + save_path}")

    # 绘制PCA降维后的所有特征热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(all_features_reduced, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('PCA of All Features')

    if save_path:
        plt.savefig(configs['problem'] + 'PCA_all_features_' + save_path)
        print(f"PCA Heatmap of All Features saved to {'PCA_all_features_' + save_path}")


# def individual_feature(population, feature_type=('AST', 'objective'), save_path='hotpic.png'):
#     population_size = len(population)
#     features = [[] for _ in range(population_size)]
#
#     # 计算AST相似度并降维
#     if 'AST' in feature_type:
#         AST = np.zeros((population_size, population_size))
#         for i in range(population_size):
#             for j in range(population_size):
#                 cal_result = calc_codebleu([population[i].code], [population[j].code], lang='python',
#                                            weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
#                 AST[i, j] = 0.5 * cal_result['syntax_match_score'] + 0.5 * cal_result['dataflow_match_score'] + 0 * \
#                             cal_result['weighted_ngram_match_score']
#
#         # 绘制热图
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(AST, annot=True, cmap='Blues', xticklabels=False, yticklabels=False)
#         plt.title('Code Similarity Heatmap (AST)')
#
#         if save_path:
#             plt.savefig(save_path)
#             print(f"Heatmap saved to {save_path}")
#         else:
#             plt.show()
#
#         # 降维（例如PCA）
#         pca = PCA(n_components=5)  # 这里选择将特征维度降到5
#         AST_reduced = pca.fit_transform(AST)
#
#         for i in range(population_size):
#             features[i].extend(AST_reduced[i, :].tolist())
#
#         print('相似度热图如下:\n', AST)
#
#         print('特征如下:\n', AST_reduced)
#
#
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(AST_reduced, annot=True, cmap='Blues', xticklabels=False, yticklabels=False)
#         plt.title('Code Feature (PCA-AST)')
#
#         if save_path:
#             plt.savefig('PCA'+save_path)
#             print(f"Heatmap saved to {save_path}")
#         else:
#             plt.show()
#
#     # 添加目标值（Objective）特征
#     if 'objective' in feature_type:
#         for i in range(population_size):
#             features[i].append(population[i].objective)
#
#     # 将特征标准化
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)
#
#     # 为每个个体设置特征
#     for i, ind in enumerate(population):
#         ind.set_feature(features[i])

class Cluster:
    """
    Individual 被聚类后形成 Cluster,
    Cluster中的algorithm \code\ objective均是最优的那个individual的对应属性
    """

    def __init__(self, clusterno, clustersize):
        self.cluster_no = clusterno
        self.population = []
        self.cluster_size = clustersize
        self.algorithm = ""
        self.objective = None
        self.code = ""
        self.offspring = []
        self.best_individual = None

    def init_best_individual(self):
        """
        选择population中objective值最低的individual作为Cluster中的最佳个体
        """
        if not self.population:
            raise ValueError("Population is empty. Cannot determine the best individual.")

        # 通过最小化objective值来选择最优个体
        best_individual = min(self.population, key=lambda individual: individual.objective)

        # 更新Cluster的属性为最佳个体的对应属性
        self.best_individual = best_individual
        self.objective = best_individual.objective
        self.algorithm = best_individual.algorithm
        self.code = best_individual.code

    def clear_offspring(self):
        self.offspring = []

    def add_offspring(self, aoffspring):
        self.offspring.append(aoffspring)

    def add_individual(self, new_individual):
        if isinstance(new_individual, dict):
            candidate = Individual()
            candidate.create_individual(ind_in_dict=new_individual)
            candidate.set_cluster(self)
            self.population.append(candidate)
        elif isinstance(new_individual, Individual):
            new_individual.set_cluster(self.cluster_no)
            self.population.append(new_individual)
        else:
            raise ValueError("Invalid individual type, Just dict and Individual are allowed")

    def choose_individual(self, m, strategy="fitness", option='re'):
        fitness_values = [ind.objective for ind in self.population]
        if option == 'lge':
            return self.population
        if m > self.cluster_size:
            print(f"警告: 请求的m={m}大于当前簇中的个体数量（{self.cluster_size}）。将返回所有个体。")
            return self.population  # 返回所有个体
        if strategy == "random":
            return random_selection(self.population, m)
        elif strategy == "fitness":
            if fitness_values is None:
                raise ValueError("Fitness values are required for fitness-based selection.")
            return fitness_proportional_selection(self.population, fitness_values, m)
        elif strategy == "rank":
            if fitness_values is None:
                raise ValueError("Fitness values are required for rank-based selection.")
            return rank_based_selection(self.population, fitness_values, m)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

    def management(self, maximize=False):
        """
        用轮盘赌的方式维护 Cluster 的大小，使其与 self.cluster_size 一致。
        更新最优个体（self.best_algo, self.best_fitness, self.best_code）。
        :param maximize: True 表示适应度值越大越好；False 表示适应度值越小越好。
        :return: None
        """
        # 假设每个个体有 `objective` 属性作为适应度值
        fitness_values = [ind.objective for ind in self.population]

        # 调整种群大小到 cluster_size
        if len(self.population) > self.cluster_size:
            self.population = fitness_proportional_selection(self.population, fitness_values, self.cluster_size,
                                                             maximize=maximize)

        # 更新最优个体信息（根据原始适应度值选择）
        if maximize:
            best_individual = max(self.population, key=lambda ind: ind.objective)
        else:
            best_individual = min(self.population, key=lambda ind: ind.objective)

        self.algorithm = best_individual.algorithm
        self.objective = best_individual.objective
        self.code = best_individual.code
        self.best_individual = best_individual

    def __iter__(self):
        self._iter_index = 0  # 每次迭代时重置索引
        return self

    def __next__(self):
        if self._iter_index < len(self.population):
            individual = self.population[self._iter_index]
            self._iter_index += 1
            return individual
        else:
            raise StopIteration  # 当迭代结束时抛出异常


# def individual_cluster(population, K):
#     """
#     基于K-means的种群聚类, 聚类后的簇对象应该使用Cluster类，参与聚类的是individual，每个individual中存在特征可以用来聚类
#     :param population: 个体列表
#     :param K: 聚类的簇数
#     :return: 聚类后的簇对象列表
#     """
#     # 提取每个个体的特征
#     features = []
#     for ind in population:
#         features.append(ind.feature)
#
#     # 将特征转换为NumPy数组，方便KMeans处理
#     features = np.array(features)
#
#     # 使用KMeans进行聚类
#     kmeans = KMeans(n_clusters=K, random_state=2024)
#     kmeans.fit(features)
#
#     # 获取每个个体的簇标签
#     labels = kmeans.labels_
#
#     # 根据标签创建Cluster对象
#     clusters = [Cluster(clusterno=i, clustersize=0) for i in range(K)]
#
#     # 将个体分配到对应的簇
#     for i, ind in enumerate(population):
#         cluster_no = labels[i]
#         clusters[cluster_no].population.append(ind)
#         clusters[cluster_no].cluster_size += 1  # 更新簇的大小
#
#     for cluster in clusters:
#         cluster.init_best_individual()
#     return clusters


def individual_cluster(population, K, clustering_algorithm='KMeans'):
    """
    基于不同聚类算法对种群进行聚类，聚类后的簇对象应该使用Cluster类，参与聚类的是individual，每个individual中存在特征可以用来聚类
    :param population: 个体列表
    :param K: 聚类的簇数
    :param clustering_algorithm: 使用的聚类算法，默认是KMeans
    :return: 聚类后的簇对象列表
    """
    # 提取每个个体的特征
    features = [ind.feature for ind in population]

    # 将特征转换为NumPy数组，方便聚类处理
    features = np.array(features)

    # 根据选择的聚类算法进行聚类
    if clustering_algorithm == 'KMeans':
        cluster_model = KMeans(n_clusters=K, random_state=2024)
    elif clustering_algorithm == 'GMM':
        cluster_model = GaussianMixture(n_components=K, random_state=2024)
    elif clustering_algorithm == 'Spectral':
        cluster_model = SpectralClustering(n_clusters=K, affinity='nearest_neighbors')
    else:
        raise ValueError(f"Unsupported clustering algorithm: {clustering_algorithm}")

    # 使用选定的聚类算法进行拟合
    cluster_model.fit(features)

    # 获取每个个体的簇标签
    if clustering_algorithm == 'GMM':
        # GMM 使用 predict 获取每个点的标签
        labels = cluster_model.predict(features)
    else:
        # KMeans 和 Spectral Clustering 使用 labels_ 获取标签
        labels = cluster_model.labels_

    # 根据标签创建Cluster对象
    clusters = [Cluster(clusterno=i, clustersize=0) for i in range(K)]

    # 将个体分配到对应的簇
    for i, ind in enumerate(population):
        cluster_no = labels[i]
        clusters[cluster_no].population.append(ind)
        clusters[cluster_no].cluster_size += 1  # 更新簇的大小

    # 初始化每个簇的最佳个体
    for cluster in clusters:
        cluster.init_best_individual()

    return clusters


def get_random_cooperator_clusters(input_list, index, m):
    if m > len(input_list):
        raise ValueError("m must be less than or equal to the total number of elements.")

    # 获取所有元素的索引
    choices = list(range(len(input_list)))

    # 从choices中排除指定的索引
    choices.remove(index)

    # 随机选择m-1个与index不同的索引
    selected_indices = random.sample(choices, m)

    # 将输入索引添加到返回的索引列表的最前面
    # selected_indices.insert(0, index)

    # 根据索引返回对应的元素
    # print("The indexs seleted are", selected_indices)
    return [input_list[i] for i in selected_indices]


def random_selection(population, m):
    """随机从种群中选择m个个体"""
    return random.sample(population, m)  # random.sample 默认不会有重复


# def fitness_proportional_selection(population, fitness_values, m, maximize=False):
#     """
#     根据适应度值选择个体。
#     :param population: 种群列表
#     :param fitness_values: 每个个体对应的适应度值
#     :param m: 需要选择的个体数量
#     :param maximize: True 表示适应度值越大概率越大；False 表示适应度值越小概率越大
#     :return: 被选中的个体列表
#     """
#     # 如果适应度值越小概率越大，则反转适应度值
#     if not maximize:
#         max_fitness = max(fitness_values) + 1  # 确保所有值正数
#         fitness_values = [max_fitness - f for f in fitness_values]
#
#     total_fitness = sum(fitness_values)
#     if total_fitness == 0:
#         raise ValueError("Total fitness is zero, cannot perform selection.")
#
#     probabilities = [f / total_fitness for f in fitness_values]
#
#     # 使用 random.choices 会允许重复选择，改用 random.sample 来避免重复
#     selected_indices = random.choices(range(len(population)), weights=probabilities, k=m)
#
#     # 使用 random.sample 来确保没有重复
#     selected_indices = random.sample(selected_indices, m)
#
#     return [population[i] for i in selected_indices]

def fitness_proportional_selection(population, fitness_values, m, maximize=False):
    """
    根据适应度值选择个体。
    :param population: 种群列表
    :param fitness_values: 每个个体对应的适应度值
    :param m: 需要选择的个体数量
    :param maximize: True 表示适应度值越大概率越大；False 表示适应度值越小概率越大
    :return: 被选中的个体列表
    """
    # 如果适应度值越小概率越大，则反转适应度值
    if not maximize:
        max_fitness = max(fitness_values) + 1  # 确保所有值为正数
        fitness_values = [max_fitness - f for f in fitness_values]

    # 确保适应度值和种群长度一致
    if len(population) != len(fitness_values):
        raise ValueError("Population and fitness values must have the same length.")

    # 转换为 NumPy 数组
    population = np.array(population)
    fitness_values = np.array(fitness_values)

    # 如果有零或负的适应度值，替换为最小值（可以选择更合适的处理方法）
    if np.any(fitness_values <= 0):
        fitness_values = np.clip(fitness_values, a_min=1e-6, a_max=None)

    # 计算选择概率
    probabilities = fitness_values / fitness_values.sum()

    # 检查是否有 NaN 值，并进行处理
    if np.any(np.isnan(probabilities)):
        print("NaN detected in probabilities!")
        probabilities = np.nan_to_num(probabilities, nan=0.0)  # 替换 NaN 为 0

    # 使用 NumPy 的无放回加权抽样
    selected_indices = np.random.choice(len(population), size=m, replace=False, p=probabilities)

    return population[selected_indices].tolist()


def rank_based_selection(population, fitness_values, m):
    """根据排序分配概率选择m个个体"""
    sorted_population = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    ranks = range(1, len(population) + 1)
    probabilities = [1 / rank for rank in ranks]
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    # 使用 random.choices 会允许重复选择，改用 random.sample 来避免重复
    selected_indices = random.choices(range(len(sorted_population)), weights=probabilities, k=m)

    # 使用 random.sample 来确保没有重复
    selected_indices = random.sample(selected_indices, m)

    return [sorted_population[i][0] for i in selected_indices]


def main():
    # 生成模拟种群
    population = []
    num_individuals = 10
    for i in range(num_individuals):
        ind = Individual(i)
        ind.create_individual(
            algorithm=f"Algorithm_{i}",
            code=f"""
def add(a, b):
    return a + b
""",
            objective=random.uniform(0, 1)  # 随机适应度
        )
        population.append(ind)

    # 为每个个体计算特征
    individual_feature(population, feature_type=['AST', 'objective'])

    # 设置聚类数K
    K = 3

    # 聚类
    clusters = individual_cluster(population, K)

    # 输出每个簇的结果
    for cluster in clusters:
        print(f"Cluster {cluster.cluster_no}:")
        print(f"Cluster size: {cluster.cluster_size}")
        print("Best Individual in Cluster:")
        print("Members:")
        for ind in cluster.population:
            print(f"  Individual {ind.individual_no}: {ind.algorithm}, {ind.objective}")
        print("\n" + "-" * 50 + "\n")

    selected_individuals = clusters[0].choose_individual(2, strategy="fitness")
    # "random""fitness"
    print(selected_individuals)


# 调用主函数
if __name__ == '__main__':
    main()
