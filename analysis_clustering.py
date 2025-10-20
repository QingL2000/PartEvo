import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class Cluster:
    def __init__(self, data):
        self.data = data
        self.clusters = defaultdict(list)
        self.features = defaultdict(list)  # 用于存储每个簇的特征
        self._assign_to_clusters()

    def _assign_to_clusters(self):
        """根据每个个体的cluster信息，将其分配到相应的簇，并提取特征"""
        for item in self.data:
            cluster_id = item['cluster']
            self.clusters[cluster_id].append(item)
            # 提取并保存特征
            self.features[cluster_id].append(item['feature'])

    def get_cluster_count(self):
        """返回每个簇的个体个数"""
        cluster_counts = {cluster_id: len(items) for cluster_id, items in self.clusters.items()}
        return cluster_counts

    def get_cluster_members(self, cluster_id):
        """返回指定簇的成员"""
        return self.clusters.get(cluster_id, [])

    def get_cluster_features(self, cluster_id):
        """返回指定簇的特征"""
        return self.features.get(cluster_id, [])

    def evaluate_clustering(self):
        """评估聚类结果"""
        all_features = []
        labels = []

        # 获取所有个体的特征以及其对应的簇标签
        for idx, item in enumerate(self.data):
            all_features.append(item['feature'])
            labels.append(item['cluster'])

        all_features = np.array(all_features)
        labels = np.array(labels)

        # 计算评估指标
        silhouette = silhouette_score(all_features, labels)
        calinski_harabasz = calinski_harabasz_score(all_features, labels)
        davies_bouldin = davies_bouldin_score(all_features, labels)

        return {
            'Silhouette Score': silhouette,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Davies-Bouldin Index': davies_bouldin
        }


source = ['AST', 'language', 'random']
expes = ['single_cluster_information', 'multi_cluster_information', 'mec_cluster_information',
         'manu_cluster_information']

# 创建一个字典来存储各个source的评估结果
source_evaluation_results = {}

# 循环通过不同的source，读取数据并评估
for src in source:
    file_path = f'D:/00_Work/00_CityU/04_AEL_MEC/partevo_v2/results_for_clustering_analysis/{src}/{expes[2]}.json'

    # 读取JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 创建Cluster对象并进行评估
    cluster_obj = Cluster(data)

    # 获取聚类评估结果
    evaluation_results = cluster_obj.evaluate_clustering()

    # 将每个source的评估结果存储到字典中
    source_evaluation_results[src] = evaluation_results

# 打印每个source的评估结果
print("\nClustering Evaluation Results for Different Sources:")
for src, results in source_evaluation_results.items():
    print(f"\nResults for {src}:")
    for metric, score in results.items():
        print(f"{metric}: {score}")

# 排序：为每个指标进行排序，好的排在前面
sorted_results = {metric: sorted(source_evaluation_results.items(), key=lambda x: x[1][metric],
                                 reverse=True if metric != 'Davies-Bouldin Index' else False)
                  for metric in ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']}

# 打印排序结果
print("\nSorted Clustering Results by Each Metric:")
for metric, sorted_sources in sorted_results.items():
    print(f"\n{metric} Ranking:")
    for rank, (src, results) in enumerate(sorted_sources, 1):
        print(f"Rank {rank}: {src} with {metric}: {results[metric]}")
