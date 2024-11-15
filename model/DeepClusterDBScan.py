import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from scipy.spatial.distance import cdist

class DeepCluster:
    def __init__(self, n_clusters, eps=0.5, min_samples=1):
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples

        # DBSCAN 模型
        self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
        # 聚类中心点
        self.cluster_centers_ = {}
        # 聚类映射关系
        self.label_mapping = {}
        # 聚类样本
        self.cluster_samples_ = {}
        # 聚类特征
        self.cluster_features_ = defaultdict(list)

    def initial(self, features, labels, original_data):
        """
        初始化聚类，使用features直接计算每个类的聚类中心
        :param features:
        :param labels:
        :param original_data:
        :return:
        """
        # 计算每个类的聚类中心
        unique_labels = np.unique(labels)

        # 遍历每个类的特征
        for label in unique_labels:
            # 计算每个类的特征均值
            cluster_center = features[labels == label].mean(axis=0)

            # 字典形式保存
            self.cluster_centers_[label] = cluster_center

            # 保存每个类对应的原始样本
            self.cluster_samples_[label] = original_data[labels == label]
            # 保存每个类对应的特征数据
            self.cluster_features_[label].extend(features[labels == label])



    def fit(self, features, labels, original_data):
        """
        更新 DBSCAN 中心点，特征和样本
        :param features:
        :param labels:
        :param original_data:
        :return:
        """
        for i in range(self.n_clusters):
            # 获取当前类的特征数据
            current_cluster_features = np.array(self.cluster_features_[i])
            new_cluster_features = features[labels == i]

            if len(new_cluster_features) > 0:
                # 合并新数据和历史数据
                updated_cluster_features = np.vstack((current_cluster_features, new_cluster_features))

                # 重新计算中心点
                new_center = updated_cluster_features.mean(axis=0)
                self.cluster_centers_[i] = new_center

                # 更新聚类的特征信息
                self.cluster_features_[i] = updated_cluster_features

                # 更新聚类的样本信息
                self.cluster_samples_[i] = np.vstack((self.cluster_samples_[i], original_data[labels == i]))

    def predict(self, features):
        """
        先对特征进行DBSCAN, 计算DBSCAN聚类中心到历史聚类中心的欧氏距离，选择最小距离的聚类中心
        :param features:
        :return:
        """
        # 重新初始化DBSCAN
        self.cluster_model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        # 进行聚类
        labels = self.cluster_model.fit_predict(features)
        # 计算每个类的聚类中心
        unique_labels = np.unique(labels)
        new_centers = {}
        for label in unique_labels:
            new_centers[label] = features[labels == label].mean(axis=0)

        # 使用向量计算加速
        new_centers = np.array(list(new_centers.values()))
        # 计算新聚类中心和历史聚类中心的欧氏距离
        distances = cdist(new_centers, np.array(list(self.cluster_centers_.values())))
        # 选择最小距离的聚类中心
        new_labels = distances.argmin(axis=1)

        # 创建一个映射，将DBSCAN的聚类标签映射到新选择的最小距离历史中心
        label_mapping = {unique_labels[i]: new_labels[i] for i in range(len(unique_labels))}

        # 为每个样本重新分配聚类中心
        final_labels = np.array([label_mapping[label] if label != -1 else -1 for label in labels])

        return final_labels