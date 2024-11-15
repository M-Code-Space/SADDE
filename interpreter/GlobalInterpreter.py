from sklearn.inspection import permutation_importance
import torch
import shap
import numpy as np
from sklearn.metrics import accuracy_score

class SklearnWrapper:
    def __init__(self, model, deep_cluster, device='cpu'):
        self.model = model
        self.deep_cluster = deep_cluster
        self.device = device

    def fit(self, X, y=None):
        # 空的 fit 方法，满足 sklearn 接口要求
        pass

    # def predict(self, X):
    #     X = torch.tensor(X, dtype=torch.float32, device=self.device)
    #     with torch.no_grad():
    #         features = self.model.extract_features(X)
    #     return self.deep_cluster.cluster_model.predict(features.cpu().numpy())

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            features = self.model.extract_features(X)
        return self.deep_cluster.predict(features.cpu().numpy())

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class GlobalInterpreter:
    def __init__(self, model, deep_cluster):
        self.model = model
        self.deep_cluster = deep_cluster

    def interpret_pfi(self, cluster_label, feature_names=None, top_n=10, device='cpu', sampling=1000, n_repeats=100):
        """
        使用 DeepCluster 的聚类结果来解释特定聚类的特征重要性。
        """
        # 从 DeepCluster 中提取属于该聚类的样本
        cluster_data = np.array(self.deep_cluster.cluster_samples_[cluster_label])

        # 标签数组，所有样本的标签都是聚类标签
        cluster_labels = np.full(cluster_data.shape[0], cluster_label)

        # 采样
        if cluster_data.shape[0] > sampling:
            cluster_data = cluster_data[:sampling]
            cluster_labels = cluster_labels[:sampling]

        # 使用 SklearnWrapper 包装模型
        wrapped_model = SklearnWrapper(self.model, self.deep_cluster, device=device)

        # 计算特征重要性
        results = permutation_importance(wrapped_model, cluster_data, cluster_labels, n_repeats=n_repeats, random_state=42)

        # 取绝对值以忽略正负方向，专注于特征的重要性强度
        # importances = results.importances_mean
        importances = np.abs(results.importances_mean)

        # 获取前 n 个重要的特征
        important_indices = np.argsort(importances)[-top_n:][::-1]
        important_features_scores = importances[important_indices]

        if feature_names is not None:
            important_feature_names = [feature_names[i] for i in important_indices]
        else:
            important_feature_names = important_indices  # 如果未提供特征名，使用索引代替

        # 返回最重要的特征名及其重要性
        return important_feature_names, important_features_scores
