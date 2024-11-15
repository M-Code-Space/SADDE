import json
import os

import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, \
    average_precision_score, roc_auc_score
from collections import Counter
import numpy as np

def print_label_distribution(loader, loader_name):
    all_labels = []

    # 遍历 loader 获取所有标签
    for _, labels in loader:
        all_labels.extend(labels.numpy())  # 将张量转换为 numpy 数组并添加到列表中

    # 统计每个标签的数量
    label_distribution = Counter(all_labels)

    print(f"Label distribution in {loader_name}: {label_distribution}")
    return label_distribution



def evaluate_model(model, data_loader, device="cuda"):
    all_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 从混淆矩阵计算 FPR 和 TPR
    tn, fp, fn, tp = cm.ravel()  # 提取混淆矩阵中的元素
    # # 如果标签反了，交换 tn 和 tp，fp 和 fn
    # tn, fp, fn, tp = tp, fn, fp, tn

    # 计算指标
    TPR = tp / (tp + fn)  # True Positive Rate, 召回率
    TNR = tn / (tn + fp)  # True Negative Rate, 特异性
    PPV = tp / (tp + fp)  # Positive Predictive Value, 精确度
    NPV = tn / (tn + fn)  # Negative Predictive Value
    FPR = fp / (fp + tn)  # False Positive Rate
    FNR = fn / (fn + tp)  # False Negative Rate
    FDR = fp / (fp + tp)  # False Discovery Rate
    ACC = (tp + tn) / (tp + tn + fp + fn)  # Accuracy

    AUC = roc_auc_score(all_labels, all_predictions)
    max_fpr = 0.05  # DarkNet
    # 计算在max_fpr下的原始部分AUC (pAUC)
    pAUC = roc_auc_score(all_labels, all_predictions, max_fpr=max_fpr)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'f1_score_macro': f1_score(all_labels, all_predictions, average='macro'),
        'f1_score_micro': f1_score(all_labels, all_predictions, average='micro'),
        'f1_score_weighted': f1_score(all_labels, all_predictions, average='weighted'),
        'f1_score_binary': f1_score(all_labels, all_predictions, average='binary'),
        'recall_macro': recall_score(all_labels, all_predictions, average='macro'),
        'recall_micro': recall_score(all_labels, all_predictions, average='micro'),
        'recall_weighted': recall_score(all_labels, all_predictions, average='weighted'),
        'recall_binary': recall_score(all_labels, all_predictions, average='binary'),
        'precision_macro': precision_score(all_labels, all_predictions, average='macro'),
        'precision_micro': precision_score(all_labels, all_predictions, average='micro'),
        'precision_weighted': precision_score(all_labels, all_predictions, average='weighted'),
        'precision_binary': precision_score(all_labels, all_predictions, average='binary'),
        'average_precision_macro': average_precision_score(all_labels, all_predictions, average='macro'),
        'average_precision_micro': average_precision_score(all_labels, all_predictions, average='micro'),
        'average_precision_weighted': average_precision_score(all_labels, all_predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),  # Convert to list for JSON serialization
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'NPV': NPV,
        'FPR': FPR,
        'FNR': FNR,
        'FDR': FDR,
        'AUC': AUC,
        'pAUC': pAUC,
    }

    return metrics, all_labels, all_predictions

def evaluate_deep_cluster(deep_cluster, model, data_loader, device="cuda"):
    all_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            # 使用模型提取特征
            features = model.extract_features(data)
            # 使用 deep cluster 进行预测
            cluster_predictions = deep_cluster.predict(features.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(cluster_predictions)

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)

        # 从混淆矩阵计算 FPR 和 TPR
        tn, fp, fn, tp = cm.ravel()  # 提取混淆矩阵中的元素
        # # 如果标签反了，交换 tn 和 tp，fp 和 fn
        # tn, fp, fn, tp = tp, fn, fp, tn

        # 计算指标
        TPR = tp / (tp + fn)  # True Positive Rate, 召回率
        TNR = tn / (tn + fp)  # True Negative Rate, 特异性
        PPV = tp / (tp + fp)  # Positive Predictive Value, 精确度
        NPV = tn / (tn + fn)  # Negative Predictive Value
        FPR = fp / (fp + tn)  # False Positive Rate
        FNR = fn / (fn + tp)  # False Negative Rate
        FDR = fp / (fp + tp)  # False Discovery Rate
        ACC = (tp + tn) / (tp + tn + fp + fn)  # Accuracy

        AUC = roc_auc_score(all_labels, all_predictions)
        max_fpr = 0.05  # DarkNet
        # 计算在max_fpr下的原始部分AUC (pAUC)
        pAUC = roc_auc_score(all_labels, all_predictions, max_fpr=max_fpr)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'f1_score_macro': f1_score(all_labels, all_predictions, average='macro'),
            'f1_score_micro': f1_score(all_labels, all_predictions, average='micro'),
            'f1_score_weighted': f1_score(all_labels, all_predictions, average='weighted'),
            'recall_macro': recall_score(all_labels, all_predictions, average='macro'),
            'recall_micro': recall_score(all_labels, all_predictions, average='micro'),
            'recall_weighted': recall_score(all_labels, all_predictions, average='weighted'),
            'precision_macro': precision_score(all_labels, all_predictions, average='macro'),
            'precision_micro': precision_score(all_labels, all_predictions, average='micro'),
            'precision_weighted': precision_score(all_labels, all_predictions, average='weighted'),
            'average_precision_macro': average_precision_score(all_labels, all_predictions, average='macro'),
            'average_precision_micro': average_precision_score(all_labels, all_predictions, average='micro'),
            'average_precision_weighted': average_precision_score(all_labels, all_predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            # Convert to list for JSON serialization
            'TPR': TPR,
            'TNR': TNR,
            'PPV': PPV,
            'NPV': NPV,
            'FPR': FPR,
            'FNR': FNR,
            'FDR': FDR,
            'AUC': AUC,
            'pAUC': pAUC,
        }

    return metrics, all_labels, all_predictions

def save_experiment_info(experiment_dir, info):
    with open(os.path.join(experiment_dir, 'experiment_info.json'), 'w') as f:
        json.dump(info, f, indent=4)

def save_predictions(experiment_dir, predictions, targets):
    torch.save({'predictions': predictions, 'targets': targets}, os.path.join(experiment_dir, 'predictions.pt'))