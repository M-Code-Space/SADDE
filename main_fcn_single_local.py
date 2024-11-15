import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import yaml
import argparse
import time
from DataFactory import DataFactory
from model.LSTMClassifier import LSTMClassifier
from model.FCNClassifier import FCNClassifier
# from model.DeepCluster import DeepCluster
from model.DeepClusterDBScan import DeepCluster
from interpreter.LocalInterpreter import LocalInterpreter
from interpreter.GlobalInterpreter import GlobalInterpreter
from train import *
from test import *
from utils import *


def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Create a directory for saving experiment results
    experiment_dir = os.path.join('experiments/fcn', time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(experiment_dir, exist_ok=True)

    # Create a model directory
    model_dir = os.path.join(experiment_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    experiment_info = {
        'deep_model_metrics': [],
        'deep_cluster_metrics': []
    }

    top_n = config["top_n"]
    overlap_percentages_threshold = config["overlap_percentages_threshold"]

    print("############################################  First Round Training  ############################################")
    # select device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Select Device: ", device)

    # Load and preprocess the dataset
    data_factory = DataFactory()
    pretrain_loader, train_loader, val_loader, test_loader, input_size, feature_names = data_factory.load_data(config["dataset"], config["dataset_path"], config["pretrain_end"], config["test_start"], config["split_num"], train_batch_size=config["train_batch_size"], enumerate_batch_size=config["enumerate_batch_size"])
    # pretrain_loader, train_loader, val_loader, test_loader, input_size, feature_names = data_factory.load_data("CICDarknet", "Data/DarkNet/Darknet.CSV")
    # pretrain_loader, train_loader, test_loader, input_size, feature_names = data_factory.load_data("CICDarknet_fcn_loader")

    # 打印三个loader的标签分布
    print_label_distribution(pretrain_loader, "Pretrain Loader")
    print_label_distribution(train_loader, "Train Loader")
    print_label_distribution(test_loader, "Test Loader")

    # Deep Learning Model Training and Testing
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]  # 对FCN来说，这表示层数
    output_size = config["output_size"]
    num_classes = config["num_classes"]
    deep_model = FCNClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, num_classes=num_classes, dropout_rate=config["dropout_rate"], use_batch_norm=config["use_batch_norm"])
    deep_model = deep_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(deep_model.parameters(), lr=config["lr"])

    print("First round Training LSTM model...")
    train_model(deep_model, pretrain_loader, val_loader, criterion, optimizer, num_epochs=config["num_epochs"], patience=config["patience"], device=device)

    # Save the initial model
    torch.save(deep_model.state_dict(), os.path.join(model_dir, 'deep_model_initial.pth'))

    print("Extracting features for DeepCluster...")
    all_features = []
    all_labels = []
    original_data = []

    for data, labels in pretrain_loader:
        data = data.to(device)
        features = deep_model.extract_features(data)
        all_features.append(features.cpu().detach().numpy())
        all_labels.append(labels.cpu().numpy())
        original_data.append(data.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    original_data = np.concatenate(original_data, axis=0)

    # 使用 DeepCluster 进行初始聚类
    print("Initializing clustering with custom centers based on labels...")
    deep_cluster = DeepCluster(n_clusters=num_classes)
    deep_cluster.initial(features=all_features, labels=all_labels,
                         original_data=original_data)  # 使用 features 和 labels 初始化聚类

    # 初始化 LocalInterpreter
    print("Initializing Local Interpreter...")
    # local_interpreter = LocalInterpreter(deep_model, original_data, all_labels)

    # 获取唯一的标签列表
    unique_labels = np.unique(all_labels)

    # 创建一个字典来存储每个标签对应的 LocalInterpreter 实例
    local_interpreters = {}

    # # 根据每个标签初始化对应的 LocalInterpreter 实例
    # for label in unique_labels:
    #     label_mask = (all_labels == label)
    #     label_data = original_data[label_mask]
    #     label_model = deep_model  # 如果每个标签使用不同的模型，这里可以替换为相应的模型
    #     local_interpreters[label] = LocalInterpreter(label_model, label_data, label)

    # 替换为单一的LocalInterpreter
    print("Initializing Local Interpreter...")
    # local_interpreter = LocalInterpreter(deep_model, original_data, np.random.choice([0, 1], size=3, p=[0.99, 0.01]))
    local_interpreter = LocalInterpreter(deep_model, original_data, all_labels)
    # local_interpreter = LocalInterpreter(deep_model, original_data, all_labels)

    # 初始化 GlobalInterpreter
    print("Initializing Global Interpreter...")
    global_interpreter = GlobalInterpreter(deep_model, deep_cluster)

    print("############################################  Testing Before Training  ############################################")
    metrics, labels_before, predictions_before = evaluate_model(deep_model, test_loader, device)
    print(f"Deep Model Before Training: ")
    print(metrics)
    experiment_info['deep_model_metrics'].append(metrics)

    metrics, labels_before, predictions_before = evaluate_deep_cluster(deep_cluster, deep_model, test_loader, device)
    print(f"Deep Cluster Before Training: ")
    print(metrics)
    experiment_info['deep_cluster_metrics'].append(metrics)

    print("############################################  Training  ############################################")
    for i, (data, labels) in enumerate(train_loader):
        print("Training Batch ", i)
        data, labels = data.to(device), labels.to(device)

        retrain_dataset = []
        original_dataset = []
        global_important_feature_indices_dict = {}

        # LSTM 进行分类
        deep_model.eval()
        with torch.no_grad():
            lstm_outputs = deep_model(data)
            features = deep_model.extract_features(data)

        # DeepCluster 进行分类
        deep_cluster_labels = deep_cluster.predict(features.cpu().detach().numpy())

        # 解释每个聚类
        for cluster_label in range(num_classes):
            print(f"Interpreting Cluster {cluster_label}...")
            global_important_feature_indices, important_features_scores = global_interpreter.interpret_pfi(cluster_label, feature_names=None, top_n=top_n, device=device)
            # global_important_feature_indices, important_features_scores = global_interpreter.interpret_shap(cluster_label, feature_names=None, top_n=top_n)
            global_important_feature_indices_dict[cluster_label] = global_important_feature_indices
            print(f"Explanation for Cluster {cluster_label}: {global_important_feature_indices}")

        # 本地解释器解释 Deep Model 的结果
        # 按标签计算重合度
        matching_samples_mask = np.zeros(len(deep_cluster_labels), dtype=bool)

        for u_label in unique_labels:
            # 选择当前标签的样本
            label_mask = (deep_cluster_labels == u_label)

            if np.sum(label_mask) == 0:
                continue  # 如果没有匹配的样本，跳过

            # 获取当前标签对应的 LocalInterpreter 实例
            interpreter = local_interpreter

            # 使用该实例进行解释
            local_indices_group, _ = interpreter.interpret_deeplift(data[label_mask].cpu().numpy(), u_label, top_n=top_n, device=device)

            # 获取当前标签的 global_important_feature_indices
            global_indices = global_important_feature_indices_dict[u_label]


            # 更新整体的 matching_samples_mask
            matching_samples_mask[label_mask] = label_matching_mask

        # 选出满足条件的样本数据和对应的 deep_cluster_prediction
        matched_data = data[matching_samples_mask].cpu().numpy()
        matched_predictions = deep_cluster_labels[matching_samples_mask]

        # 将匹配的样本添加到 retrain_dataset 和 original_dataset
        retrain_dataset.extend(zip(matched_data, matched_predictions.reshape(-1, 1)))
        original_dataset.extend(matched_data)


        # 将 retrain_dataset 用于下一轮训练
        if retrain_dataset:
            original_dataset = np.asarray(original_dataset)
            retrain_dataset = list(zip(*retrain_dataset))
            retrain_data = torch.tensor(np.array(retrain_dataset[0]), dtype=torch.float32).to(device)
            retrain_labels = torch.tensor(np.array(retrain_dataset[1]).squeeze(), dtype=torch.long).to(device)

            retrain_loader = DataLoader(TensorDataset(retrain_data, retrain_labels), batch_size=32,
                                        shuffle=False)

            print(f"Retrain dataset size: {len(retrain_data)}")

            # 重新训练 LSTM
            deep_model.train()
            train_model(deep_model, retrain_loader, val_loader, criterion, optimizer, num_epochs=config["num_epochs"], patience=config["patience"], device=device)

            # 使用新的特征更新 KMeans 质心
            retrain_features = []
            for retrain_data_batch, _ in retrain_loader:
                retrain_features_batch = deep_model.extract_features(retrain_data_batch)
                retrain_features.append(retrain_features_batch.cpu().detach().numpy())

            retrain_features = np.concatenate(retrain_features, axis=0)
            retrain_labels = retrain_labels.cpu().numpy()

            # 更新DeepCluster的质心
            deep_cluster.fit(retrain_features, retrain_labels, original_dataset)

        print("############################################  Testing During Training  ############################################")
        metrics, labels_before, predictions_before = evaluate_model(deep_model, test_loader, device)
        print(f"Deep Model During Training: ")
        print(metrics)
        experiment_info['deep_model_metrics'].append(metrics)

        metrics, labels_before, predictions_before = evaluate_deep_cluster(deep_cluster, deep_model, test_loader,
                                                                           device)
        print(f"Deep Cluster During Training: ")
        print(metrics)
        experiment_info['deep_cluster_metrics'].append(metrics)

        # 保存当前批次训练好的model
        torch.save(deep_model.state_dict(), os.path.join(model_dir, f'deep_model_{i}.pth'))

    print("############################################  Testing After Training  ############################################")
    metrics, labels_before, predictions_before = evaluate_model(deep_model, test_loader, device)
    print(f"Deep Model After Training: ")
    print(metrics)
    experiment_info['deep_model_metrics'].append(metrics)

    metrics, labels_before, predictions_before = evaluate_deep_cluster(deep_cluster, deep_model, test_loader, device)
    print(f"Deep Cluster After Training: ")
    print(metrics)
    experiment_info['deep_cluster_metrics'].append(metrics)

    save_experiment_info(experiment_dir, experiment_info)
    save_predictions(experiment_dir, labels_before, predictions_before)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model based on YAML configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    main(args.config)

