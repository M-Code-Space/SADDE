import numpy as np
from captum.attr import DeepLift



class LocalInterpreter:
    # 问题： Lime只能处理二维数据
    def __init__(self, model, train_data, train_labels, mode='classification', device='cpu'):
        """
        初始化本地解释器。
        :param model: 要解释的模型 (LSTM 分类器)
        :param train_data: 用于初始化 LIME 的训练数据
        :param train_labels: 用于初始化 LIME 的训练数据标签
        :param mode: 模式，'classification' 或 'regression'
        """
        self.model = model
        self.sample_num = train_data.shape[0]
        self.num_features = train_data.shape[1]

        # DeepLift
        self.deeplift = DeepLift(self.model)


    def interpret_deeplift(self, data_point, target_class, top_n=10, device='cpu'):
        """
        使用DeepLIFT对单个数据点进行解释，返回最重要的前n个特征。
        :param data_point: 要解释的数据点，形状为[num_features]
        :param top_n: 返回最重要的前n个特征
        :return: 最重要的特征索引和对应的归因值
        """
        # 将数据点转换为tensor并添加batch维度
        data_point_tensor = torch.tensor(data_point, dtype=torch.float32, device=device)
        # 将预测的类别转换为PyTorch tensor，并确保类型为torch.long
        target_class = torch.tensor(target_class, dtype=torch.long, device=device)

        # 计算归因
        attributions = self.deeplift.attribute(data_point_tensor, target=target_class)

        # 将归因结果转换为NumPy数组
        attributions = attributions.detach().cpu().numpy()

        # 初始化存储结果的列表
        important_features = []
        important_features_indices = []

        # 对每个样本的归因结果进行处理，获取最重要的n个特征
        for i in range(attributions.shape[0]):
            sample_attributions = attributions[i]
            # 取绝对值以关注影响的强度
            abs_attributions = np.abs(sample_attributions)
            top_n_indices = np.argsort(abs_attributions)[-top_n:]
            top_n_importances = abs_attributions[top_n_indices]

            # 存储该样本的结果
            important_features.append(top_n_importances)
            important_features_indices.append(top_n_indices)

        return important_features_indices, important_features