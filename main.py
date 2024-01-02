import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import Dataset
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import ndcg_score


def encode_categorical_features(feature: pd.Series) -> np.ndarray:
    """
    将分类特征编码为整数。

    参数:
    feature (pd.Series): 待编码的分类特征。

    返回:
    np.ndarray: 编码后的整数数组。
    """
    if feature.dtype == object:
        encoder = LabelEncoder()
        feature = encoder.fit_transform(feature)
    return feature


def process_checkin_times(checkin_times: pd.Series) -> (pd.Series, pd.Series, pd.Series, pd.Series):
    """
    处理签到时间，提取年、月、日、小时。

    参数:
    checkin_times (pd.Series): 签到时间的序列。

    返回:
    tuple: 包含年、月、日、小时的序列。
    """
    checkin_datetimes = pd.to_datetime(checkin_times)
    years = checkin_datetimes.dt.year
    months = checkin_datetimes.dt.month
    days = checkin_datetimes.dt.day
    hours = checkin_datetimes.dt.hour
    return years, months, days, hours


def convert_to_sparse_matrix(user_ids: pd.Series, venue_ids: pd.Series, categories: pd.Series,
                             checkin_times: pd.Series, latitudes: pd.Series, longitudes: pd.Series) -> csr_matrix:
    """
    将用户ID、场所ID、类别、签到时间、纬度和经度转换为稀疏矩阵。

    参数:
    user_ids (pd.Series): 用户ID序列。
    venue_ids (pd.Series): 场所ID序列。
    categories (pd.Series): 类别序列。
    checkin_times (pd.Series): 签到时间序列。
    latitudes (pd.Series): 纬度序列。
    longitudes (pd.Series): 经度序列。

    返回:
    csr_matrix: 组合后的稀疏矩阵。
    """
    # 处理签到时间并编码
    years, months, days, hours = process_checkin_times(checkin_times)
    encoded_user_ids = encode_categorical_features(user_ids)
    encoded_venue_ids = encode_categorical_features(venue_ids)
    encoded_categories = encode_categorical_features(categories)
    encoded_years = encode_categorical_features(years)
    encoded_months = encode_categorical_features(months)
    encoded_days = encode_categorical_features(days)
    encoded_hours = encode_categorical_features(hours)

    # 创建稀疏矩阵
    sparse_matrix_list = [csr_matrix((np.ones(len(feature)), (np.arange(len(feature)), feature)),
                                     shape=(len(feature), feature.max() + 1))
                          for feature in [encoded_user_ids, encoded_venue_ids, encoded_categories,
                                          encoded_years, encoded_months, encoded_days, encoded_hours]]

    # 地理空间聚类并编码（假设cluster_venues函数已定义）
    cluster_labels = cluster_venues(latitudes, longitudes, n_clusters=10)
    encoded_cluster_labels = encode_categorical_features(cluster_labels)

    # 加入聚类结果
    sparse_matrix_list.append(csr_matrix((np.ones(len(encoded_cluster_labels)),
                                          (np.arange(len(encoded_cluster_labels)), encoded_cluster_labels))))

    # 组合所有特征为一个稀疏矩阵
    final_sparse_matrix = hstack(sparse_matrix_list)
    return final_sparse_matrix


def create_labels_for_areas(tuesday_data: pd.DataFrame) -> (pd.Series, LabelEncoder):
    """
    对给定数据集中的场所ID进行编码。

    参数:
    tuesday_data (pd.DataFrame): 包含场所ID ('venueId') 的数据帧。

    返回:
    tuple:
    - 第一个元素为编码后的场所标签（pd.Series）。
    - 第二个元素为用于编码的LabelEncoder实例。
    """
    venue_encoder = LabelEncoder()
    encoded_labels = venue_encoder.fit_transform(tuesday_data['venueId'])
    return encoded_labels, venue_encoder


def cluster_venues(latitudes: np.ndarray, longitudes: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """
    使用K均值聚类算法对地点进行聚类。

    参数:
    latitudes (np.ndarray): 地点的纬度数组。
    longitudes (np.ndarray): 地点的经度数组。
    n_clusters (int): 聚类的数量，默认为10。

    返回:
    np.ndarray: 聚类结果的标签数组。
    """
    coords = np.column_stack((latitudes, longitudes))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    return kmeans.labels_


def evaluate_model(model: torch.nn.Module, X: np.ndarray, y_true: np.ndarray) -> (float, float, float, float):
    """
    对模型进行评估，计算准确率、召回率、平均精度和归一化折扣累计增益。

    参数:
    model (torch.nn.Module): 要评估的模型。
    X (np.ndarray): 输入特征数据。
    y_true (np.ndarray): 真实的标签。

    返回:
    tuple: 包含准确率、召回率、平均精度和NDCG的四元组。
    """
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X.toarray()))
        predicted_scores = predictions.numpy()
        accuracy = accuracy_score(y_true, np.argmax(predicted_scores, axis=1))
        recall = recall_score(y_true, np.argmax(predicted_scores, axis=1), average='macro', zero_division=0)
        map_score = calculate_map(y_true, predicted_scores)
        ndcg_score_value = calculate_ndcg(y_true, predicted_scores, k=3)
        return accuracy, recall, map_score, ndcg_score_value


def calculate_ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """
    计算归一化折扣累计增益 (NDCG)。

    参数:
    y_true (np.ndarray): 真实的标签。
    y_score (np.ndarray): 模型预测的分数。
    k (int): 计算NDCG时考虑的顶部k个元素。

    返回:
    float: NDCG分数。
    """
    y_true_binary = np.zeros_like(y_score)
    rows = np.arange(len(y_true))
    y_true_binary[rows, y_true] = 1
    return ndcg_score(y_true_binary, y_score, k=k)


def calculate_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算平均精度均值 (MAP)。

    参数:
    y_true (np.ndarray): 真实的标签。
    y_score (np.ndarray): 模型预测的分数。

    返回:
    float: MAP分数。
    """
    y_true_binary = np.zeros_like(y_score)
    rows = np.arange(len(y_true))
    y_true_binary[rows, y_true] = 1
    return label_ranking_average_precision_score(y_true_binary, y_score)


def prepare_data(file_path: str):
    """
    准备和处理数据。

    参数:
    file_path (str): 数据文件的路径。

    返回:
    tuple: 包含训练和测试集的元组。
    """
    data = pd.read_csv(file_path)
    data['utcTimestamp'] = pd.to_datetime(data['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')
    tuesday_data = data[(data['utcTimestamp'].dt.weekday == 1) & (data['utcTimestamp'].dt.hour >= 18)]

    labels, venue_encoder = create_labels_for_areas(tuesday_data)
    X = convert_to_sparse_matrix(tuesday_data['userId'], tuesday_data['venueId'], tuesday_data['venueCategory'],
                                 tuesday_data['utcTimestamp'], tuesday_data['latitude'], tuesday_data['longitude'])

    return train_test_split(X, labels, test_size=0.2, random_state=42), venue_encoder


def train_model(model, criterion, optimizer, scheduler, X_train, y_train, X_test, y_test, epochs: int):
    """
    训练模型并评估其性能。

    参数:
    model: 训练的模型。
    criterion: 损失函数。
    optimizer: 优化器。
    scheduler: 学习率调度器。
    X_train, y_train: 训练数据和标签。
    X_test, y_test: 测试数据和标签。
    epochs (int): 训练的周期数。

    返回:
    list: 训练过程中收集的指标。
    """
    best_score = 0.00
    metrics = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(torch.FloatTensor(X_train.toarray()))
        loss = criterion(output, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()
        scheduler.step()

        accuracy, recall, map_score, ndcg_score = evaluate_model(model, X_test, y_test)
        all_score = (recall + map_score + ndcg_score) / 3
        print(f"Epoch {epoch}/{epochs}, Accuracy: {accuracy}, Recall: {recall}, MAP: {map_score}, nDCG: {ndcg_score}")

        if all_score > best_score:
            best_score = all_score
            torch.save(model.state_dict(), './best.pth')

        metrics.append({
            'Epoch': epoch,
            'Loss': loss.item(),
            'Accuracy': accuracy,
            'Recall': recall,
            'MAP': map_score,
            'nDCG': ndcg_score
        })

    return metrics


class SparseMatrixDataset(Dataset):
    """
    稀疏矩阵数据集类，用于处理稀疏特征数据。
    """

    def __init__(self, sparse_matrix: np.ndarray, labels: np.ndarray):
        self.sparse_matrix = sparse_matrix
        self.labels = labels

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx: int):
        x = self.sparse_matrix[idx].toarray().astype(np.float32).squeeze()
        y = self.labels[idx]
        return x, y


class AttentionMechanism(nn.Module):
    """
    注意力机制模块。
    """

    def __init__(self, input_dim: int):
        super(AttentionMechanism, self).__init__()
        self.attention_fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention_fc(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights


class FactorizationMachine(nn.Module):
    """
    因子分解机模型，用于处理稀疏特征数据。
    """

    def __init__(self, n: int, k: int, num_classes: int):
        super(FactorizationMachine, self).__init__()
        self.n = n
        self.k = k
        self.num_classes = num_classes
        self.linear = nn.Linear(n, num_classes)
        self.v = nn.Parameter(torch.randn(n, k))
        self.attention = AttentionMechanism(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear(x)
        interaction_sum = torch.mm(x, self.v)
        interaction_sum_squared = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))

        # 应用注意力机制
        attention_weights = self.attention(x)
        weighted_interaction = attention_weights * (interaction_sum ** 2 - interaction_sum_squared)
        interaction_part = 0.5 * torch.sum(weighted_interaction, dim=1, keepdim=True)
        interaction_part = interaction_part.expand(x.size(0), self.num_classes)

        return torch.softmax(linear_part + interaction_part, dim=1)


if __name__ == '__main__':
    random.seed(42)

    # 数据准备
    (X_train, X_test, y_train, y_test), venue_encoder = prepare_data("./Restaurant/dataset_TSMC2014_TKY.csv")

    # 模型初始化
    model = FactorizationMachine(n=X_train.shape[1], k=15, num_classes=len(venue_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 模型训练
    metrics = train_model(model, criterion, optimizer, scheduler, X_train, y_train, X_test, y_test, epochs=100)

    # 保存结果
    pd.DataFrame(metrics).to_excel('./result2.xlsx', index=False)
