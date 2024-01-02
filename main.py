import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix, hstack
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def encode_categorical_features(feature):
    if feature.dtype == object:
        encoder = LabelEncoder()
        feature = encoder.fit_transform(feature)
    return feature


def process_checkin_times(checkin_times):
    checkin_datetimes = pd.to_datetime(checkin_times)
    years = checkin_datetimes.dt.year
    months = checkin_datetimes.dt.month
    days = checkin_datetimes.dt.day
    hours = checkin_datetimes.dt.hour
    return years, months, days, hours


def convert_to_sparse_matrix(user_ids, venue_ids, categories, checkin_times):
    years, months, days, hours = process_checkin_times(checkin_times)
    sparse_matrix_list = []

    for feature in [user_ids, venue_ids, categories, years, months, days, hours]:
        feature = encode_categorical_features(feature)
        row_indices = np.arange(len(feature))
        col_indices = feature
        sparse_matrix = csr_matrix((np.ones(len(feature)), (row_indices, col_indices)),
                                   shape=(len(feature), np.max(col_indices) + 1))
        sparse_matrix_list.append(sparse_matrix)

    final_sparse_matrix = hstack(sparse_matrix_list)
    return final_sparse_matrix


def create_labels_for_areas(tuesday_data):
    venue_encoder = LabelEncoder()
    encoded_labels = venue_encoder.fit_transform(tuesday_data['venueId'])
    return encoded_labels, venue_encoder


class SparseMatrixDataset(Dataset):
    def __init__(self, sparse_matrix, labels):
        self.sparse_matrix = sparse_matrix
        self.labels = labels

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        x = self.sparse_matrix[idx].toarray().astype(np.float32).squeeze()
        y = self.labels[idx]
        return x, y


class FactorizationMachine(nn.Module):
    def __init__(self, n, k, num_classes):
        super(FactorizationMachine, self).__init__()
        self.n = n
        self.k = k
        self.num_classes = num_classes
        self.linear = nn.Linear(n, num_classes)
        self.v = nn.Parameter(torch.randn(n, k))

    def forward(self, x):
        linear_part = self.linear(x)
        interaction_sum = torch.mm(x, self.v)
        interaction_sum_squared = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        interaction_part = 0.5 * torch.sum(torch.pow(interaction_sum, 2) - interaction_sum_squared, dim=1, keepdim=True)
        interaction_part = interaction_part.expand(x.size(0), self.num_classes)
        # return linear_part + interaction_part
        return torch.softmax(linear_part + interaction_part, dim=1)


def calculate_ndcg(y_true, y_score, k=10):
    from sklearn.metrics import ndcg_score
    # Transform y_true to a binary array
    y_true_binary = np.zeros_like(y_score)
    rows = np.arange(len(y_true))
    y_true_binary[rows, y_true] = 1
    return ndcg_score(y_true_binary, y_score, k=k)


def calculate_map(y_true, y_score):
    from sklearn.metrics import label_ranking_average_precision_score
    # Transform y_true to a binary array
    y_true_binary = np.zeros_like(y_score)
    rows = np.arange(len(y_true))
    y_true_binary[rows, y_true] = 1
    return label_ranking_average_precision_score(y_true_binary, y_score)


if __name__ == '__main__':
    random.seed(42)
    data = pd.read_csv("./Restaurant/dataset_TSMC2014_NYC.csv")[:]
    data['utcTimestamp'] = pd.to_datetime(data['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')

    tuesday_data = data[(data['utcTimestamp'].dt.weekday == 1) & (data['utcTimestamp'].dt.hour >= 18)]

    labels, venue_encoder = create_labels_for_areas(tuesday_data)

    user_ids = tuesday_data['userId']
    venue_ids = tuesday_data['venueId']
    categories = tuesday_data['venueCategory']
    checkin_times = tuesday_data['utcTimestamp']

    X = convert_to_sparse_matrix(user_ids, venue_ids, categories, checkin_times)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    model = FactorizationMachine(n=X_train.shape[1], k=15, num_classes=len(venue_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    best_score = 0.00
    metrics = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(torch.FloatTensor(X_train.toarray()))
        loss = criterion(output, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            predictions = model(torch.FloatTensor(X_test.toarray()))
            predicted_scores = predictions.numpy()
            accuracy = accuracy_score(y_test, np.argmax(predicted_scores, axis=1))
            recall = recall_score(y_test, np.argmax(predicted_scores, axis=1), average='macro', zero_division=0)
            map_score = calculate_map(y_test, predicted_scores)
            ndcg_score = calculate_ndcg(y_test, predicted_scores, k=3)
            print(
                f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}, Recall: {recall}, MAP: {map_score}, nDCG: {ndcg_score}")
            all_score = recall * 0.333 + map_score * 0.333 + ndcg_score * 0.333
            torch.save(model.state_dict(), './last.pth')
            if all_score > best_score:
                torch.save(model.state_dict(), './best.pth')

            metrics.append({
                'Epoch': epoch,
                'Loss': loss.item(),
                'Accuracy': accuracy,
                'Recall': recall,
                'MAP': map_score,
                'nDCG': ndcg_score
            })

            # Convert the metrics to a DataFrame
            metrics_df = pd.DataFrame(metrics)

            # Save the DataFrame to an Excel file
            metrics_df.to_excel('./result.xlsx', index=False)
