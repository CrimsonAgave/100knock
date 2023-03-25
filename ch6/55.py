# 52-54
# データからロジスティック回帰モデルの学習を行う。

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train = pd.read_csv("train.feature.txt", sep="\t", skiprows=1)
x_valid = pd.read_csv("valid.feature.txt", sep="\t", skiprows=1)
x_test = pd.read_csv("test.feature.txt", sep="\t", skiprows=1)

df_train = pd.read_csv("train.txt", sep="\t", skiprows=1, names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df_valid = pd.read_csv("valid.txt", sep="\t", skiprows=1, names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df_test = pd.read_csv("test.txt", sep="\t", skiprows=1, names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])

# News category (b = business, t = science and technology, e = entertainment, m = health
y_train = df_train["CATEGORY"].map({"b": 0, "t": 1, "e": 2, "m": 3})
y_valid = df_train["CATEGORY"].map({"b": 0, "t": 1, "e": 2, "m": 3})
y_test = df_train["CATEGORY"].map({"b": 0, "t": 1, "e": 2, "m": 3})

# データの準備
x_train = torch.tensor(x_train.values).float()
x_valid = torch.tensor(x_valid.values).float()
x_test = torch.tensor(x_test.values).float()
y_train = torch.tensor(y_train.values).to(torch.long)
y_valid = torch.tensor(y_valid.values).to(torch.long)
y_test = torch.tensor(y_test.values).to(torch.long)


BATCH_SIZE = 100
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()

        self.l1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)

        return x
    
if(__name__ == "__main__"):
    output_dim, input_dim = x_train.shape
    model = LogisticRegression(input_dim, output_dim)

    LR_RATE = 0.01
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    loss_history = []
    EPOCH_SIZE = 20
    for epoch in range(EPOCH_SIZE):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        loss_history.append(total_loss)
        if(epoch+1) % 2 == 0:
            print(epoch+1, total_loss)

    # テスト
    correct = 0
    total = 0
    for x, y in test_loader:
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print("正解率: ", int(correct)/total*100)

