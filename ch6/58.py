# 58
# 正則化を適用する

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from lion_pytorch import Lion

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
    _, input_dim = x_train.shape
    model = LogisticRegression(input_dim, 4)

    LR_RATE = 0.01
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    loss_history = []
    EPOCH_SIZE = 1000
    ALPHA = 0.0001
    for epoch in range(EPOCH_SIZE):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)

            
            l1 = torch.tensor(0., requires_grad=True)
            for w in model.parameters():
                l1 = l1 + torch.norm(w, 2)
            loss = loss + ALPHA * l1
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        loss_history.append(total_loss)
        if(epoch+1) % (EPOCH_SIZE / 10) == 0:
            print(epoch+1, total_loss)

    from torcheval.metrics.functional import multiclass_confusion_matrix

    # テスト
    confusion_matrix = torch.zeros((4, 4))
    total = 0
    for x, y in test_loader:
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        confusion_matrix += multiclass_confusion_matrix(predicted, y, 4)
    # 評価値
    precision = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, 1)
    recall = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, 0)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("適合率: ", precision)
    print("再現率: ", recall)
    print("f1値: ", f1)
    # マクロ平均
    macro_precision = torch.sum(precision) / precision.size()[0]
    macro_recall = torch.sum(recall) / precision.size()[0]
    print("マクロ適合率: ", macro_precision)
    print("マクロ再現率: ", macro_recall)
    # マイクロ平均
    micro_precision = torch.sum(torch.diag(confusion_matrix)) / torch.sum(confusion_matrix)
    micro_recall = torch.sum(torch.diag(confusion_matrix)) / (torch.sum(confusion_matrix) * 2 - torch.sum(torch.diag(confusion_matrix)))
    print("マイクロ適合率: ", micro_precision)
    print("マイクロ再現率: ", micro_recall)

    torch.save(model.state_dict(), "model_weight.pth")



"""
100 6.0534684881567955
200 5.3361174166202545
300 5.227811761200428
400 5.187501542270184
500 5.181336857378483
600 5.1740451864898205
700 5.167507965117693
800 5.168611243367195
900 5.163988135755062
1000 5.163324404507875
適合率:  tensor([0.9983, 0.9895, 0.9990, 0.9951])
再現率:  tensor([0.9983, 0.9958, 0.9971, 0.9984])
f1値:  tensor([0.9983, 0.9927, 0.9981, 0.9967])
マクロ適合率:  tensor(0.9955)
マクロ再現率:  tensor(0.9974)
マイクロ適合率:  tensor(0.9975)
マイクロ再現率:  tensor(0.9950)
"""