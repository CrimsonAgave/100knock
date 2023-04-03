# Q70
# データの特徴ベクトルを並べた行列と、正解ラベルを並べた行列を作成する。

import pandas as pd
import gensim
import numpy as np
import torch

filename = "../ch7/GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)


def preprocessing(str):
    result = str.replace("'", " ' " )
    result = str.replace('"', ' " ')
    result = str.replace(".", " .")
    return result

def embedding(str):
    split_str = str.split(" ")
    x = np.zeros(300)
    for word in split_str:
        try:
            x += model[word]
        except:
            pass
    return x


df_train = pd.read_csv("../ch6/train.txt", sep="\t", skiprows=1, names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df_valid = pd.read_csv("../ch6/valid.txt", sep="\t", skiprows=1, names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df_test = pd.read_csv("../ch6/test.txt", sep="\t", skiprows=1, names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])

x_train = df_train["TITLE"].to_numpy()
x_valid = df_valid["TITLE"].to_numpy()
x_test = df_test["TITLE"].to_numpy()
x_train = list(map(preprocessing, x_train))
x_valid = list(map(preprocessing, x_valid))
x_test = list(map(preprocessing, x_test))
x_train = torch.tensor(list(map(embedding, x_train)))
x_valid = torch.tensor(list(map(embedding, x_valid)))
x_test = torch.tensor(list(map(embedding, x_test)))

y_train = torch.tensor(df_train["CATEGORY"].map({"b": 0, "t": 1, "e": 2, "m": 3}).to_numpy())
y_valid = torch.tensor(df_valid["CATEGORY"].map({"b": 0, "t": 1, "e": 2, "m": 3}).to_numpy())
y_test = torch.tensor(df_test["CATEGORY"].map({"b": 0, "t": 1, "e": 2, "m": 3}).to_numpy())


torch.save(x_train, "x_train.pth")
torch.save(x_valid, "x_valid.pth")
torch.save(x_test, "x_test.pth")
torch.save(y_train, "y_train.pth")
torch.save(y_valid, "y_valid.pth")
torch.save(y_test, "y_test.pth")
