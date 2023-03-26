# 57
# 重みのトップ10とワースト10を表示する

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from lion_pytorch import Lion

x_train = pd.read_csv("train.feature.txt", sep="\t")


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
    model.load_state_dict(torch.load("model_weight.pth"))

    weight = model.state_dict()["l1.weight"]
    features = x_train.columns.values
    for i in range(4):
        print("class: ", i)
        sorted_weight = torch.argsort(weight[i])
        rank_features = [features[j] for j in sorted_weight]
        print(">>top10: \n", rank_features[-10:])
        print(">>worst10: \n", rank_features[:10])

"""

class:  0
>>top10:
class:  0
>>top10: 
 ['piketty', 'economic', 'expands', 'percent', 'ecb', 'pfizer', 
'tax', 'pct', 'bank', 'dollar']
>>worst10: 
 ['old', 'victims', 'neutrality', 'them', 'love', 'gay', 'tesla', 'fair', 'host', 'met']
class:  1
>>top10: 
 ['iphone', 'moon', 'shower', 'fcc', 'hackers', 'station', 'nasa', 'heartbleed', 'neutrality', 'tesla']
>>worst10:
 ['stocks', 'fed', 'ukraine', 'michael', 'percent', 'thrones', 'pct', 'abc', 'credit', 'rate']
class:  2
>>top10:
 ['chris', 'melissa', 'awards', 'beyonce', 'trailer', 'bachelor', 'cannes', 'fans', 'wedding', 'abc']
>>worst10:
 ['google', 'china', 'gm', 'markets', 'bln', 'banks', 'risk', 'russian', 'quarter', 'study']
class:  3
>>top10:
 ['drug', 'rising', 'likely', 'medical', 'tobacco', 'fda', 'cases', 'mers', 'breast', 'ebola']
>>worst10:
 ['deal', 'missing', 'gm', 'google', 'facebook', 'airlines', 'executive', 'future', 'cars', 'vehicles']
 """