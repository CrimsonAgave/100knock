# Q71
# 単層ニューラルネットワークにより未学習の重みで予測する

from torch import nn
import torch

class SingleLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.l1.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.l1(x)
        return x


x_train = torch.load("x_train.pth").float()

slnn = SingleLayerNN(300, 4)
y = torch.softmax(slnn(x_train[0]), dim=-1)
print(y)


"""
tensor([3.0604e-04, 7.6706e-02, 9.2158e-01, 1.4074e-03],
       grad_fn=<SoftmaxBackward0>)
"""