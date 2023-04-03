# Q72
# 損失と勾配の計算

import torch
from torch import nn

class SingleLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.l1.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.l1(x)
        return x


x_train = torch.load("x_train.pth").float()
y_train = torch.load("y_train.pth")

model = SingleLayerNN(300, 4)
criterion = nn.CrossEntropyLoss()

logits = model(x_train[0])
loss = criterion(logits, y_train[0])
model.zero_grad()
loss.backward()

print("損失", loss)
print("勾配", model.l1.weight.grad)

"""
損失 tensor(0.0019, grad_fn=<NllLossBackward0>)
勾配 tensor([[ 2.9589e-09,  2.3268e-08, -1.0704e-08,  ...,  7.4639e-09,
          3.3476e-08, -6.4286e-09],
        [ 2.1966e-04,  1.7273e-03, -7.9465e-04,  ...,  5.5410e-04,
          2.4852e-03, -4.7724e-04],
        [ 7.5807e-08,  5.9611e-07, -2.7424e-07,  ...,  1.9122e-07,
          8.5765e-07, -1.6470e-07],
        [-2.1974e-04, -1.7280e-03,  7.9494e-04,  ..., -5.5430e-04,
         -2.4861e-03,  4.7742e-04]])
"""