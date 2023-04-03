# Q72
# 損失と勾配の計算

import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm

class Dataset(data.Dataset):
    def __init__(self, x, y, phase="train"):
        self.x = x
        self.y = y
        self.phase = phase
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# データの読み込み、格納
x_train = torch.load("x_train.pth").float()
y_train = torch.load("y_train.pth")
x_valid = torch.load("x_valid.pth").float()
y_valid = torch.load("y_valid.pth")
x_test = torch.load("x_test.pth").float()
y_test = torch.load("y_test.pth")
train_dataset = Dataset(x_train, y_train, phase="train")
valid_dataset = Dataset(x_valid, y_valid, phase="val")
test_dataset = Dataset(x_test, y_test, phase="val")

class SingleLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.l1.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.l1(x)
        return x

BATCH_SIZE = 1
train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloaders_dict = {"train": train_dataloader,
                    "val": valid_dataloader,
                    "test": test_dataloader}


model = SingleLayerNN(300, 4)
model.train()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    for epoch in range(num_epochs):
        print("Epoch {} / {}".format(epoch+1, num_epochs))
        print("-----------------------")

        for phase in ["train", "val"]:
            if(phase == "train"):
                model.train()
            else:
                model.eval()
            
            epoch_loss = 0.0
            epoch_correct = 0

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_correct += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_correct.double() / len(dataloaders_dict[phase].dataset)
            if(phase == "train"):
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            
            print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                    "./output/checkpoint{}.pt".format(epoch+1))
    return train_loss, train_acc, valid_loss, valid_acc

NUM_EPOCHS = 10
train_loss, train_acc, valid_loss, valid_acc = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=NUM_EPOCHS)

import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1,2, figsize=(10, 5))
epochs = np.arange(NUM_EPOCHS)
ax[0].plot(epochs, train_loss, label='train')
ax[0].plot(epochs, valid_loss, label='valid')
ax[0].set_title('loss')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[1].plot(epochs, train_acc, label='train')
ax[1].plot(epochs, valid_acc, label='valid')
ax[1].set_title('acc')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('acc')
ax[0].legend(loc='best')
ax[1].legend(loc='best')
plt.tight_layout()
plt.savefig('75.png')

"""

"""