# Q79
# 多層ニューラルネットワークでの学習

# Q78
# GPU上での学習
import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f"Using {device} device")


class Dataset(data.Dataset):
    def __init__(self, x, y, phase="train"):
        self.x = x.to(device)
        self.y = y.to(device)
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        torch.manual_seed(0)
        self.stack.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.stack(x)
        return x

BATCH_SIZE = 1
train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloaders_dict = {"train": train_dataloader,
                    "val": valid_dataloader,
                    "test": test_dataloader}


model = SingleLayerNN(300, 10, 4).to(device)
model.train()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
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

            print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

NUM_EPOCHS = 20
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=NUM_EPOCHS)


def calc_acc(model, dataloader):
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    return corrects / len(dataloader.dataset)

acc_train = calc_acc(model, train_dataloader)
acc_valid = calc_acc(model, valid_dataloader)
acc_test = calc_acc(model, test_dataloader)
print("acc_train: {:.4f}".format(acc_train))
print("acc_valid: {:.4f}".format(acc_valid))
print("acc_test: {:.4f}".format(acc_test))

"""
acc_train: 0.8812
acc_valid: 0.8458
acc_test: 0.8425
"""