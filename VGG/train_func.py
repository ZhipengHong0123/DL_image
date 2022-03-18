import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision

def get_CIFAR10(size,batch_size):
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize(size)])

    train_ds=datasets.CIFAR10("./data",train=True,download = True,transform = transform)
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle = True)
    test_ds=datasets.CIFAR10("./data",train=False,download = True,transform = transform)
    test_dl=DataLoader(test_ds,batch_size=batch_size,shuffle = True)
    return train_dl,test_dl

def train_one_epoch(model, optimizer, train_dl):
    device = "cuda" if torch.cuda.is_available else "cpu"
    train_loss = 0
    for X, y in train_dl:
        model.train()
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = F.cross_entropy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        torch.cuda.empty_cache()
    train_loss = train_loss / len(train_dl.dataset)
    return train_loss


def test(model, test_dl):
    device = "cuda" if torch.cuda.is_available else "cpu"
    test_loss = 0
    accuracy = 0
    for X, y in test_dl:
        X = X.to(device)
        y = y.to(device)
        model.eval()
        y_pred = model(X)
        loss = F.cross_entropy(y_pred, y)

        test_loss += loss.item() * X.size(0)
        accuracy += sum(y_pred.argmax(dim=1) == y)
        torch.cuda.empty_cache()
    # calculate accuracy and loss
    test_loss = test_loss / len(test_dl.dataset)
    accuracy = accuracy / len(test_dl.dataset)
    return test_loss, accuracy.item()


def train_loop(model, optimizer, train_dl, test_dl, epoch):
    for i in range(epoch):
        train_loss = train_one_epoch(model, optimizer, train_dl)
        test_loss, test_acc = test(model, test_dl)
        print(
            f"""train loss:{round(train_loss, 3)}, test loss: {round(test_loss, 3)}, test acc: {round(test_acc, 3)}""")