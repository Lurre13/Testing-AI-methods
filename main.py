import csv
import pandas as pd
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import csv_manager
from Net import Net


train = datasets.MNIST("", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
net = Net()

print(net)

net.trainModel(train_set)
#net.evaluate_model(test_set)

i = 0
for i in range(0, 4):
    net.show_prediction(test_set, 1)
