import csv
import pandas as pd
import torch
import torchvision
from torchvision import transforms, datasets
import csv_manager

#train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#test = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset