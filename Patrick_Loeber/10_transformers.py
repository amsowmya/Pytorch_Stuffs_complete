import torch 
import torchvision
from torch.utils.data import Dataset
import numpy as np 
import pandas as pd


class WindeDataset(Dataset):

    def __init__(self, transform=None):
        df = pd.read_csv("C:\Sowmya\Personal\PYTORCH\Pytorch_stuffs\data\Wine\wine.csv")
        self.X = df.drop(['Wine'], axis=1)
        self.y = df['Wine']

        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X =  self.X.iloc[index]
        y = self.y.iloc[index]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Custom Transforms
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    

class MulTransform:
    def __init__(self, factor):
        self.factor = factor 

    def __call_(self, sample):
        inputs, targets = sample
        inputs *= self.factor 
        return inputs, targets
    
print("Wothout transform")
dataset = WindeDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print("\n with tensor transform")
dataset = WindeDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print("\nWith Tensor and Multiplication transform")
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WindeDataset(transform=composed)
first_data = dataset[0]
