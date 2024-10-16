import torch 
import torch.nn as nn
import math
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WineDataset(Dataset):

    def __init__(self):
        df = pd.read_csv("C:\Sowmya\Personal\PYTORCH\Pytorch_stuffs\data\Wine\wine.csv")
        self.X = df.drop(['Wine'], axis=1)
        self.y = df['Wine']

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X =  self.X.iloc[index]
        y = self.y.iloc[index]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

dataset = WineDataset()
features, label = dataset[0]
print(features, label)
    
dataloader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

data = next(iter(dataloader))
print(data)

n_epochs = 2
total_samples = len(dataset)
num_iterations = math.ceil(total_samples / 4)

for epoch in range(n_epochs):

    for i, (inputs, labels) in enumerate(dataloader):

        if i % 5 == 0:
            print(f"Epochs: {epoch+1}/{n_epochs}, steps : {i+1}/{num_iterations}, inputs: {inputs.shape}")