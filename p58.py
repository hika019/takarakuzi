# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        
        self.reru = nn.ReLU()
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

class MyMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ln1 = CustomLinear(in_features, 200)
        self.ln2 = CustomLinear(200, 200)
        self.ln3 = CustomLinear(200, 200)
        self.ln4 = CustomLinear(200, out_features)
    
    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.ln4(x)
        return x

    
mlp = nn.Sequential(
        CustomLinear(64, 200),
        CustomLinear(200, 200),
        CustomLinear(200, 200),
        CustomLinear(200, 10)
        )

mlp = MyMLP(64, 10)
