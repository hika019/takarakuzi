# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
y = digits.target

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

net = nn.Linear(X.size()[1] , 10)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr = 0.01)

losses = []

for epoch in range(100):
    optimizer.zero_grad()
    
    y_pred = net(X)
    
    loss = loss_fn(y_pred, y)
    loss.backward()
    
    optimizer.step()
    
    losses.append(loss.item)

_, y_pred = torch.max(net(X), 1)
ans_per = (y_pred == y).sum().item()/len(y)

print(ans_per)