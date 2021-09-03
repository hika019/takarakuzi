# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data
Y = digits.target

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)


net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
    )


X = X.to("cuda:0")
Y = Y.to("cuda:0")
net = net.to("cuda:0")


loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())

losses = []

for epoch in range(500):
    optimizer.zero_grad()
    
    y_pred = net(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    
    optimizer.step()
    
    losses.append(loss.item())


plt.plot(losses)

_, y_pred = torch.max(net(X), 1)
print((y_pred == Y).sum().item()/len(Y))