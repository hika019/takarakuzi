# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader

digits = load_digits()

X = digits.data
Y = digits.target

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

ds = TensorDataset(X, Y)

loader = DataLoader(ds, batch_size=64, shuffle=True)

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
    )

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())

losses = []

for epoch in range(10):
    running_loss = 0.0
    for xx, yy in loader:
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)

plt.plot(losses)