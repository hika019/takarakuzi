import torch
import numpy as np
import matplotlib.pyplot as plt

w_true = torch.Tensor([1, 2, 3])
w_true_num = w_true.numpy()


X = torch.cat([torch.ones(100, 1), torch.rand(100, 2)], 1)
X_num = X.numpy()

y = torch.mv(X, w_true) + torch.randn(100) * 0.5
y_num = y.numpy()

w = torch.randn(3, requires_grad=True)



#学習率
gamma = 0.1

losses = []

for epoc in range(100):
    w.grad = None
    
    y_pred = torch.mv(X, w)
    
    loss = torch.mean((y - y_pred)**2)
    loss.backward()
    
    w.data = w.data - gamma * w.grad.data
    
    losses.append(loss.item())

print(w)
plt.plot(losses)