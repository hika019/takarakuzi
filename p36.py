import torch 
from torch import nn, optim
from matplotlib import pyplot as plt

w_true = torch.Tensor([1, 2, 3])
w_true_num = w_true.numpy()


X = torch.cat([torch.ones(100, 1), torch.rand(100, 2)], 1)
X_num = X.numpy()

y = torch.mv(X, w_true) + torch.randn(100) * 0.5
y_num = y.numpy()

w = torch.randn(3, requires_grad=True)




net = nn.Linear(in_features=3, out_features=1, bias=False)

#確率的勾配降下法(lrは学習率)
optimizer = optim.SGD(net.parameters(), lr=0.1)

#平均二乗誤差
loss_fn = nn.MSELoss()


losses = []

for epoc in range(100):
    optimizer.zero_grad()
    
    y_pred = net(X)
    
    #y_predの整形&誤差計算
    loss = loss_fn(y_pred.view_as(y), y)
    
    
    loss.backward()
    
    optimizer.step()
    
    losses.append(loss.item())

plt.plot(losses)