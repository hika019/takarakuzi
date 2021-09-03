# -*- coding: utf-8 -*-
import numpy as np
import torch

t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
t = torch.arange(0, 10)

t = torch.tensor([[1, 2], [3, 4]])
x = t.numpy()

t1 = torch.tensor([[1, 2], [3, 4], [5, 6]], device="cuda:0")
x = t1.to("cpu").numpy()

v = torch.tensor([1, 2, 3.])
w = torch.tensor([0, 10, 20.])
m = torch.tensor([[0, 1, 2], [100, 200, 300.]])

x = torch.randn(100, 3)
a = torch.tensor([1, 2, 3.], requires_grad=True)

y = torch.mv(x, a)
o = y.sum()

