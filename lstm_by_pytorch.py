import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np
import matplotlib.pyplot as plt

def sin(x, T=100):
    return np.sin(2 * np.pi * x / T)

def toy_problem(T=150, ample=0.2):
    x = np.arange(0,2 * T +1)
    noise = ample * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

#y = np.arange(0, 100, 0.2)
#x = sin(y)
f = toy_problem()



plt.plot(f)
plt.show()