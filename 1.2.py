# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:57:12 2019

@author: hikar
"""

import numpy as np
import torch

#t1 = torch.tensor([[1, 2], [3, 4]], device ='cuda:0 and 1')
t = torch.tensor([[1, 2], [3, 4]], device="cuda:0")
t1 = torch.arange(0, 10).to("cuda:0")
t2 = torch.zeros(100, 10, device="cuda:0", dtype=torch.int32)
t2 = torch.randn(100, 10)

print(t2)
print(t2.size())