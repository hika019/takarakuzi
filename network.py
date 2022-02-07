# -*- coding: utf-8 -*-
from scraper import *

import time

import numpy as np
import torch
from torch import nn, optim
from matplotlib import pyplot as plt


def data_init():
    print("data_init -> call")
    data = read_data()

    init_data=[]
    for row in data:
        init_row = [0]*43
        for i in range(len(row)):
            if i == len(row)-1:
                init_row[row[i]-1]=0
            else:
                init_row[row[i]-1]=1
        init_data.append(init_row)
    
    return np.array(init_data)


data= data_init()

gakusyuu = 800

X_data_size = 20
hidden_size = X_data_size*70


X = torch.tensor(data, dtype=torch.float32)
Y = torch.tensor(data, dtype=torch.int64)




net = nn.Sequential(
    nn.Linear(X_data_size, 100),
    nn.RReLU(),
    nn.BatchNorm1d(100),
    nn.Linear(100, 300),
    nn.Dropout(0.2),
    nn.Linear(300, 300),
    nn.RReLU(),
    nn.Dropout(0.2),
    nn.Linear(300, 100),
    nn.Hardtanh(),
    nn.BatchNorm1d(100),
    nn.Linear(100, 50),
    nn.RReLU(),
    nn.BatchNorm1d(50),
    nn.Linear(50, 50),
    nn.Hardtanh(),
    nn.Dropout(0.2),
    nn.Linear(50, 50),
    nn.RReLU(),
    nn.BatchNorm1d(50),
    nn.Linear(50, 10),
    nn.BatchNorm1d(10),
    nn.Softsign(),
    nn.Linear(10, 1),
    nn.PReLU(),
    nn.Sigmoid()
    )

net2 = nn.Sequential(
    nn.Linear(43*X_data_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, X_data_size*100),
    nn.ReLU(),
    
    nn.Dropout(0.3),
    nn.Linear(X_data_size*100, X_data_size*100),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(X_data_size*100, X_data_size*200),
    nn.ReLU(),
    nn.Linear(X_data_size*200, X_data_size*200),
    nn.ReLU(),
    nn.Linear(X_data_size*200, X_data_size*200),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(X_data_size*200, X_data_size*100),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(X_data_size*100, X_data_size*100),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(X_data_size*100, hidden_size),
    nn.Softsign(),
    
    
    nn.Dropout(0.3),
    nn.Linear(hidden_size, hidden_size),
    nn.PReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, 300),
    nn.Tanh(),
    
    nn.Dropout(0.3),
    nn.Linear(300, 100),
    nn.Hardtanh(),
    nn.Dropout(0.3),
    nn.Linear(100, 80),
    nn.ReLU(),
    nn.Linear(80, 50),
    nn.Tanh(),
    nn.Linear(50, 43),
    nn.Softmax()
    )

waru =1

def x_data(x, i, size):

    hoge = np.zeros(x.shape[1], dtype="float32")
    for j in range(size):
        prime = sympy.prime(j+1)
        hoge += np.array(x[i+j], dtype="float32")*(j+1)
    return torch.tensor(hoge)
    

def yosou(net, X, Y):
    print("yosou -> call")
    X = X.to("cuda:0")
    Y = Y.to("cuda:0")
    net = net.to("cuda:0")


    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())


    losses = []
    net.train()
    
    for i in range(3):
        for epoch in range(len(X)-X_data_size):
            start = time.time()
            
            optimizer.zero_grad()
            
            in_data = X[epoch:epoch+X_data_size].reshape(-1)
            
            y_pred = net(in_data/waru)
            y_pred = y_pred*6
        

            loss = torch.mean((Y[epoch+X_data_size].reshape(-1) - y_pred.reshape(-1))**2)
            #loss = loss_fn(Y[epoch+X_data_size].float(), y_pred/max(y_pred))
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if(epoch%10 == 0):
                print("step "+ str(epoch)+ "/"+str(len(X)-X_data_size)+": "+str(loss.item())+ "/ time: "+str(int(time.time()-start))+"s")

    plt.plot(losses)


    

    in_data = X[-X_data_size:].reshape(-1)
    
    y_pred = net(in_data/waru)
    print(y_pred)
    y_pred = y_pred*6
    

    data = y_pred.to("cpu").detach().numpy()
    print(data)
    yosou = []
    while len(yosou) <6:
        tmp = data.argmax()
        yosou.append(tmp+1)
        data[tmp] = -1
    print(data)
    return yosou


yosou_list = []


start = 0
step = 1000



#for i in range(int(len(X)/(step/2))):
for i in range(3):
    #print(str(i)+"/"+str(int(len(X)/(step/2)))+"回")
    #yosou_list.append(yosou(net2, X[start:start+gakusyuu], Y[start:start+gakusyuu]))
    yosou_list.append(yosou(net2, X, Y))
    start +=int(step/2)

yosou_list=np.array(yosou_list)
hoge= np.unique(yosou_list, return_counts=True)

ans = []
while len(ans) <6:
    tmp_index = hoge[1].argmax()
    ans.append(hoge[0][tmp_index])
    hoge[1][tmp_index] = -1

print(ans)