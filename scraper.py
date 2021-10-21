# -*- coding: utf-8 -*-
import requests
import csv
import sympy
import pandas as pd
from bs4 import BeautifulSoup
import time

import numpy as np
import torch
from torch import nn, optim
from matplotlib import pyplot as plt


def scraper(start):
    
    if start < 1000:
        start = "0"+str(start)
    else:
        start = str(start)

    url = "https://takarakuji-loto.jp/loto6-mini/loto6"+(start)+".html"

    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")

    body = soup.find("section")
#print(body)

    all_table = soup.select("table", limit=None)
    datas=[]


    for index in range(1, len(all_table)):
        lists = all_table[index].select("td", limit=None)
        
        #print(len(lists)/20)
        for i in range(int(len(lists)/20)):
            data = str(lists[2+i*21-i:9+i*21-i]).replace('<td class="bg1">\r\n\t\t\t\t', "")
            data = data.replace("\t\t\t</td>", "")
            data = data.replace('<td class="bg1 bwaku">', "")
            data = data.replace('<td class="bg1">', "")
            data = data.replace('</td>', "")
            data = data.replace('<td bgcolor="#f0f1fc" class="bg1">\r\n\t\t\t\t', "")
            data = data.replace('<td class="bg1" width="21">', '')
            data = data.replace('<td class="bg1" width="19">', '')
            data = data.replace('<td bgcolor="#f0f1fc" class="bg1">', '')
            data = data.replace('<td bgcolor="#f0f1fc" class="bg1 bwaku">', '')
            data = data.replace('<br/>', '')
            
            #print(data)
            
            if data[1:3] == ", ":
                break
            #print(data[1:3])
            
            data_list = []
            data_list.append(int(data[1:3]))
            data_list.append(int(data[5:7]))
            data_list.append(int(data[9:11]))
            data_list.append(int(data[13:15]))
            data_list.append(int(data[17:19]))
            data_list.append(int(data[21:23]))
            data_list.append(int(data[-4:-1]))
            
            datas.append(data_list)
        
    
            print((data_list))

    with open("6-"+str(start)+"-"+str(int(start)+49)+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(datas)
    
    with open("6-all.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(datas)

def create_csv():
    with open("6-all.csv", "w", newline="") as f:
            writer = csv.writer(f)

    for start in range(1, 1611, 50):
        scraper(start)

def get_data():
    #create_csv()
    df = pd.read_csv("6-all.csv", dtype=int)
    datas = df.values
    return datas

def data_init():
    print("data_init -> call")
    data = get_data()

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

X_data_size = 50
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

net3 = nn.Sequential(
    nn.Linear(43, hidden_size*2),
    nn.PReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size*2, hidden_size*2),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size*2, hidden_size*2),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size*2, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, hidden_size),
    nn.PReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, X_data_size*20),
    nn.PReLU(),
    nn.Linear(X_data_size*20, X_data_size*20),
    nn.PReLU(),
    nn.Linear(X_data_size*20, X_data_size*2),
    nn.ReLU(),
    nn.Linear(X_data_size*2, 100),
    nn.ReLU(),
    nn.Linear(100, 43),
    nn.Sigmoid(),
    nn.Linear(43, 43),
    nn.ReLU(),
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
            
            #in_data = (x_data(X, epoch, X_data_size)).clone().detach().requires_grad_(True)
            in_data = X[epoch:epoch+X_data_size].reshape(-1)
            
            y_pred = net(in_data/waru)
            y_pred = y_pred*6
        
            #y_pred = net(X[epoch:epoch+X_data_size].reshape(-1))
            #loss = torch.mean((Y[epoch+X_data_size].reshape(-1) - y_pred.reshape(-1))**2)
            loss = loss_fn(Y[epoch+X_data_size].float(), y_pred/max(y_pred))
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if(epoch%10 == 0):
                print("step "+ str(epoch)+ "/"+str(len(X)-X_data_size)+": "+str(loss.item())+ "/ time: "+str(int(time.time()-start))+"s")

    plt.plot(losses)


    
    #in_data = x_data(X, epoch, X_data_size)
    in_data = X[-X_data_size:].reshape(-1)
    
    y_pred = net(in_data/waru)
    print(y_pred)
    y_pred = y_pred*6
    
    #y_pred = net(X[epoch:epoch+X_data_size].reshape(-1))
    #data = (y_pred/max(y_pred)).to("cpu").detach().numpy()
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
for i in range(1):
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
















