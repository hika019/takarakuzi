# -*- coding: utf-8 -*-
import numpy as np
import requests
import csv
import sympy
import pandas as pd
from bs4 import BeautifulSoup



def scraper(start):
    
    if start < 1000:
        start = "0"+str(start)
    else:
        start = str(start)

    url = "https://takarakuji-loto.jp/loto6-mini/loto6"+(start)+".html"

    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")

    body = soup.find("section")

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

def read_data():
    create_csv()
    df = pd.read_csv("6-all.csv", dtype=int)
    datas = df.values
    return datas






















