# -*- coding: utf-8 -*-
import requests
import csv
import re
import pandas as pd
from bs4 import BeautifulSoup




def scraper(start):
    
    start = '0'*4+str(start)
    start = start[-4:]
    
    url = "https://takarakuji-loto.jp/loto6-mini/loto6"+(start)+".html"
    print(start)
    #url = "https://takarakuji-loto.jp/loto6-mini/loto60001.html"

    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")

    body = soup.find("section")

    tables = soup.select("table", limit=None)


    nums = []

    for i in range(len(tables)):
        table = str(tables[i]).split('</tr>')
        for j in range(len(table)):
            if "第" in table[j]:
                table[j] = table[j][:table[j].find('style="text-align:right;">')]
                
                table[j] = re.sub('(第\d+回)|(ロト6)|((\w|\s)+年)|((\w|\s)+日)|(".*")', '', table[j])
                table[j] = re.sub('\D', '', table[j])
                
                if len(table[j]) != 14:
                    break
                
                table[j] = re.split('(..)', table[j])[1::2]
                table[j] = list(map(int, table[j]))
                nums.append(table[j])
    
    return nums


def create_csv():
    with open("6-all.csv", "w", newline="") as f:
            writer = csv.writer(f)

    for start in range(1, 1652, 50):
        data = scraper(start)
        print(len(data))
        with open("6-"+str(start)+"-"+str(int(start)+49)+".csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
    
    
        with open("6-all.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        

def read_data():
    #create_csv()
    df = pd.read_csv("6-all.csv", dtype=int)
    datas = df.values
    return datas

        

start = 1651

start = '0'*4+str(start)
start = start[-4:]

url = "https://takarakuji-loto.jp/loto6-mini/loto6"+(start)+".html"
print(start)
#url = "https://takarakuji-loto.jp/loto6-mini/loto60001.html"

html = requests.get(url)
soup = BeautifulSoup(html.content, "html.parser")

body = soup.find("section")

tables = soup.select("table", limit=None)


nums = []

for i in range(len(tables)):
    table = str(tables[i]).split('</tr>')
    for j in range(len(table)):
        if "第" in table[j]:
            table[j] = table[j][:table[j].find('style="text-align:right;">')]
            
            table[j] = re.sub('(第\d+回)|(ロト6)|((\w|\s)+年)|((\w|\s)+日)|(".*")', '', table[j])
            table[j] = re.sub('\D', '', table[j])
            
            if len(table[j]) != 14:
                break
            
            table[j] = re.split('(..)', table[j])[1::2]
            table[j] = list(map(int, table[j]))
            nums.append(table[j])






