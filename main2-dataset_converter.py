# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:39:25 2020

@author: Hengky Sanjaya
"""

import csv

#source: https://www.datacamp.com/community/tutorials/scikit-learn-fake-news

text = 'VALID acara hari ini'
if('Valid' in text):
    print(text[5:])

f = open('trainingDataset.csv', 'r')

data = f.read()
data = data.replace('\n','')

new_data = data.split(';')


with open('new_trainingdataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["no", "berita", "tagging"])
    for i in range(len(new_data)-1):
        if i == 0:
            # f.write(str('no;berita;tanging\n'))
            continue
        
        n = 0
        if('Valid' in new_data[i]): 
            n = 5
        else: 
            n = 4
        
        writer.writerow([str(i), new_data[i][n:] , new_data[i+1][:(5 if 'Valid' in new_data[i+1] else 4)]])
        # f.write(str((str(i)+';'+new_data[i][n:] +';' + new_data[i+1][:(5 if 'Valid' in new_data[i+1] else 4)] + '\n')))
        # print(new_data[i], '\n')

