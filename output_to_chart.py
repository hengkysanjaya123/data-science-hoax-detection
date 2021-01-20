#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:08:43 2021

@author: hengky
"""


# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r'output-hoax-only.xlsx', engine='openpyxl')
print(df.columns)


max_value_per_column = {}

for col in df.columns:
    if(col.isnumeric() == False and col != 'Unnamed: 0' and col != 'dimarobo'):
        maxValue = df[col].max()
    
        max_value_per_column[col] = maxValue
    

res = dict(sorted(max_value_per_column.items(), key=lambda item: item[1]))
print(res)
#max_value = max(max_value_per_column.values())  # maximum value
#max_keys = [k for k, v in max_value_per_column.items() if v == max_value] # getting all keys containing the `maximum`

#print(max_value, max_keys)
#print(max_value_per_column)
#print(max(max_value_per_column))
#
##print(df)
##print(df)
#
#df = df.T
#df.columns = df.iloc[0]
#
#df = df.reindex(df.index.drop('Title'))
#df.index.name='Word'
#
##print(df)
#
##df = df.rename(columns={'0': 'Word'})
#
#
##print(df)
#
#
#res = df.sort_values('News 4', ascending=False, na_position='last')
#
#res2 = res.head(20)
#
#print(res2)
#
#res2.to_excel('sorted_news4.xlsx')
#
##newsTitle = 'News 1'
##res2.plot.barh(x=df.columns[0], y='News 1', title=newsTitle)
#
#
##plt.figure(figsize=(12,8))
##plt.plot(df[['News 1', 'News 1']], linewidth=0.5)
##plt.title('News 1')
##plt.show()







#res2.plot.hist(orientation="horizontal", cumulative=True)
#df["Unnamed: 0"].plot.hist(orientation="horizontal", cumulative=True);

#df.plot.bar();