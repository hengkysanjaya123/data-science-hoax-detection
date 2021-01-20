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