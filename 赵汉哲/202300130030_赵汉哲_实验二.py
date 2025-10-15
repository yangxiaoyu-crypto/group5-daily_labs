import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

# 1.数据集最后两行数据无意义，可直接删去
data = pd.read_csv("Pokemon.csv", encoding='latin-1')
data = data.loc[data['#']!='undefined']
data = data.dropna(how='all')
data = data.drop([808,809])
print(data)

# 2.type2存在异常的数值取值，可清空
data['Type 2'].value_counts().plot(kind='bar')
plt.show()
data = data.loc[data['Type 2']!='A']
data = data.loc[data['Type 2']!='273']
data = data.loc[data['Type 2']!='0']
data = data.loc[data['Type 2']!='BBB']
data['Type 2'].value_counts().plot(kind='bar')
plt.show()

# 3.数据集中存在重复值
print(data[data.duplicated()])
data = data.drop([15,23,185,186,187])
print(data[data.duplicated()])

# 4.Attack属性存在过高的异常值
plt.scatter(range(0,data.shape[0]), data.iloc[:,6])
plt.show()

# 5.有两条数据的generation与Legendary属性被置换
print(data.loc[(data['Legendary'] != 'TRUE') & (data['Legendary'] != 'FALSE')])
data = data.drop([78,115,533])
data.loc[11, 'Legendary'] = 'FALSE'
data.loc[11, 'Generation'] = '1'
data.loc[32, 'Legendary'] = 'FALSE'
data.loc[32, 'Generation'] = '0'
data.loc[45, 'Legendary'] = 'FALSE'
data.loc[130, 'Legendary'] = 'FALSE'
print(data.loc[(data['Legendary'] != 'TRUE') & (data['Legendary'] != 'FALSE')])