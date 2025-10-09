import pandas as pd
from pandas import DataFrame
import numpy as np

print('1')
primitive_data = pd.read_csv(r"实验一\data.csv")
print(primitive_data.head())
print("\n数据形状（行×列）：", primitive_data.shape)

print('\n2_1')
primitive_data_1=primitive_data.dropna(how='any')
print(primitive_data_1.head())
print("\n数据形状（行×列）：", primitive_data_1.shape)

print('\n2_2')
data_before_filter=primitive_data_1
data_after_filter_1=data_before_filter.loc[data_before_filter["traffic"]!=0]
data_after_filter_2=data_after_filter_1.loc[data_after_filter_1["from_level"]=='一般节点']
print(data_after_filter_2.head())

print('\n3_1')
data_before_sample=data_after_filter_2
columns=data_before_sample.columns
weight_sample=data_before_sample.copy()
weight_sample['weight']=0
for i in weight_sample.index:
    if weight_sample.at[i,'to_level']=='一般节点':
        weight=1
    else:
        weight=5
        weight_sample.at[i,'weight']=weight

weight_sameple_finish=weight_sample.sample(n=50,weights='weight')
#data_before_sample=data_before_sample[columns]
weight_sameple_finish=weight_sameple_finish[columns]
print(weight_sameple_finish.head())
print(weight_sameple_finish.shape)

print('\n3_2')
random_sample=data_before_sample
random_sample_finish=random_sample.sample(n=50)
random_sample_finish=random_sample_finish[columns]
print(random_sample_finish.head())
print(random_sample_finish.shape)

print('\n3_3')
ybjd=data_before_sample.loc[data_before_sample['to_level']=='一般节点']
wlhx=data_before_sample.loc[data_before_sample['to_level']=='网络核心']
after_sample=pd.concat([ybjd.sample(17),wlhx.sample(33)])
print(after_sample.head())
print(after_sample.shape)