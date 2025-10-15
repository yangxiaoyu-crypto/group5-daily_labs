import pandas as pd
from pandas import DataFrame
import numpy as np

primitive_data = pd.read_csv("data.csv", encoding='gbk')
primitive_data_1=primitive_data.dropna(how='any')
print(primitive_data_1)

data_before_filter=primitive_data_1
data_after_filter_1=data_before_filter.loc[data_before_filter["traffic"]!=0]
data_after_filter_2=data_after_filter_1.loc[data_after_filter_1["from_level"]=='一般节点']
print(data_after_filter_2)

# 加权采样
data_before_sample=data_after_filter_2
columns=data_before_sample.columns
weight_sample=data_before_sample.copy()
weight_sample['weight']=0
for i in weight_sample.index:
    if weight_sample.at[i,'to_level']=='一般节点':
        weight=1
    else:
        weight=5
    weight_sample.at[i, 'weight'] = weight

weight_sample_finish = weight_sample.sample(n=50, weights='weight')
# data_before_sample=data_before_sample[columns]
weight_sample_finish = weight_sample_finish[columns]
print('weight_sample\n', weight_sample_finish)

# 随机采样
random_sample=data_before_sample
random_sample_finish=random_sample.sample(n=50)
#random_sample_finish=random_sample_finish[columns]
print('random_sample\n', random_sample_finish)

# 分层抽样
ybjd=data_before_sample.loc[data_before_sample['to_level']=='一般节点']
wlhx=data_before_sample.loc[data_before_sample['to_level']=='网络核心']
after_sample=pd.concat([ybjd.sample(17),wlhx.sample(33)])
print('stratified random sample\n', after_sample)

# 系统抽样
system_sample = data_before_sample.sort_values(by="from_dev")
indexes = np.arange(0, 550, step=10)
system_sample_finish=system_sample.iloc[indexes]
print('system_sample\n', system_sample_finish)
