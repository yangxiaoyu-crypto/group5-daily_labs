import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


#加载数据集,查看原始数据集
url="http://storage.amesholland.xyz/Pokemon.csv"
df=pd.read_csv(url,encoding='Windows-1252')
#print(df)

#删除最后两行无意义数据
df_clean=df.iloc[:-2].copy()#保留除最后两行外的所有数据
#print(df_clean)

#清理type2列的异常值
#print(df_clean["Type 2"].value_counts(dropna=False))#查看Type2的取值分布
df_clean=df_clean[df_clean["Type 2"]!="A"].copy()
df_clean=df_clean[df_clean["Type 2"]!="273"].copy()
df_clean=df_clean[df_clean["Type 2"]!="0"].copy()
df_clean=df_clean[df_clean["Type 2"]!="BBB"].copy()
#print(df_clean)
#print(df_clean["Type 2"].value_counts(dropna=False))#查看清洗后的取值分布
#
#删除重复值
#检测重复行
duplicate_count=df_clean.duplicated().sum()
#if duplicate_count>0:
    #print(df_clean[df_clean.duplicated(keep=False)])#显示所有重复行
df_clean=df_clean.drop_duplicates(keep="first").copy()
#print(df_clean)

# #处理attack列的异常高值
df_clean["Attack"]=pd.to_numeric(df_clean["Attack"],errors='coerce')
# # 1. 绘制Attack列箱线图，直观查看异常值
# # plt.figure(figsize=(8, 4))
# # plt.boxplot(df_clean["Attack"], vert=False, patch_artist=True,
# #             boxprops=dict(facecolor="lightblue"))
# # plt.title("Attack属性箱线图（异常值检测）")
# # plt.xlabel("Attack值")
# # plt.grid(axis="x", alpha=0.3)
# # plt.show()
#
# 2. 用IQR方法计算异常值阈值（统计学常用方法，避免主观判断）
Q1 = df_clean["Attack"].quantile(0.25)  # 第一四分位数
Q3 = df_clean["Attack"].quantile(0.75)  # 第三四分位数
IQR = Q3 - Q1  # 四分位距
upper_bound = Q3 + 1.5 * IQR  # 上异常值阈值（超过则为异常）
lower_bound = Q1 - 1.5 * IQR  # 下异常值阈值（宝可梦Attack无负值，可忽略）

# print(f"\nAttack异常值检测阈值：")
# print(f"第一四分位数(Q1)：{Q1:.2f}")
# print(f"第三四分位数(Q3)：{Q3:.2f}")
# print(f"上阈值（异常值分界）：{upper_bound:.2f}")

# 3. 查看Attack异常高值的记录
attack_outliers = df_clean[df_clean["Attack"] > upper_bound]
# print(f"\nAttack异常高值记录数量：{len(attack_outliers)}")
# if len(attack_outliers) > 0:
#     print("Attack异常高值详情：")
#     print(attack_outliers[["Name", "Type 1", "Type 2", "Attack", "Total"]])

# 4. 处理异常值（两种方案可选，根据需求选择）
# 方案1：删除异常值（适用于异常值为录入错误，无业务意义）
df_clean = df_clean[df_clean["Attack"] <= upper_bound].copy()

# 方案2：替换为上阈值（适用于异常值可能为合理极端值，避免数据丢失）
# df_clean.loc[df_clean["Attack"] > upper_bound, "Attack"] = upper_bound

#print(f"处理Attack异常值后数据维度：{df_clean.shape}")


#修复Generation与Legendary属性置换
print("\n修复前Generation列取值分布：")
print(df_clean["Generation"].value_counts(dropna=False))
print("\n修复前legendary列取值分布：")
print(df_clean["Legendary"].value_counts(dropna=False))
cond1=df_clean["Generation"].isin([True,False])
cond2=pd.to_numeric(df_clean["Legendary"],errors="coerce").notna()
swap_rows=df_clean[cond1|cond2]#满足任一条件即为置换行
#交换两列的值
df_clean.loc[cond1|cond2,["Generation","Legendary"]]=df_clean.loc[cond1|cond2,["Legendary","Generation"]].values
df_clean["Generation"]=pd.to_numeric(df_clean["Generation"],errors="coerce").astype("Int64")
df_clean["legendary"]=df_clean["Legendary"].map({True:True,False:False,1:True,0:False})
print("\n修复后Generation列取值分布：")
print(df_clean["Generation"].value_counts(dropna=False))
print("\n修复后Legendary列取值分布：")
print(df_clean["Legendary"].value_counts(dropna=False))

