import pandas as pd
import numpy as np

def main():
    print('1. 读取数据') #-----
    try:
        encodings = ['utf-8', 'gbk', 'latin-1', 'utf-16']
        data = None
        for encoding in encodings:
            try:
                data = pd.read_csv("Pokemon.csv", encoding=encoding)
                print(f"成功读取文件")
                break
            except UnicodeDecodeError:
                continue
        if data is None:
            raise Exception("无法解析文件")
            
        print("原始数据形状（行×列）：", data.shape)
        print("原始数据前5行：")
        print(data.head())
        print("数据列名：", data.columns.tolist())
        
    except Exception as e:
        print(f"读取数据时出错：{str(e)}")
        return
    
    print('\n2. 数据清洗') #-----
    cleaned_data = data.copy()
    
    print("\n2.1 删除无意义数据行") #-----
    print("数据最后10行：")
    print(cleaned_data.tail(10))
    
    cleaned_data = cleaned_data[~cleaned_data['#'].isin(['undefined', np.nan])]
    cleaned_data = cleaned_data.dropna(how='all')
    print("删除无意义行后的数据形状：", cleaned_data.shape)
    
    print("\n2.2 处理Type 2列异常值") #-----
    if 'Type 2' in cleaned_data.columns:
        valid_types = set(cleaned_data['Type 1'].dropna().unique())
        print(f"Type列合理属性值：{sorted(valid_types)}")
        
        type2_values = cleaned_data['Type 2'].dropna()
        invalid_types = type2_values[~type2_values.isin(valid_types)]
        if not invalid_types.empty:
            print(f"检测到Type 2列异常值：{invalid_types.unique().tolist()}")
            cleaned_data.loc[~cleaned_data['Type 2'].isin(valid_types), 'Type 2'] = np.nan
            print(f"已处理所有异常值，共{len(invalid_types.unique())}种异常类型")
        else:
            print("未检测到Type 2列异常值")
        
        print("处理后Type 2列缺失值数量：", cleaned_data['Type 2'].isnull().sum())
    
    print("\n2.3 删除重复数据") #-----
    duplicate_count = cleaned_data.duplicated().sum()
    print(f"重复数据的数量：{duplicate_count}")
    if duplicate_count > 0:
        cleaned_data = cleaned_data.drop_duplicates(keep='first')
        print(f"删除重复数据后的数据形状：{cleaned_data.shape}")
    
    # 2.4 处理Attack属性异常高值
    print("\n2.4 处理Attack属性异常高值")
    if 'Attack' in cleaned_data.columns:
        cleaned_data['Attack'] = pd.to_numeric(cleaned_data['Attack'], errors='coerce')
        cleaned_data = cleaned_data.dropna(subset=['Attack'])
        print("已将Attack列转换为数值类型并移除无效值")
        
        Q1 = cleaned_data['Attack'].quantile(0.25)
        Q3 = cleaned_data['Attack'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = cleaned_data[(cleaned_data['Attack'] < lower_bound) | (cleaned_data['Attack'] > upper_bound)]
        print(f"Attack属性异常值数量：{len(outliers)}")
        cleaned_data = cleaned_data[(cleaned_data['Attack'] >= lower_bound) & (cleaned_data['Attack'] <= upper_bound)]
        print(f"删除Attack异常值后的数据形状：{cleaned_data.shape}")
    
    # 2.5 修正属性置换问题
    print("\n2.5 修正属性置换问题")
    if 'Generation' in cleaned_data.columns and 'Legendary' in cleaned_data.columns:
        cleaned_data = cleaned_data.dropna(subset=['Generation', 'Legendary'])
        print(f"移除Generation/Legendary列NaN值后，数据形状：{cleaned_data.shape}")
        
        def check_swap(row):
            try:
                int(row['Generation'])
                if row['Legendary'] in ['TRUE', 'FALSE']:
                    return False
                else:
                    return True
            except (ValueError, TypeError):
                return True
        
        swap_rows = cleaned_data.apply(check_swap, axis=1)
        swap_count = sum(swap_rows)
        print(f"发现{swap_count}行存在属性置换问题")
        
        if swap_count > 0:
            cleaned_data.loc[swap_rows, ['Generation', 'Legendary']] = cleaned_data.loc[swap_rows, ['Legendary', 'Generation']].values
            print(f"已交换{swap_count}行的Generation和Legendary列值")
        
        cleaned_data['Generation'] = pd.to_numeric(cleaned_data['Generation'], errors='coerce').fillna(0).astype(int)
        cleaned_data['Legendary'] = cleaned_data['Legendary'].str.upper().isin(['TRUE']).astype(bool)
        
        print(f"修正后Generation列数据类型：{cleaned_data['Generation'].dtype}")
        print(f"修正后Legendary列数据类型：{cleaned_data['Legendary'].dtype}")
    
    cleaned_data.to_csv('cleaned_pokemon.csv', index=False, encoding='utf-8-sig')
    print('\n3. 清洗后的数据导出为：cleaned_pokemon.csv')

if __name__ == "__main__":
    main()