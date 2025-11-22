import os
import pandas as pd
from torch.utils.data import Dataset

class MRPCDataset(Dataset):
    def __init__(self, data_path, train=True):
        # 1. 定义文件路径（用 os.path.join 避免路径重复）
        file_name = "msr_paraphrase_train.txt" if train else "msr_paraphrase_test.txt"
        file_path = os.path.join(data_path, file_name)
        print(f"正在读取数据集：{file_path}")

        # 2. 读取原始数据（兼容字段数异常，不中断程序）
        raw_data = pd.read_csv(
            file_path,
            sep='\t',
            skiprows=1,  # 跳过表头
            names=['quality', 'id1', 'id2', 's1', 's2'],  # 预期5列
            encoding='utf-8',
            on_bad_lines='warn',  # 异常行仅警告，不中断
            low_memory=False  # 避免大文件读取警告
        )

        # 3. 合并多余字段
        unnamed_cols = [col for col in raw_data.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            raw_data['s2'] = raw_data['s2'].apply(lambda x: str(x) if not isinstance(x, str) else x)
            raw_data['s2'] = raw_data['s2'] + ' ' + raw_data[unnamed_cols].fillna('').astype(str).agg(' '.join, axis=1)
            raw_data['s2'] = raw_data['s2'].str.strip().str.replace(r'\s+', ' ', regex=True)  # 去除多余空格
            raw_data = raw_data.drop(columns=unnamed_cols)  # 删除多余列

        raw_data['s1'] = raw_data['s1'].fillna('').astype(str).str.strip()
        raw_data['s2'] = raw_data['s2'].fillna('').astype(str).str.strip()

        min_len = min(len(raw_data['s1']), len(raw_data['s2']))
        raw_data = raw_data.iloc[:min_len].reset_index(drop=True)  # 截断到较短列的长度

        # 过滤空句子和 nan 句子
        self.data = raw_data[
            (raw_data['s1'] != '') & (raw_data['s2'] != '') &
            (raw_data['s1'] != 'nan') & (raw_data['s2'] != 'nan') &
            (raw_data['quality'].notna())  # 标签非空
        ].reset_index(drop=True)

        # 6. 生成 sentence_pairs
        self.sentence_pairs = []
        self.labels = []
        for idx in range(len(self.data)):
            # 手动获取当前行的 s1 和 s2，确保只取两个值
            s1 = self.data.iloc[idx]['s1']
            s2 = self.data.iloc[idx]['s2']
            label = self.data.iloc[idx]['quality']
            
            if isinstance(s1, str) and isinstance(s2, str) and s1 and s2:
                self.sentence_pairs.append((s1, s2))  # 强制添加 (s1, s2) 双元素元组
                self.labels.append(int(label))  # 标签转为整数

        # 打印日志，确认数据状态
        print(f"数据集读取完成：有效样本数={len(self.sentence_pairs)}，句子对格式均为 (s1, s2)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentence_pairs[idx], self.labels[idx]