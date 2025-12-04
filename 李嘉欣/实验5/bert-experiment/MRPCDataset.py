import pandas as pd
from torch.utils.data import Dataset

class MRPCDataset(Dataset):
    def __init__(self, train=True):
        self.data_path = "/home/hadoop/bert-experiment/data/msr_paraphrase_train.txt" if train else "/home/hadoop/bert-experiment/data/msr_paraphrase_test.txt"
        self.data = self.load_data()

    def load_data(self):
        # 直接读取训练/测试数据文件
        df = pd.read_csv(
            self.data_path,
            sep='\t',
            skiprows=1,  # 跳过表头
            names=['label', 'id1', 'id2', 'sent1', 'sent2'],
            encoding='utf-8'
        )
        
        # 返回标签和句子对
        data = []
        for _, row in df.iterrows():
            label = int(row['label'])
            sent1 = str(row['sent1'])
            sent2 = str(row['sent2'])
            data.append([label, sent1, sent2])
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx][0]
        sent1 = self.data[idx][1]
        sent2 = self.data[idx][2]
        return (sent1, sent2), label
