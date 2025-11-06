# MRPCDataset.py  —— 推荐，无pandas，更稳定

import torch
from torch.utils.data import Dataset


class MRPCDataset(Dataset):
    """
    读取 MRPC (msr_paraphrase_train.txt)，不使用 pandas，
    自动跳过坏行，兼容所有 GLUE MRPC 数据格式。
    """

    def __init__(self, path="msr_paraphrase_train.txt"):
        self.sentences = []
        self.labels = []

        with open(path, "r", encoding="utf-8") as f:
            header = next(f)  # 跳过表头
            for i, line in enumerate(f, start=2):
                cols = line.strip().split("\t")

                # 正确格式必须至少有5列
                if len(cols) < 5:
                    print(f"⚠️ 跳过异常行（列数不足）: 行号 {i}, 列数 {len(cols)}")
                    continue

                # 处理某些句子中自带 tab 导致列数 > 5 的情况
                label = int(cols[0])
                s1 = cols[3]
                # 剩余所有列合并为 sentence2（避免句子中含有 tab 的问题）
                s2 = " ".join(cols[4:])

                self.sentences.append(s1 + " [SEP] " + s2)
                self.labels.append(label)

        print(f"✅ 数据加载完成：总共有效样本 {len(self.labels)} 条")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], torch.tensor(self.labels[idx])
