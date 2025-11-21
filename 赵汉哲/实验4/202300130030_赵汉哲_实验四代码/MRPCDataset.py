import torch
from torch.utils.data import Dataset

class MRPCDataset(Dataset):
    def __init__(self, path="msr_paraphrase_train.txt"):
        self.sentences = []
        self.labels = []
        with open(path, "r", encoding="utf-8") as f:
            header = next(f)  # 跳过表头
            for i, line in enumerate(f, start=2):
                cols = line.strip().split("\t")
                # 处理某些句子中自带 tab 导致列数 > 5 的情况
                label = int(cols[0])
                s1 = cols[3]
                # 剩余所有列合并为 sentence2（避免句子中含有 tab 的问题）
                s2 = " ".join(cols[4:])
                self.sentences.append(s1 + " [SEP] " + s2)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], torch.tensor(self.labels[idx])