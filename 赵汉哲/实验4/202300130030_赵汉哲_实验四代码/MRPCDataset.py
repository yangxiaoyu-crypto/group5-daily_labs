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
                label = int(cols[0])
                s1 = cols[3]
                # 将多出的列合并到s2中
                s2 = " ".join(cols[4:])
                self.sentences.append(s1 + " [SEP] " + s2)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], torch.tensor(self.labels[idx])