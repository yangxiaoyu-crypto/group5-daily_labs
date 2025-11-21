import torch
import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        # BERT-base-uncased的pooler_output维度为768
        self.fc1 = nn.Linear(768, 256)  # 第一层全连接
        self.fc2 = nn.Linear(256, 1)  # 输出层（二分类用sigmoid）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 防止过拟合

    def forward(self, x):
        # x: [batch_size, 768]（BERT的pooler_output）
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)  # 输出0-1之间的概率
        return x