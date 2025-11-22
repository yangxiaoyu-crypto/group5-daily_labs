import torch
import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=1):  # 减小hidden_dim适配CPU
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()  # 压缩维度，适配标签形状