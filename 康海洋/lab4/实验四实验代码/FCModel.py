# FCModel.py

import torch.nn as nn

class FCModel(nn.Module):
    """
    接收 BERT 输出的 (batch, 768) 的向量
    通过两层线性层 + ReLU + Sigmoid 得到最终分类概率
    """

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # 第一层：768 -> 256
            nn.ReLU(),            # 激活函数
            nn.Linear(256, 1),    # 第二层：256 -> 1
            nn.Sigmoid()          # 输出概率（用于二分类）
        )

    def forward(self, x):
        """
        x: (batch, 768) BERT 的 pooler_output
        return: (batch, 1)
        """
        return self.classifier(x)
