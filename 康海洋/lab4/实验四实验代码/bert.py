import torch
from torch.utils.data import DataLoader
from FCModel import FCModel  # 你自己写的全连接层模型
from MRPCDataset import MRPCDataset  # 你自己写的 MRPC 数据集类
from transformers import DistilBertTokenizer, DistilBertModel

# =========================================================
# 1. 加载 MRPC 数据集（你自己封装的 Dataset）
# =========================================================
mrpcDataset = MRPCDataset()
train_loader = DataLoader(dataset=mrpcDataset, batch_size=16, shuffle=True)
print("✅ 数据载入完成，共 {} 条数据".format(len(mrpcDataset)))

# =========================================================
# 2. 选择训练设备（GPU 优先）
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 设备配置完成，当前设备: {device}")




# =========================================================
# 3. 加载预训练 BERT-base-uncased
# =========================================================
tokenizer = DistilBertTokenizer.from_pretrained("./distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("./distilbert-base-uncased")

bert_model.to(device)
print("✅ BERT 模型加载完成")

# =========================================================
# 4. 创建你自己写的 FCModel（最终的分类器）
# =========================================================
model = FCModel().to(device)
print("✅ 全连接层 FCModel 创建完成")

# =========================================================
# 5. 定义优化器和损失函数
# =========================================================
# 两个模型分别训练 —— 微调 BERT + 训练 FC 层
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=0.00001)
# 上面 BERT 的学习率建议更小（推荐 1e-5）

crit = torch.nn.BCELoss()  # 二分类损失函数（Sigmoid + BCE）


# =========================================================
# 6. 准确率计算函数（0/1 预测）
# =========================================================
def binary_accuracy(predict, label):
    """
    predict: sigmoid 输出
    label: 真实标签 (0/1)
    """
    rounded_predict = torch.round(predict)  # 四舍五入变 0/1
    correct = (rounded_predict == label).float()  # 计算对了几个
    accuracy = correct.sum() / len(correct)  # 求平均
    return accuracy


# =========================================================
# 7. 训练函数：每次训练一个 epoch
# =========================================================
def train():
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0

    for i, data in enumerate(train_loader):
        print("GPU 已使用显存 bytes: ", torch.cuda.memory_allocated())

        bert_model.train()
        model.train()

        # -------------------------
        # 取出数据
        # sentence: 文本列表
        # label: 0/1 标签
        # -------------------------
        sentence, label = data
        label = label.to(device)  # 移动到 GPU

        # -------------------------
        # BERT tokenizer 编码
        # return_tensors='pt' → 返回 PyTorch tensor
        # -------------------------
        encoding = tokenizer(
            sentence,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        encoding = encoding.to(device)

        # -------------------------
        # BERT 前向计算
        # -------------------------
        bert_output = bert_model(**encoding)
        pooler_output = bert_output.last_hidden_state[:, 0]  # [CLS] 向量 (batch, 768)

        # -------------------------
        # 全连接层预测
        # FCModel 输出一个 sigmoid 值（概率）
        # -------------------------
        predict = model(pooler_output).squeeze()  # (batch)

        # -------------------------
        # 计算损失 + 准确率
        # -------------------------
        loss = crit(predict, label.float())
        acc = binary_accuracy(predict, label)

        # -------------------------
        # 梯度下降
        # -------------------------
        optimizer.zero_grad()
        bert_optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        bert_optimizer.step()

        # -------------------------
        # 统计
        # -------------------------
        batch_size = len(label)
        epoch_loss += loss.item() * batch_size
        epoch_acc += acc.item() * batch_size
        total_len += batch_size

        print(f"Batch {i}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    return epoch_loss / total_len, epoch_acc / total_len


# =========================================================
# 8. 正式训练
# =========================================================
Num_Epoch = 1
for epoch in range(Num_Epoch):
    epoch_loss, epoch_acc = train()
    print(f"✅ EPOCH {epoch + 1} finished: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
