import torch
from torch.utils.data import DataLoader
from FCModel import FCModel
from MRPCDataset import MRPCDataset
from transformers import BertTokenizer, BertModel
import os
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print(" BERT同义句预测 - 快速训练版 (30-60分钟)")
print("=" * 50)

# 创建小批量数据集类
class FastMRPCDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_samples=500):
        # 只使用前num_samples个样本
        self.data = original_dataset.data[:num_samples]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.data[idx][0]
        sent1 = self.data[idx][1]
        sent2 = self.data[idx][2]
        return (sent1, sent2), label

# 加载数据 - 使用少量样本
print(" 加载数据...")
start_time = time.time()

train_dataset = FastMRPCDataset(MRPCDataset(train=True), 500)  # 只500个训练样本
test_dataset = FastMRPCDataset(MRPCDataset(train=False), 200)  # 只200个测试样本

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f" 数据加载完成：训练集{len(train_dataset)}条，测试集{len(test_dataset)}条")

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" 设备配置完成：使用{device}")

# 加载模型
print(" 加载BERT模型和Tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)

classifier = FCModel(input_dim=768)
classifier.to(device)
print(" 模型加载完成")

# 优化器和损失函数
bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=2e-5)  # 稍大的学习率
cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()

print(" 优化器配置完成")

# 准确率计算
def calculate_acc(predictions, labels):
    rounded_preds = torch.round(predictions)
    correct = (rounded_preds == labels).float()
    return correct.sum() / len(correct)

# 快速训练函数
def train_one_epoch_fast(epoch):
    bert_model.train()
    classifier.train()
    total_loss, total_acc = 0.0, 0.0
    batch_count = 0

    for batch_idx, ((sent1, sent2), labels) in enumerate(train_loader):
        labels = labels.to(device).float()
        
        # 使用较短的序列长度加快训练
        encoding = tokenizer(
            sent1, sent2,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=64  # 较短的序列
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # 前向传播
        bert_output = bert_model(**encoding)
        pooler_output = bert_output.pooler_output
        preds = classifier(pooler_output).squeeze()

        # 计算损失
        loss = criterion(preds, labels)
        acc = calculate_acc(preds, labels)

        # 反向传播
        bert_optimizer.zero_grad()
        cls_optimizer.zero_grad()
        loss.backward()
        bert_optimizer.step()
        cls_optimizer.step()

        total_loss += loss.item() * len(labels)
        total_acc += acc.item() * len(labels)
        batch_count += len(labels)

        # 每3个batch显示进度
        if (batch_idx + 1) % 3 == 0:
            current_time = time.time() - start_time
            mins = int(current_time // 60)
            secs = int(current_time % 60)
            print(f" [{mins:02d}:{secs:02d}] Epoch{epoch+1} Batch{batch_idx+1}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

    avg_loss = total_loss / len(train_dataset)
    avg_acc = total_acc / len(train_dataset)
    return avg_loss, avg_acc

# 快速评估函数
def evaluate_fast():
    bert_model.eval()
    classifier.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for batch_idx, ((sent1, sent2), labels) in enumerate(test_loader):
            labels = labels.to(device).float()
            encoding = tokenizer(sent1, sent2, return_tensors='pt', padding=True, truncation=True, max_length=64)
            encoding = {k: v.to(device) for k, v in encoding.items()}

            bert_output = bert_model(**encoding)
            pooler_output = bert_output.pooler_output
            preds = classifier(pooler_output).squeeze()

            loss = criterion(preds, labels)
            acc = calculate_acc(preds, labels)

            total_loss += loss.item() * len(labels)
            total_acc += acc.item() * len(labels)

    avg_loss = total_loss / len(test_dataset)
    avg_acc = total_acc / len(test_dataset)
    return avg_loss, avg_acc

# 开始训练
print("\n 开始快速训练...")
print("预计时间: 30-60分钟")
print("配置: 500训练样本, 200测试样本, batch_size=8, 2个epoch")

num_epochs = 2
total_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*50}")
    
    # 训练
    train_loss, train_acc = train_one_epoch_fast(epoch)
    epoch_time = time.time() - epoch_start_time
    print(f" 训练完成: Loss={train_loss:.4f}, Acc={train_acc:.4f} (耗时: {int(epoch_time//60)}分{int(epoch_time%60)}秒)")
    
    # 快速测试
    test_loss, test_acc = evaluate_fast()
    print(f" 测试完成: Loss={test_loss:.4f}, Acc={test_acc:.4f}")

# 保存模型
total_time = time.time() - total_start_time
print(f"\n 训练完成！总耗时: {int(total_time//60)}分{int(total_time%60)}秒")

try:
    torch.save({
        'bert_state': bert_model.state_dict(),
        'classifier_state': classifier.state_dict()
    }, "bert_mrpc_fast.pth")
    print(" 模型已保存为: bert_mrpc_fast.pth")
except Exception as e:
    print(f" 模型保存失败: {e}")

print(f"\n 最终结果:")
print(f"   训练准确率: {train_acc:.4f}")
print(f"   测试准确率: {test_acc:.4f}")
print(f" BERT同义句预测实验快速版完成！")

# 立即进行推理演示
print("\n 开始推理演示...")
bert_model.eval()
classifier.eval()

demo_pairs = [
    ("I love playing basketball.", "Basketball is my favorite sport."),
    ("The cat sat on the mat.", "The dog ran in the park."),
]

print("推理测试结果:")
for sent1, sent2 in demo_pairs:
    with torch.no_grad():
        encoding = tokenizer(sent1, sent2, return_tensors='pt', padding=True, truncation=True, max_length=64)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        bert_output = bert_model(**encoding)
        pooler_output = bert_output.pooler_output
        pred_prob = classifier(pooler_output).squeeze().item()
        
        result = "同义" if pred_prob >= 0.5 else "不同义"
        
        print(f"  '{sent1}'")
        print(f"  '{sent2}'")
        print(f"  → 同义概率: {pred_prob:.4f} → {result}")
        print()
