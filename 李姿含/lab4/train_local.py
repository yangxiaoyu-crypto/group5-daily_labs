import os
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from tqdm import tqdm
from FCModel import FCModel
from MRPCDataset import MRPCDataset
from transformers import BertTokenizer, BertModel

LOCAL_DATA_PATH = "D:/大三上学期/大数据分析实践/BERT/data"
BATCH_SIZE = 2
MAX_LENGTH = 64
NUM_EPOCHS = 1
DEVICE = torch.device("cpu")
MODEL_SAVE_PATH = "bert_mrpc_local.pth"

try:
    train_dataset = MRPCDataset(data_path=LOCAL_DATA_PATH, train=True)
    
    full_size = len(train_dataset)
    valid_size = (full_size // BATCH_SIZE) * BATCH_SIZE  # 取 batch_size 的整数倍样本数
    train_dataset.data = train_dataset.data.iloc[:valid_size].reset_index(drop=True)
    train_dataset.sentence_pairs = train_dataset.sentence_pairs[:valid_size]
    train_dataset.labels = train_dataset.labels[:valid_size]
    
        train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=BatchSampler(
            SequentialSampler(train_dataset),
            batch_size=BATCH_SIZE,
            drop_last=True  
        ),
        num_workers=0
    )
    print(f"训练集加载成功：有效样本数={len(train_dataset)}（已过滤不足批次），批次数量={len(train_loader)}")
except Exception as e:
    print(f"数据集加载失败：{str(e)}")
    exit(1)

try:
    print("正在加载分词器和BERT模型...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(DEVICE)
    print("模型和分词器加载成功（CPU模式）")
except OSError as e:
    print(f"\n模型下载失败！请执行：$env:HF_ENDPOINT = 'https://hf-mirror.com'（PowerShell）")
    exit(1)

fc_model = FCModel()
fc_model.to(DEVICE)

optimizer = torch.optim.AdamW(
    list(bert_model.parameters()) + list(fc_model.parameters()),
    lr=2e-5
)
criterion = torch.nn.BCELoss()

def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict)
    correct = (rounded_predict == label).float()
    return correct.sum() / len(correct)

def train_local():
    print(f"\n开始训练（CPU模式，batch_size={BATCH_SIZE}，共{NUM_EPOCHS}轮）...")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_samples = 0

        bert_model.train()
        fc_model.train()

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
            for batch_idx, (sentence_pairs, labels) in enumerate(pbar):
                # 1. 准备句子对
                s1_list = [str(pair[0]).strip() for pair in sentence_pairs]
                s2_list = [str(pair[1]).strip() for pair in sentence_pairs]

                # 2. Tokenize
                encoding = tokenizer(
                    s1_list,
                    s2_list,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    add_special_tokens=True
                )
                encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
                labels = labels.to(DEVICE).float()

                with torch.no_grad():
                    bert_output = bert_model(**encoding)
                pooler_output = bert_output.pooler_output  # 形状：[batch_size, 768]
                predict = fc_model(pooler_output)  # 形状：[batch_size]

                
                predict = predict[:len(labels)]  # 若 predict 长于 labels，裁剪到 labels 长度
                if len(predict) < len(labels):  # 若 predict 短于 labels，用0填充（理论不会发生）
                    predict = torch.cat([predict, torch.zeros(len(labels)-len(predict)).to(DEVICE)])

                
                if batch_idx < 2:
                    print(f"\n第{batch_idx}批次维度：")
                    print(f"predict: {predict.shape}, labels: {labels.shape}")  # 均为 [2]

                # 4. 计算损失和准确率
                loss = criterion(predict, labels)
                acc = binary_accuracy(predict, labels)

                # 5. 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 6. 累计指标
                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_acc += acc.item() * batch_size
                total_samples += batch_size

                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc.item():.4f}"})

        avg_loss = epoch_loss / total_samples
        avg_acc = epoch_acc / total_samples
        print(f"\nEpoch {epoch+1} 训练完成 | 平均损失：{avg_loss:.4f} | 平均准确率：{avg_acc:.4f}")

    torch.save({
        "bert_state_dict": bert_model.state_dict(),
        "fc_state_dict": fc_model.state_dict(),
        "avg_acc": avg_acc
    }, MODEL_SAVE_PATH)
    print(f"\n模型保存成功！路径：{os.path.abspath(MODEL_SAVE_PATH)}")

if __name__ == "__main__":
    train_local()