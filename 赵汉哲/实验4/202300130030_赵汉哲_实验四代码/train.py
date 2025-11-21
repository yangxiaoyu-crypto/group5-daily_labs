import os
# 配置Hugging Face镜像源（清华大学镜像，稳定且速度快）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader
from FCModel import FCModel# 该项目中也是自己写的全连接层
from MRPCDataset import MRPCDataset# 该项目中自己写的Dataset
from transformers import BertTokenizer,BertModel#huggingface预训练BERT所需要的组件

#载入数据预处理模块
mrpcDataset=MRPCDataset()
train_loader=DataLoader(dataset=mrpcDataset,batch_size=16, shuffle=True)
print("数据载入完成")

#设置运行设备
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("设备配置完成")

#加载bert模型
tokenizer =BertTokenizer.from_pretrained("bert-base-uncased")
bert_model=BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
print("bert层模型创建完成")

#创建模型对象
model =FCModel()
model = model.to(device)
print("全连接层模型创建完成")

#定义优化器&损失函数
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)
bert_optimizer=torch.optim.Adam(bert_model.parameters(),lr=0.001)
crit =torch.nn.BCELoss()

#计算准确率的公式
def binary_accuracy(predict, label):
    rounded_predict=torch.round(predict)
    correct = (rounded_predict== label).float()
    accuracy =correct.sum() /len(correct)
    return accuracy

#定义训练方法

def train():
    #记录统计信息
    epoch_loss,epoch_acc= 0., 0.
    total_len= 0

    #分batch进行训练
    for i, data in enumerate(train_loader):
        print("torch.cuda.memory_allocated(): ",torch.cuda.memory_allocated())
        bert_model.train()
        model.train()

        sentence, label = data
        label =label.cuda()

        encoding =tokenizer(sentence,return_tensors='pt', padding=True, truncation=True)
        bert_output=bert_model(**encoding.to(device))
        pooler_output=bert_output.pooler_output
        predict = model(pooler_output).squeeze()

        loss =crit(predict,label.float())
        acc =binary_accuracy(predict, label)

        #gd
        optimizer.zero_grad() #把梯度重置为零
        bert_optimizer.zero_grad()
        loss.backward() #求导
        optimizer.step() #更新模型
        bert_optimizer.step()

        epoch_loss+= loss *len(label)
        epoch_acc+= acc *len(label)
        total_len+=len(label)

        print("batch %d loss:%f accuracy:%f" % (i, loss, acc))

    return epoch_loss/total_len,epoch_acc/total_len

#开始训练
Num_Epoch= 1
index = 0
for epoch in range(Num_Epoch):
    epoch_loss,epoch_acc=train()
    index += 1
    print("EPOCH %d loss:%f accuracy:%f" % (index,epoch_loss,epoch_acc))
