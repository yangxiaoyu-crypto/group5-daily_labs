from MRPCDataset import MRPCDataset

print("测试简化版MRPC数据加载...")

try:
    # 测试训练集
    train_dataset = MRPCDataset(train=True)
    print(f"训练集样本数: {len(train_dataset)}")
    print("第一个样本:")
    print(f"  标签: {train_dataset[0][1]}")
    print(f"  句子1: {train_dataset[0][0][0]}")
    print(f"  句子2: {train_dataset[0][0][1]}")
    
    # 测试测试集
    test_dataset = MRPCDataset(train=False)  
    print(f"测试集样本数: {len(test_dataset)}")
    
    print(" 数据加载成功！")
    print(f"总样本数: 训练集{len(train_dataset)} + 测试集{len(test_dataset)} = {len(train_dataset) + len(test_dataset)}")
    
except Exception as e:
    print(f" 数据加载失败: {e}")
    import traceback
    traceback.print_exc()
