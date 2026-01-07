#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版本的联邦学习入侵检测系统训练脚本
Performance-Optimized Training Script for Intrusion Detection
基于kdd-hl配置（准确率可达97.5%）+ GPU加速
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import random
import time
import json
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE
import copy

pd.options.display.float_format = "{:,.4f}".format

# ============================================================================
# GPU Configuration (自动使用最优设备)
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print("🚀 性能优化版训练脚本启动")
print("=" * 80)
print(f"📊 使用设备: {device}")
if device.type == 'cuda':
    print(f"🎮 GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # 启用cudnn自动优化
    torch.backends.cudnn.benchmark = True
    print("⚡ CuDNN自动优化: 已启用")
print("=" * 80)

# ============================================================================
# 最优配置参数（基于实验结果）
# ============================================================================
THREAT_TYPE = 'threat_type'

# 训练超参数 (最优配置 - 请勿随意修改)
learning_rate = 0.01        # 学习率
numEpoch = 20               # 训练轮数
batch_size = 32             # 批次大小 ⚠️ 最优值，不要改成64
momentum = 0.9              # SGD动量
print_amount = 3            # 每个epoch打印次数
number_of_slices = 2        # 联邦学习节点数（2个节点表现最好）
isSmote = True              # 使用SMOTE数据平衡（显著提升性能）

# 模型保存配置
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f"best_model_{timestamp}.pth"
results_save_path = f"training_results_{timestamp}.json"

print(f"\n📋 训练配置:")
print(f"   - 学习率: {learning_rate}")
print(f"   - 训练轮数: {numEpoch}")
print(f"   - 批次大小: {batch_size}")
print(f"   - 联邦节点数: {number_of_slices}")
print(f"   - SMOTE数据平衡: {'✓ 启用' if isSmote else '✗ 禁用'}")
print(f"   - 模型保存路径: {model_save_path}")
print("=" * 80)

data_path = "./data/"

# 数据列定义
colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'threat_type']

# ============================================================================
# 神经网络模型定义
# ============================================================================
class Net2nn(nn.Module):
    """3层全连接神经网络 (优化版本)"""
    def __init__(self, inputs, outputs):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(inputs, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, outputs)
        
        # 权重初始化 (Xavier初始化，提升性能)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============================================================================
# 数据加载和预处理
# ============================================================================
print("\n📂 加载NSL-KDD数据集...")
start_time = time.time()

df_train = pd.read_csv(data_path + "KDDTrain+.csv", header=None)
df_train = df_train.iloc[:, :-1]

df_test = pd.read_csv(data_path + "KDDTest+.csv", header=None)
df_test = df_test.iloc[:, :-1]

df_train.columns = colnames
df_test.columns = colnames

print(f"   ✓ 训练集: {df_train.shape[0]} 样本")
print(f"   ✓ 测试集: {df_test.shape[0]} 样本")

# 威胁类型编码 (训练集)
print("\n🔄 编码威胁类型...")
threat_mapping = {
    'normal': 0,
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,  # DoS
    'buffer_overflow': 2, 'loadmodule': 2, 'perl': 2, 'rootkit': 2,  # U2R
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3,  # R2L
    'spy': 3, 'warezclient': 3, 'warezmaster': 3,
    'ipsweep': 4, 'nmap': 4, 'portsweep': 4, 'satan': 4,  # Probe
    'unknown': 6
}

for threat, code in tqdm(threat_mapping.items(), desc='编码训练集威胁类型', ncols=80):
    df_train.loc[df_train['threat_type'] == threat, 'threat_type'] = code
    df_test.loc[df_test['threat_type'] == threat, 'threat_type'] = code

# 测试集额外威胁类型
test_extra_mapping = {
    'apache2': 1, 'processtable': 1, 'mailbomb': 1, 'udpstorm': 1, 'worm': 1,
    'ps': 2, 'xterm': 2, 'sqlattack': 2,
    'snmpgetattack': 3, 'httptunnel': 3, 'snmpguess': 3, 'named': 3,
    'sendmail': 3, 'xlock': 3, 'xsnoop': 3,
    'mscan': 4, 'saint': 4
}

for threat, code in tqdm(test_extra_mapping.items(), desc='编码测试集额外类型', ncols=80):
    df_test.loc[df_test['threat_type'] == threat, 'threat_type'] = code

# 合并数据集
df_full = pd.concat([df_train, df_test])

# 确保威胁类型列为数值类型，将任何非数值标签转换为0（正常）
df_full[THREAT_TYPE] = pd.to_numeric(df_full[THREAT_TYPE], errors='coerce').fillna(0).astype(int)

# 高级威胁分类 (0=正常, 1=攻击)
df_full.loc[(df_full[THREAT_TYPE] != 0), THREAT_TYPE] = 1
print(f"\n📊 威胁类型分布:")
print(df_full[THREAT_TYPE].value_counts())

# 数据归一化
print(f"\n🔄 数据归一化处理...")
threat_type_df = df_full['threat_type'].copy()

# 验证并清理威胁类型数据
print(f"   威胁类型唯一值: {sorted(threat_type_df.unique())}")
if threat_type_df.isna().sum() > 0:
    print(f"   ⚠️ 发现 {threat_type_df.isna().sum()} 个NaN值，已填充为0")
    threat_type_df = threat_type_df.fillna(0)
threat_type_df = threat_type_df.astype(int)

numerical_columns = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                     'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                     'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                     'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                     'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                     'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                     'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                     'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

numerical_df_full = df_full[numerical_columns].copy()
numerical_df_full = numerical_df_full.loc[:, (numerical_df_full != numerical_df_full.iloc[0]).any()]

# 保存最大值用于后续归一化
max_values = numerical_df_full.max()

# 归一化到[0,1]
final_df_full = numerical_df_full / numerical_df_full.max()
df_normalized = pd.concat([final_df_full, threat_type_df], axis=1)

print(f"   ✓ 归一化后数据维度: {df_normalized.shape}")
print(f"⏱️  数据加载耗时: {time.time() - start_time:.2f}秒")

# ============================================================================
# 辅助函数
# ============================================================================
def divide_train_test(df, proportion=0.1):
    """划分训练集和测试集（保持类别混合）"""
    # 按类别分层采样，然后合并
    df_test_parts = []
    df_train_parts = []
    
    for key in df[THREAT_TYPE].unique():
        df_part = df[df[THREAT_TYPE] == key].copy()
        split_idx = int(df_part.shape[0] * proportion)
        df_test_parts.append(df_part.iloc[:split_idx])
        df_train_parts.append(df_part.iloc[split_idx:])
    
    # 合并并打乱
    df_test = pd.concat(df_test_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = pd.concat(df_train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_train, df_test

def get_data_for_slices(df_train, number_of_slices, isSmote=False):
    """为每个联邦节点准备数据（确保每个节点都有混合类别）"""
    x_data_dict = dict()
    y_data_dict = dict()
    
    # 将训练数据平均分配给各个节点
    total_samples = len(df_train)
    samples_per_slice = total_samples // number_of_slices
    
    for i in range(number_of_slices):
        xname = "x_train" + str(i)
        yname = "y_train" + str(i)
        
        # 每个节点获取一部分数据
        start_idx = i * samples_per_slice
        if i == number_of_slices - 1:
            # 最后一个节点获取剩余所有数据
            df_part = df_train.iloc[start_idx:].copy()
        else:
            end_idx = (i + 1) * samples_per_slice
            df_part = df_train.iloc[start_idx:end_idx].copy()
        
        y = df_part.pop(THREAT_TYPE).values
        x = df_part.values
        
        # 确保y为整数类型 (处理numpy数组)
        y = pd.Series(y)
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int).values
        
        # 显示节点数据分布
        unique, counts = np.unique(y, return_counts=True)
        print(f"         节点{i} - 样本数: {len(y)}, 类别分布: {dict(zip(unique, counts))}")
        
        # SMOTE数据平衡
        if isSmote:
            sm = SMOTE(random_state=42)
            try:
                x, y = sm.fit_resample(x, y)
                unique_after, counts_after = np.unique(y, return_counts=True)
                print(f"         节点{i} SMOTE后 - 样本数: {len(y)}, 类别分布: {dict(zip(unique_after, counts_after))}")
            except ValueError as e:
                print(f"⚠️  节点 {i} SMOTE警告: {e}，跳过SMOTE处理")
        
        x = torch.tensor(x).float()
        y = torch.tensor(y.astype('int')).type(torch.LongTensor)
        
        x_data_dict.update({xname: x})
        y_data_dict.update({yname: y})
    
    return x_data_dict, y_data_dict

# ============================================================================
# 训练和验证函数
# ============================================================================
def train(model, train_loader, criterion, optimizer, show_progress=False):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(train_loader, desc='Training', leave=False) if show_progress else train_loader
    
    for data, target in iterator:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if show_progress:
            iterator.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    accuracy = correct / total
    avg_loss = train_loss / len(train_loader)
    return avg_loss, accuracy

def validation(model, test_loader, criterion):
    """验证模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy

def confusion_mat(model, test_loader):
    """计算混淆矩阵"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return confusion_matrix(y_true, y_pred)

# ============================================================================
# 联邦学习核心函数
# ============================================================================
def create_model_optimizer_criterion_dict(number_of_slices, inputs, outputs, learning_rate, momentum):
    """为每个节点创建模型、优化器和损失函数"""
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_slices):
        model_name = "model" + str(i)
        model_info = Net2nn(inputs, outputs).to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def FedAvg(w):
    """联邦平均算法"""
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def train_model_one_iteration(model_dict, optimizer_dict, criterion_dict, x_train_dict,
                              y_train_dict, number_of_slices, batch_size, numEpoch):
    """训练所有节点模型一次迭代"""
    for i in range(number_of_slices):
        model_name = "model" + str(i)
        model = model_dict[model_name]
        
        optimizer_name = "optimizer" + str(i)
        optimizer = optimizer_dict[optimizer_name]
        
        criterion_name = "criterion" + str(i)
        criterion = criterion_dict[criterion_name]
        
        x_name = "x_train" + str(i)
        x_train = x_train_dict[x_name]
        
        y_name = "y_train" + str(i)
        y_train = y_train_dict[y_name]
        
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        print(f"\n      🔹 节点 {i+1}/{number_of_slices} 训练中...")
        epoch_pbar = tqdm(range(numEpoch), desc=f'      节点{i+1}', ncols=100, leave=True)
        for epoch in epoch_pbar:
            loss, acc = train(model, train_dl, criterion, optimizer, show_progress=False)
            epoch_pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})

# ============================================================================
# 主训练流程
# ============================================================================
print("\n" + "=" * 80)
print("🎯 开始训练流程")
print("=" * 80)

# 准备数据
print("\n📦 准备训练和测试数据...")
print(f"   数据集总样本数: {len(df_normalized)}")
print(f"   正常样本: {len(df_normalized[df_normalized[THREAT_TYPE]==0])}")
print(f"   攻击样本: {len(df_normalized[df_normalized[THREAT_TYPE]==1])}")
df_train, df_test = divide_train_test(df_normalized, proportion=0.1)

print(f"\n   训练集: {len(df_train)} 样本")
print(f"   测试集: {len(df_test)} 样本")

x_train_dict, y_train_dict = get_data_for_slices(df_train, number_of_slices, isSmote)

y_test = df_test.pop(THREAT_TYPE).values
x_test = df_test.values

x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test.astype('int')).type(torch.LongTensor)

inputs = x_test.shape[1]
outputs = 2  # 二分类: 正常(0) vs 攻击(1)

print(f"   ✓ 输入特征数: {inputs}")
print(f"   ✓ 输出类别数: {outputs}")

# 创建主模型
print("\n🏗️  创建主模型...")
main_model = Net2nn(inputs, outputs).to(device)
main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum)
main_criterion = nn.CrossEntropyLoss()

# 创建节点模型
print("🏗️  创建节点模型...")
model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(
    number_of_slices, inputs, outputs, learning_rate, momentum
)

# 训练前评估
print("\n📊 训练前模型性能...")
test_ds = TensorDataset(x_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2)
before_loss, before_acc = validation(main_model, test_dl, main_criterion)
print(f"   主模型准确率: {before_acc:.4f}")

# 联邦学习训练
print("\n🔄 开始联邦学习训练...")
train_start_time = time.time()

num_iterations = 10  # 联邦学习迭代次数（增加到10次以提升性能）

print(f"\n{'='*80}")
print(f"📡 联邦学习训练进度")
print(f"{'='*80}")

iteration_pbar = tqdm(range(num_iterations), desc='联邦学习迭代', ncols=100, position=0)

for iteration in iteration_pbar:
    iteration_pbar.set_description(f'📡 迭代 {iteration+1}/{num_iterations}')
    
    # 训练所有节点
    train_model_one_iteration(model_dict, optimizer_dict, criterion_dict,
                             x_train_dict, y_train_dict, number_of_slices,
                             batch_size, numEpoch)
    
    # 收集模型权重
    w = []
    for i in range(number_of_slices):
        model_name = "model" + str(i)
        w.append(copy.deepcopy(model_dict[model_name].state_dict()))
    
    # 联邦平均
    w_avg = FedAvg(w)
    
    # 更新主模型
    main_model.load_state_dict(w_avg)
    
    # 评估主模型
    loss, acc = validation(main_model, test_dl, main_criterion)
    iteration_pbar.set_postfix({'accuracy': f'{acc:.4f}', 'loss': f'{loss:.4f}'})
    print(f"\n   ✅ 迭代 {iteration+1} 完成 - 主模型准确率: {acc:.4f}")

train_time = time.time() - train_start_time
print(f"\n⏱️  总训练时间: {train_time:.2f}秒 ({train_time/60:.2f}分钟)")

# 最终评估
print("\n" + "=" * 80)
print("📊 最终模型评估")
print("=" * 80)

final_loss, final_acc = validation(main_model, test_dl, main_criterion)
print(f"\n🎯 最终准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")

# 混淆矩阵
cm = confusion_mat(main_model, test_dl)
print(f"\n混淆矩阵:")
print(cm)

# 计算精确率、召回率、F1分数
if outputs == 2:
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_dl:
            data = data.to(device)
            output = main_model(data)
            _, predicted = output.max(1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 详细指标:")
    print(f"   精确率 (Precision): {precision:.4f}")
    print(f"   召回率 (Recall): {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")

# ============================================================================
# 保存模型
# ============================================================================
print("\n" + "=" * 80)
print("💾 保存模型")
print("=" * 80)

# 保存模型权重和配置
save_dict = {
    'model_state_dict': main_model.state_dict(),
    'max_values': max_values.to_dict(),
    'numerical_columns': list(max_values.index),
    'inputs': inputs,
    'outputs': outputs,
    'accuracy': final_acc,
    'training_config': {
        'learning_rate': learning_rate,
        'numEpoch': numEpoch,
        'batch_size': batch_size,
        'momentum': momentum,
        'number_of_slices': number_of_slices,
        'isSmote': isSmote,
        'num_iterations': num_iterations
    },
    'timestamp': timestamp,
    'device': str(device)
}

torch.save(save_dict, model_save_path)
print(f"✅ 模型已保存: {model_save_path}")

# 保存训练结果
results = {
    'timestamp': timestamp,
    'final_accuracy': float(final_acc),
    'final_loss': float(final_loss),
    'training_time_seconds': train_time,
    'confusion_matrix': cm.tolist(),
    'config': save_dict['training_config']
}

if outputs == 2:
    results['precision'] = float(precision)
    results['recall'] = float(recall)
    results['f1_score'] = float(f1)

with open(results_save_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ 训练结果已保存: {results_save_path}")

print("\n" + "=" * 80)
print("🎉 训练完成！")
print("=" * 80)
print(f"\n📁 生成的文件:")
print(f"   1. 模型文件: {model_save_path}")
print(f"   2. 结果文件: {results_save_path}")
print(f"\n💡 使用此模型进行检测:")
print(f"   from intrusion_detection_system.intrusion_detector import IntrusionDetector")
print(f"   detector = IntrusionDetector('{model_save_path}')")
print("=" * 80)
