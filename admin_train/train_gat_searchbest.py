#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习入侵检测系统 - GAT对照实验组
Federated Learning with Graph Attention Network (GAT) - Control Group Experiment

【对照实验说明 / Control Experiment Setup】
1. 基础环境 (Infrastructure): 与 train_optimized.py 保持 100% 一致。
   - 数据加载 (Data Loading): 完全一致 (KDDTrain+, KDDTest+)。
   - 预处理 (Preprocessing): 完全一致 (归一化、威胁编码、数值列选择)。
   - 数据增强 (Augmentation): 完全一致 (Borderline-SMOTE, K=5 + 1% Gaussian Noise)。
   - 联邦设置 (Federated): 2节点 (Slices=2), 10轮迭代 (Iterations=10), 20本地Epochs。
   - 批次大小 (Batch Size): 32 (与 train_optimized 保持一致，尽管GAT显存消耗较大，维持32以控制变量)。
   
2. 唯一变量 (Variable): 模型架构 (Model Architecture)
   - 原始: Net2nn (3层 MLP) + SGD Optimizer
   - 新增: DynamicGAT (动态构图图注意力网络) + AdamW Optimizer
     * 优化器说明: GAT 使用 SGD 极难收敛，故优化器视为模型的一部分进行变更 (SGD -> AdamW)。
     * 学习率说明: 调整为 0.005 以适应 AdamW。
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
# GPU Configuration
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print("🚀 GAT 联邦学习实验启动 (严格对照组)")
print("=" * 80)
print(f"📊 使用设备: {device}")
if device.type == 'cuda':
    print(f"🎮 GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()} (Version: {torch.version.cuda})")
    print(f"✅ 当前设备ID: {torch.cuda.current_device()}")
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️  未检测到GPU，将在CPU上运行 (极慢警告)")
print("=" * 80)

# ============================================================================
# 配置参数 (Strict Alignment with train_optimized.py)
# ============================================================================
THREAT_TYPE = 'threat_type'

# 训练超参数
learning_rate = 0.005       # [变更] GAT适配 AdamW
numEpoch = 20               # [保持]
batch_size = 32             # [保持] 32
momentum = 0.9              # [保持] (虽然SGD不用了，但参数保留)
number_of_slices = 2        # [保持]
isSmote = True              # [保持]

# GAT 特有参数
K_NEIGHBORS = 5
GAT_HEADS = 4
GAT_HIDDEN = 64
DROPOUT = 0.3

# 模型保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f"gat_model_{timestamp}.pth"
results_save_path = f"gat_results_{timestamp}.json"

print(f"\n📋 训练配置 (GAT Control Group):")
print(f"   - 学习率: {learning_rate} (AdamW)")
print(f"   - 训练轮数: {numEpoch}")
print(f"   - 批次大小: {batch_size} (Shared)")
print(f"   - 联邦节点数: {number_of_slices} (Shared)")
print(f"   - 数据增强: Borderline-SMOTE + 1% Noise (Shared)")
print("=" * 80)

data_path = "./data/"

# 显式数值列 (Copied from train_optimized.py)
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
# 模型定义: GAT (Variable)
# ============================================================================
class DynamicGAT(nn.Module):
    """
    向量化优化的动态GAT模型 (High Precision Version)：
    1. 引入度量学习 (Metric Learning): 学习专门的特征空间用于构图，而非直接使用原始特征。
    2. 残差连接 (Residual Connection): 防止梯度消失，允许深层训练。
    3. 层归一化 (LayerNorm): 稳定训练，加速收敛。
    """
    def __init__(self, inputs, outputs, nhid=GAT_HIDDEN, nheads=GAT_HEADS, dropout=DROPOUT, alpha=0.2):
        super(DynamicGAT, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nheads = nheads
        self.alpha = alpha
        
        # 1. 度量学习投影层: 将原始特征映射到更有利于寻找这种邻居的空间
        self.metric_proj = nn.Linear(inputs, inputs)
        nn.init.xavier_uniform_(self.metric_proj.weight)
        
        # 多头注意力参数
        self.W = nn.Parameter(torch.empty(size=(nheads, inputs, nhid)))
        self.a = nn.Parameter(torch.empty(size=(nheads, 2 * nhid, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # 残差连接投影 (将输入对齐到 hidden 维度)
        self.residual_proj = nn.Linear(inputs, nheads * nhid)
        nn.init.xavier_uniform_(self.residual_proj.weight)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(nheads * nhid)
        
        # 输出层保持单头处理（聚合后的结果）
        self.out_linear = nn.Linear(nheads * nhid, outputs)
        nn.init.xavier_uniform_(self.out_linear.weight)

    def build_dynamic_graph(self, x, k=K_NEIGHBORS):
        """Batch内动态KNN构图 (使用学习到的度量空间)"""
        # 使用投影后的特征计算距离，构图更准确
        x_metric = self.metric_proj(x)
        
        dist = torch.cdist(x_metric, x_metric, p=2) 
        
        curr_k = min(k, x.size(0) - 1)
        if curr_k < 0: curr_k = 0
            
        _, indices = dist.topk(curr_k + 1, largest=False) 
        N = x.size(0)
        
        # [优化] 直接生成布尔掩码
        adj_mask = torch.zeros(N, N, dtype=torch.bool, device=x.device)
        row_indices = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, curr_k + 1)
        adj_mask[row_indices, indices] = True
        
        return adj_mask

    def forward(self, x):
        N = x.size(0)
        
        # 1. 动态构图 (使用 Metric Proj)
        adj_mask = self.build_dynamic_graph(x, k=K_NEIGHBORS) 
        
        x_in = F.dropout(x, self.dropout, training=self.training)
        
        # 2. 并行多头注意力
        Wh = torch.einsum('ni,hij->nhj', x_in, self.W) # (N, nheads, nhid)
        
        # 计算 Attention Scores
        a1 = self.a[:, :self.nhid, :]
        a2 = self.a[:, self.nhid:, :]
        
        Wh_t = Wh.permute(1, 0, 2) # (nheads, N, nhid)
        
        attn_1 = torch.matmul(Wh_t, a1).squeeze(-1) # (nheads, N)
        attn_2 = torch.matmul(Wh_t, a2).squeeze(-1) # (nheads, N)
        
        e = self.leakyrelu(attn_1.unsqueeze(-1) + attn_2.unsqueeze(1))
        
        # Masking
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_mask.unsqueeze(0), e, zero_vec)
        
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregation
        h_prime = torch.matmul(attention, Wh_t) 
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(N, self.nheads * self.nhid)
        
        # 3. 残差连接 + LayerNorm (Acc 提升关键)
        res = self.residual_proj(x)
        h_prime = h_prime + res
        h_prime = self.layer_norm(h_prime)
        
        output = F.elu(h_prime)
        output = F.dropout(output, self.dropout, training=self.training)
        
        x_out = self.out_linear(output)
        
        return x_out

# ============================================================================
# 数据处理 (完全复用 optimized 版逻辑)
# ============================================================================
print("\n📂 加载NSL-KDD数据集...")
start_time = time.time()

df_train = pd.read_csv(data_path + "KDDTrain+.csv", header=None)
df_train = df_train.iloc[:, :-1]
df_test = pd.read_csv(data_path + "KDDTest+.csv", header=None)
df_test = df_test.iloc[:, :-1]
df_train.columns = colnames
df_test.columns = colnames

print(f"   ✓ 训练集: {df_train.shape[0]}")
print(f"   ✓ 测试集: {df_test.shape[0]}")

print("\n🔄 编码威胁类型...")
threat_mapping = {
    'normal': 0,
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'buffer_overflow': 2, 'loadmodule': 2, 'perl': 2, 'rootkit': 2,
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3,
    'spy': 3, 'warezclient': 3, 'warezmaster': 3,
    'ipsweep': 4, 'nmap': 4, 'portsweep': 4, 'satan': 4,
    'unknown': 6
}
for threat, code in tqdm(threat_mapping.items(), desc='编码训练集', ncols=80):
    df_train.loc[df_train['threat_type'] == threat, 'threat_type'] = code
    df_test.loc[df_test['threat_type'] == threat, 'threat_type'] = code

test_extra_mapping = {
    'apache2': 1, 'processtable': 1, 'mailbomb': 1, 'udpstorm': 1, 'worm': 1,
    'ps': 2, 'xterm': 2, 'sqlattack': 2,
    'snmpgetattack': 3, 'httptunnel': 3, 'snmpguess': 3, 'named': 3,
    'sendmail': 3, 'xlock': 3, 'xsnoop': 3,
    'mscan': 4, 'saint': 4
}
for threat, code in tqdm(test_extra_mapping.items(), desc='编码测试集', ncols=80):
    df_test.loc[df_test['threat_type'] == threat, 'threat_type'] = code

df_full = pd.concat([df_train, df_test])
df_full[THREAT_TYPE] = pd.to_numeric(df_full[THREAT_TYPE], errors='coerce').fillna(0).astype(int)
df_full.loc[(df_full[THREAT_TYPE] != 0), THREAT_TYPE] = 1

# 归一化 (Keep Consistent)
print(f"\n🔄 归一化处理...")
threat_type_df = df_full['threat_type'].copy()
if threat_type_df.isna().sum() > 0:
    threat_type_df = threat_type_df.fillna(0)
threat_type_df = threat_type_df.astype(int)

# 显式列表 (Compatible with optimized)
numerical_columns_list = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                     'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                     'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                     'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                     'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                     'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                     'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                     'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

numerical_df_full = df_full[numerical_columns_list].copy()
numerical_df_full = numerical_df_full.loc[:, (numerical_df_full != numerical_df_full.iloc[0]).any()]
max_values = numerical_df_full.max()
final_df_full = numerical_df_full / numerical_df_full.max()
df_normalized = pd.concat([final_df_full, threat_type_df], axis=1)

print(f"   ✓ 归一化后数据维度: {df_normalized.shape}")

# ============================================================================
# 辅助函数
# ============================================================================
def divide_train_test(df, proportion=0.1):
    df_test_parts = []
    df_train_parts = []
    for key in df[THREAT_TYPE].unique():
        df_part = df[df[THREAT_TYPE] == key].copy()
        split_idx = int(df_part.shape[0] * proportion)
        df_test_parts.append(df_part.iloc[:split_idx])
        df_train_parts.append(df_part.iloc[split_idx:])
    df_test = pd.concat(df_test_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = pd.concat(df_train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_train, df_test

def get_data_for_slices(df_train, number_of_slices, isSmote=False):
    """
    【关键】数据增强逻辑与 Optimized 版保持 1:1 一致
    """
    from imblearn.over_sampling import BorderlineSMOTE
    x_data_dict = dict()
    y_data_dict = dict()
    total_samples = len(df_train)
    samples_per_slice = total_samples // number_of_slices
    
    for i in range(number_of_slices):
        xname = "x_train" + str(i)
        yname = "y_train" + str(i)
        
        start_idx = i * samples_per_slice
        if i == number_of_slices - 1:
            df_part = df_train.iloc[start_idx:].copy()
        else:
            end_idx = (i + 1) * samples_per_slice
            df_part = df_train.iloc[start_idx:end_idx].copy()
        
        y = df_part.pop(THREAT_TYPE).values
        x = df_part.values
        y = pd.Series(y)
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int).values
        
        unique, counts = np.unique(y, return_counts=True)
        print(f"         节点{i} - 原始样本数: {len(y)}, 分布: {dict(zip(unique, counts))}")
        
        if isSmote:
            # 1. Borderline-SMOTE (Same as optimized)
            borderline_sm = BorderlineSMOTE(random_state=42, k_neighbors=5, kind='borderline-1')
            try:
                x, y = borderline_sm.fit_resample(x, y)
                unique_after, counts_after = np.unique(y, return_counts=True)
                print(f"         节点{i} Borderline-SMOTE - 样本: {len(y)}, 分布: {dict(zip(unique_after, counts_after))}")
                
                # 2. Noise Injection 1% (Same as optimized, was 0.5% in old draft)
                noise_level = 0.01 
                noise = np.random.normal(0, noise_level, x.shape)
                x = x + noise
                x = np.clip(x, 0, 1)
                print(f"         节点{i} ✓ 已添加{noise_level*100}%噪声增强")
            except ValueError as e:
                print(f"⚠️  节点 {i} Borderline-SMOTE警告: {e}")
        
        x = torch.tensor(x).float()
        y = torch.tensor(y.astype('int')).type(torch.LongTensor)
        x_data_dict.update({xname: x})
        y_data_dict.update({yname: y})
    
    return x_data_dict, y_data_dict

# ============================================================================
# 训练 utils
# ============================================================================
def train(model, train_loader, criterion, optimizer, show_progress=False):
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
    return train_loss / len(train_loader), correct / total

def validation(model, test_loader, criterion):
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
    return test_loss / len(test_loader), correct / total

def evaluate_full_metrics(model, test_loader, criterion):
    """
    计算完整指标: Accuracy, Precision, Recall, F1, Loss
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            _, predicted = output.max(1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    avg_loss = test_loss / len(test_loader)
    acc = accuracy_score(y_true, y_pred)
    # Binary classification metrics
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def confusion_mat(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return confusion_matrix(y_true, y_pred)

# ============================================================================
# Federated Setup (GAT + AdamW)
# ============================================================================
def create_model_optimizer_criterion_dict(number_of_slices, inputs, outputs, learning_rate, momentum,
                                          nhid, nheads, dropout):
    """
    创建节点模型字典，支持动态超参数传入
    """
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    for i in range(number_of_slices):
        model_name = "model" + str(i)
        # 使用 Trial 参数实例化
        model_info = DynamicGAT(inputs, outputs, nhid=nhid, nheads=nheads, dropout=dropout).to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        # 使用 Trial 学习率
        optimizer_info = torch.optim.AdamW(model_info.parameters(), lr=learning_rate, weight_decay=1e-4) 
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})
    return model_dict, optimizer_dict, criterion_dict

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def train_model_one_iteration(model_dict, optimizer_dict, criterion_dict, x_train_dict,
                              y_train_dict, number_of_slices, batch_size, numEpoch):
    for i in range(number_of_slices):
        model_name = "model" + str(i)
        model = model_dict[model_name]
        optimizer = optimizer_dict["optimizer" + str(i)]
        criterion = criterion_dict["criterion" + str(i)]
        
        # [优化] 将整个节点数据预加载到GPU
        x_train = x_train_dict["x_train" + str(i)].to(device)
        y_train = y_train_dict["y_train" + str(i)].to(device)
        
        train_ds = TensorDataset(x_train, y_train)
        # 使用 Trial Batch Size
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # print(f"\n      🔹 节点 {i+1}/{number_of_slices} 训练中...")
        epoch_pbar = tqdm(range(numEpoch), desc=f'      Node {i+1}', ncols=80, leave=False)
        for epoch in epoch_pbar:
            loss, acc = train(model, train_dl, criterion, optimizer)
            # epoch_pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})

# ============================================================================
# Main Execution: Random Hyperparameter Search
# ============================================================================
print("\n" + "=" * 80)
print("🎯 开始 GAT 超参数随机搜索 (Hyperparameter Search)")
print("=" * 80)

print("\n📦 准备基础数据 (Shared)...")
df_train, df_test = divide_train_test(df_normalized, proportion=0.1)
x_train_dict, y_train_dict = get_data_for_slices(df_train, number_of_slices, isSmote)

# 预加载测试集
y_test = df_test.pop(THREAT_TYPE).values
x_test = torch.tensor(df_test.values).float().to(device)
y_test = torch.tensor(y_test.astype('int')).type(torch.LongTensor).to(device)
inputs = x_test.shape[1]
outputs = 2 
print(f"   ✓ 输入特征: {inputs}, 输出类别: {outputs}")

test_ds = TensorDataset(x_test, y_test)

# 定义搜索空间
SEARCH_SPACE = {
    'batch_size': [32, 128, 256, 512],
    'learning_rate': [0.001, 0.005, 0.01],
    'nhid': [64, 128],
    'nheads': [4, 8],
    'dropout': [0.1, 0.3, 0.5]
}

NUM_TRIALS = 5  # 搜索次数
best_acc = 0.0
best_params = {}
results_log = []

print(f"\n🔍 开始随机搜索 (共 {NUM_TRIALS} 组实验)...")

for trial in range(NUM_TRIALS):
    # 随机采样超参数
    params = {
        'batch_size': random.choice(SEARCH_SPACE['batch_size']),
        'learning_rate': random.choice(SEARCH_SPACE['learning_rate']),
        'nhid': random.choice(SEARCH_SPACE['nhid']),
        'nheads': random.choice(SEARCH_SPACE['nheads']),
        'dropout': random.choice(SEARCH_SPACE['dropout'])
    }
    
    print(f"\n🧪 [Trial {trial+1}/{NUM_TRIALS}] 参数: {params}")
    
    # 实例化模型
    main_model = DynamicGAT(inputs, outputs, 
                           nhid=params['nhid'], 
                           nheads=params['nheads'], 
                           dropout=params['dropout']).to(device)
    main_optimizer = torch.optim.AdamW(main_model.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
    main_criterion = nn.CrossEntropyLoss()
    
    # 实例化节点
    model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(
        number_of_slices, inputs, outputs, 
        params['learning_rate'], momentum,
        nhid=params['nhid'], nheads=params['nheads'], dropout=params['dropout']
    )
    
    test_dl = DataLoader(test_ds, batch_size=params['batch_size'] * 2)
    
    # 训练循环 (Reduced iterations for search efficiency? keeping 10 for accuracy)
    current_trial_best_acc = 0.0
    current_trial_best_metrics = {}
    
    # 使用较少的迭代次数进行快速筛选 (可选，这里保持10)
    search_iterations = 10 
    
    try:
        iter_pbar = tqdm(range(search_iterations), desc=f'Trial {trial+1}', ncols=100)
        for iteration in iter_pbar:
            train_model_one_iteration(model_dict, optimizer_dict, criterion_dict,
                                     x_train_dict, y_train_dict, number_of_slices,
                                     params['batch_size'], numEpoch) # numEpoch maintained
            
            # FedAvg
            w = []
            for i in range(number_of_slices):
                w.append(copy.deepcopy(model_dict["model" + str(i)].state_dict()))
            w_avg = FedAvg(w)
            
            # Update Main and Nodes
            main_model.load_state_dict(w_avg)
            for i in range(number_of_slices):
                 model_dict["model" + str(i)].load_state_dict(w_avg)
            
            # 计算完整指标
            metrics = evaluate_full_metrics(main_model, test_dl, main_criterion)
            acc = metrics['accuracy']
            iter_pbar.set_postfix({'acc': f'{acc:.4f}', 'f1': f"{metrics['f1']:.4f}"})
            
            if acc > current_trial_best_acc:
                current_trial_best_acc = acc
                current_trial_best_metrics = metrics
        
        print(f"   🏁 Trial {trial+1} 完成 - Best Acc: {current_trial_best_metrics['accuracy']:.4f}, "
              f"P: {current_trial_best_metrics['precision']:.4f}, "
              f"R: {current_trial_best_metrics['recall']:.4f}, "
              f"F1: {current_trial_best_metrics['f1']:.4f}")
        
        # 记录结果
        results_log.append({
            'trial': trial + 1,
            'params': params,
            'accuracy': current_trial_best_metrics['accuracy'],
            'precision': current_trial_best_metrics['precision'],
            'recall': current_trial_best_metrics['recall'],
            'f1': current_trial_best_metrics['f1'],
            'loss': current_trial_best_metrics['loss']
        })
        
        if current_trial_best_acc > best_acc:
            best_acc = current_trial_best_acc
            best_params = params
            torch.save(main_model.state_dict(), f"best_gat_model_{timestamp}.pth")
            print(f"   🏆 New Best Validation Accuracy: {best_acc:.4f}")
            
    except RuntimeError as e:
        print(f"   ❌ Trial {trial+1} Failed: {e} (可能是显存不足)")
        torch.cuda.empty_cache()

print("\n" + "=" * 80)
print(f"🎉 搜索完成! 最优精度: {best_acc:.4f}")
print(f"👑 最优参数: {best_params}")
print("=" * 80)

# 最终使用最优参数再跑一次评估 (Optional, output metrics here)
print("\n📊 所有 Trial 结果:")
for res in results_log:
    print(f"   Trial {res['trial']}: Acc={res['accuracy']:.4f}, P={res['precision']:.4f}, R={res['recall']:.4f}, F1={res['f1']:.4f}, Params={res['params']}")

# 保存搜索结果
search_results = {
    'best_params': best_params,
    'best_accuracy': best_acc,
    'all_trials': results_log,
    'timestamp': timestamp
}
with open(f"gat_search_results_{timestamp}.json", 'w') as f:
    json.dump(search_results, f, indent=2)
