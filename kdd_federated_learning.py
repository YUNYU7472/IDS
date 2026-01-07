#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated Learning Implementation with NSL-KDD Dataset
A Simple Implementation of FedAvg with PyTorch on IID Data
支持训练模式和检测模式
"""

import numpy as np
import pandas as pd
import argparse
import os
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE

# Progress bar for training visualization
from tqdm import tqdm

pd.options.display.float_format = "{:,.4f}".format

# ============================================================================
# Device Configuration (GPU/CPU)
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Configuration Parameters
# ============================================================================
THREAT_TYPE = 'threat_type'

# 列名定义
colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'threat_type']

# 数值列名（用于特征提取）
numerical_colnames = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                      'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                      'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                      'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                      'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                      'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                      'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                      'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

# 威胁类型映射
THREAT_LABELS = {
    0: '正常',
    1: 'DoS攻击',
    2: 'U2R攻击',
    3: 'R2L攻击',
    4: '探测攻击',
    6: '未知'
}


# ============================================================================
# Neural Network Model Definition
# ============================================================================
class Net2nn(nn.Module):
    """Simple 3-layer neural network for classification"""
    def __init__(self, inputs, outputs):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(inputs, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# Data Preprocessing Functions
# ============================================================================
def encode_threat_types(df):
    """将威胁类型编码为数字标签"""
    df = df.copy()
    
    # 训练集和测试集的编码
    encoding_map = {
        'back': 1, 'neptune': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
        'apache2': 1, 'processtable': 1, 'mailbomb': 1, 'udpstorm': 1, 'worm': 1,
        'buffer_overflow': 2, 'loadmodule': 2, 'perl': 2, 'rootkit': 2,
        'ps': 2, 'xterm': 2, 'sqlattack': 2,
        'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3,
        'spy': 3, 'warezclient': 3, 'warezmaster': 3, 'snmpgetattack': 3,
        'httptunnel': 3, 'snmpguess': 3, 'named': 3, 'sendmail': 3, 'xlock': 3, 'xsnoop': 3,
        'ipsweep': 4, 'nmap': 4, 'portsweep': 4, 'satan': 4, 'mscan': 4, 'saint': 4,
        'normal': 0, 'unknown': 6
    }
    
    for threat, label in encoding_map.items():
        df.loc[df['threat_type'] == threat, 'threat_type'] = label
    
    return df


def normalize_data(df, max_values=None):
    """数据归一化处理"""
    threat_type_df = df['threat_type'].copy()
    
    # 提取数值列
    numerical_df = df[numerical_colnames].copy()
    
    # 移除常数列
    if numerical_df.shape[0] > 0:
        numerical_df = numerical_df.loc[:, (numerical_df != numerical_df.iloc[0]).any()]
    
    # 归一化（使用提供的max_values或计算新的）
    if max_values is None:
        max_values = numerical_df.max()
        # 避免除零
        max_values = max_values.replace(0, 1)
    
    final_df = numerical_df / max_values
    
    # 合并结果
    df_normalized = pd.concat([final_df, threat_type_df], axis=1)
    
    return df_normalized, max_values


def preprocess_new_data(df, max_values, colnames):
    """预处理新数据（用于检测）"""
    # 如果列数不匹配，可能需要调整
    if len(df.columns) == len(colnames) - 1:
        df.columns = colnames[:-1]  # 不包括threat_type
        # 添加虚拟的threat_type列
        df['threat_type'] = 0
    elif len(df.columns) == len(colnames):
        df.columns = colnames
    else:
        raise ValueError(f"数据列数不匹配！期望 {len(colnames)-1} 或 {len(colnames)} 列，得到 {len(df.columns)} 列")
    
    # 编码威胁类型（如果存在）
    df = encode_threat_types(df)
    
    # 归一化
    df_normalized, _ = normalize_data(df, max_values)
    
    # 提取特征（移除threat_type列）
    if 'threat_type' in df_normalized.columns:
        x_data = df_normalized.drop('threat_type', axis=1).values
    else:
        x_data = df_normalized.values
    
    return x_data


# ============================================================================
# Training Functions
# ============================================================================
def train(model, train_loader, criterion, optimizer):
    """训练模型一个epoch"""
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def validation(model, test_loader, criterion):
    """验证模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)


def divide_train_test(df, proportion=0.1):
    """将数据集划分为训练集和测试集"""
    df_train_list = []
    df_test_list = []
    for key, val in df[THREAT_TYPE].value_counts().items():
        df_part = df[df['threat_type'] == key]
        df_test_list.append(df_part[0: int(df_part.shape[0] * proportion)])
        df_train_list.append(df_part[int(df_part.shape[0] * proportion):df_part.shape[0]])

    return df_train_list, df_test_list


def get_data_for_slices(df_train_list, number_of_slices, isSmote=False, x_name="x_train", y_name="y_train"):
    """将数据分布到联邦节点"""
    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(number_of_slices):
        xname = x_name + str(i)
        yname = y_name + str(i)
        df_types = []

        for df in df_train_list:
            df_type = df[int(df.shape[0] * i / number_of_slices):int(df.shape[0] * (i + 1) / number_of_slices)]
            df_types.append(df_type)

        slice_df = pd.concat(df_types)
        y_info = slice_df.pop('threat_type').values
        x_info = slice_df.values
        y_info = y_info.astype('int')

        if isSmote:
            sm = SMOTE(random_state=42)
            x_info, y_info = sm.fit_resample(x_info, y_info)

        x_info = torch.tensor(x_info).float()
        y_info = torch.tensor(y_info).type(torch.LongTensor)

        x_data_dict.update({xname: x_info})
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


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


def get_averaged_weights(model_dict, name_of_models, number_of_slices):
    """计算所有本地模型的平均权重（FedAvg）"""
    fc1_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc1.weight.shape).to(device)
    fc1_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc1.bias.shape).to(device)

    fc2_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc2.weight.shape).to(device)
    fc2_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc2.bias.shape).to(device)

    fc3_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc3.weight.shape).to(device)
    fc3_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc3.bias.shape).to(device)

    with torch.no_grad():
        for i in range(number_of_slices):
            fc1_mean_weight += model_dict[name_of_models[i]].fc1.weight.data.clone()
            fc1_mean_bias += model_dict[name_of_models[i]].fc1.bias.data.clone()

            fc2_mean_weight += model_dict[name_of_models[i]].fc2.weight.data.clone()
            fc2_mean_bias += model_dict[name_of_models[i]].fc2.bias.data.clone()

            fc3_mean_weight += model_dict[name_of_models[i]].fc3.weight.data.clone()
            fc3_mean_bias += model_dict[name_of_models[i]].fc3.bias.data.clone()

        fc1_mean_weight = fc1_mean_weight / number_of_slices
        fc1_mean_bias = fc1_mean_bias / number_of_slices

        fc2_mean_weight = fc2_mean_weight / number_of_slices
        fc2_mean_bias = fc2_mean_bias / number_of_slices

        fc3_mean_weight = fc3_mean_weight / number_of_slices
        fc3_mean_bias = fc3_mean_bias / number_of_slices

    return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias


def set_averaged_weights_as_main_model_weights(main_model, model_dict, name_of_models, number_of_slices):
    """使用平均权重更新主模型"""
    fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias = get_averaged_weights(
        model_dict, name_of_models, number_of_slices)
    with torch.no_grad():
        main_model.fc1.weight.data = fc1_mean_weight.data.clone()
        main_model.fc2.weight.data = fc2_mean_weight.data.clone()
        main_model.fc3.weight.data = fc3_mean_weight.data.clone()

        main_model.fc1.bias.data = fc1_mean_bias.data.clone()
        main_model.fc2.bias.data = fc2_mean_bias.data.clone()
        main_model.fc3.bias.data = fc3_mean_bias.data.clone()
    return main_model


def send_main_model_to_nodes(main_model, model_dict, name_of_models, number_of_slices):
    """将主模型参数发送到所有本地节点"""
    with torch.no_grad():
        for i in range(number_of_slices):
            model_dict[name_of_models[i]].fc1.weight.data = main_model.fc1.weight.data.clone()
            model_dict[name_of_models[i]].fc2.weight.data = main_model.fc2.weight.data.clone()
            model_dict[name_of_models[i]].fc3.weight.data = main_model.fc3.weight.data.clone()

            model_dict[name_of_models[i]].fc1.bias.data = main_model.fc1.bias.data.clone()
            model_dict[name_of_models[i]].fc2.bias.data = main_model.fc2.bias.data.clone()
            model_dict[name_of_models[i]].fc3.bias.data = main_model.fc3.bias.data.clone()

    return model_dict


def train_node_models(model_dict, optimizer_dict, criterion_dict, name_of_models, name_of_x_train_sets, 
                      name_of_y_train_sets, x_train_dict, y_train_dict, x_test, y_test, 
                      number_of_slices, numEpoch, batch_size, print_progress=False):
    """训练所有本地节点模型"""
    for i in range(number_of_slices):
        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_models[i].replace('model', 'criterion')]
        optimizer = optimizer_dict[name_of_models[i].replace('model', 'optimizer')]

        # 为每个节点的epoch训练添加进度条
        epoch_pbar = tqdm(range(numEpoch), desc=f'节点 {i+1}/{number_of_slices} 训练', 
                         leave=False, ncols=100, unit='epoch')
        
        for epoch in epoch_pbar:
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer)
            # 实时更新进度条显示loss和accuracy
            epoch_pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'acc': f'{train_accuracy:.4f}',
                'epoch': f'{epoch+1}/{numEpoch}'
            })
        
        epoch_pbar.close()


# ============================================================================
# Model Save/Load Functions
# ============================================================================
def save_model(model, max_values, inputs, outputs, model_path):
    """保存模型和预处理参数"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'max_values': max_values,
        'inputs': inputs,
        'outputs': outputs,
        'model_class': 'Net2nn'
    }
    torch.save(save_dict, model_path)
    print(f"模型已保存到: {model_path}")


def load_model(model_path, device):
    """加载模型和预处理参数"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = Net2nn(checkpoint['inputs'], checkpoint['outputs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    max_values = checkpoint['max_values']
    inputs = checkpoint['inputs']
    outputs = checkpoint['outputs']
    
    print(f"模型已从 {model_path} 加载")
    print(f"输入特征数: {inputs}, 输出类别数: {outputs}")
    
    return model, max_values, inputs, outputs


# ============================================================================
# Detection Functions
# ============================================================================
def detect(model, data_path, max_values, output_path=None):
    """检测模式：对数据进行入侵检测"""
    print("=" * 80)
    print("开始检测...")
    print("=" * 80)
    
    # 加载数据
    print(f"正在加载数据: {data_path}")
    df = pd.read_csv(data_path, header=None)
    if df.shape[1] == 43:
        df = df.iloc[:, :-1]
    
    # 预处理数据
    print("正在预处理数据...")
    x_data = preprocess_new_data(df, max_values, colnames)
    
    # 转换为tensor
    x_tensor = torch.tensor(x_data).float().to(device)
    
    # 预测
    print("正在进行预测...")
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        # 批量处理
        batch_size = 1024
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i:i+batch_size]
            output = model(batch)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1).cpu().numpy()
            predictions.extend(pred)
            probabilities.extend(probs.cpu().numpy())
    
    # 转换预测结果
    threat_names = [THREAT_LABELS.get(pred, f'未知({pred})') for pred in predictions]
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '预测标签': predictions,
        '威胁类型': threat_names,
        '置信度': [probs[pred] for pred, probs in zip(predictions, probabilities)]
    })
    
    # 统计信息
    print("\n" + "=" * 80)
    print("检测结果统计:")
    print("=" * 80)
    print(results_df['威胁类型'].value_counts())
    print(f"\n总样本数: {len(results_df)}")
    print(f"正常流量: {(results_df['预测标签'] == 0).sum()}")
    print(f"攻击流量: {(results_df['预测标签'] != 0).sum()}")
    
    # 保存结果
    if output_path:
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n检测结果已保存到: {output_path}")
    else:
        # 默认保存路径
        default_output = data_path.replace('.csv', '_检测结果.csv')
        results_df.to_csv(default_output, index=False, encoding='utf-8-sig')
        print(f"\n检测结果已保存到: {default_output}")
    
    return results_df


# ============================================================================
# Training Mode
# ============================================================================
def train_mode(args):
    """训练模式"""
    print("=" * 80)
    print("训练模式")
    print("=" * 80)
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 80)
    
    # 配置参数
    learning_rate = args.learning_rate
    numEpoch = args.epochs
    batch_size = args.batch_size
    momentum = args.momentum
    number_of_slices = args.slices
    isSmote = args.smote
    data_path = args.data_path
    
    # 加载数据
    print("\n正在加载NSL-KDD数据集...")
    train_file = os.path.join(data_path, "KDDTrain+.csv")
    test_file = os.path.join(data_path, "KDDTest+.csv")
    
    df_train = pd.read_csv(train_file, header=None)
    df_train = df_train.iloc[:, :-1]
    
    df_test = pd.read_csv(test_file, header=None)
    df_test = df_test.iloc[:, :-1]
    
    df_train.columns = colnames
    df_test.columns = colnames
    
    # 编码威胁类型
    df_train = encode_threat_types(df_train)
    df_test = encode_threat_types(df_test)
    
    df_full = pd.concat([df_train, df_test])
    
    # 归一化
    print("\n正在进行数据归一化...")
    df_normalized, max_values = normalize_data(df_full)
    
    # 准备联邦学习数据
    print("\n正在准备联邦学习数据分布...")
    df_train_list, df_test_list = divide_train_test(df_normalized, proportion=0.1)
    
    x_train_dict, y_train_dict = get_data_for_slices(df_train_list, number_of_slices, isSmote)
    
    df_test_combined = pd.concat(df_test_list)
    y_test = df_test_combined.pop(THREAT_TYPE).values
    x_test = df_test_combined.values
    
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test.astype('int')).type(torch.LongTensor)
    
    inputs = x_test.shape[1]
    outputs = 5
    
    print(f"输入特征数: {inputs}, 输出类别数: {outputs}")
    
    # 创建主模型
    print("\n创建主模型...")
    main_model = Net2nn(inputs, outputs).to(device)
    main_criterion = nn.CrossEntropyLoss()
    
    # 创建节点模型
    print("创建节点模型...")
    model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(
        number_of_slices, inputs, outputs, learning_rate, momentum)
    
    name_of_x_train_sets = list(x_train_dict.keys())
    name_of_y_train_sets = list(y_train_dict.keys())
    name_of_models = list(model_dict.keys())
    
    # 联邦学习训练
    print("\n开始联邦学习训练...")
    num_iterations = args.iterations
    
    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)
    
    # 为联邦学习迭代添加主进度条
    iteration_pbar = tqdm(range(num_iterations), desc='联邦学习迭代', 
                         ncols=100, unit='iter', position=0)
    
    for iteration in iteration_pbar:
        # 更新迭代进度条描述
        iteration_pbar.set_description(f'联邦学习迭代 [{iteration+1}/{num_iterations}]')
        
        # 将主模型参数发送到节点
        model_dict = send_main_model_to_nodes(main_model, model_dict, name_of_models, number_of_slices)
        
        # 训练节点模型（内部会显示节点级别的进度条）
        train_node_models(model_dict, optimizer_dict, criterion_dict, name_of_models,
                         name_of_x_train_sets, name_of_y_train_sets, x_train_dict, y_train_dict,
                         x_test, y_test, number_of_slices, numEpoch, batch_size,
                         print_progress=False)
        
        # 聚合权重
        main_model = set_averaged_weights_as_main_model_weights(
            main_model, model_dict, name_of_models, number_of_slices)
        
        # 评估
        test_loss, test_accuracy = validation(main_model, test_dl, main_criterion)
        
        # 更新主进度条显示测试准确率
        iteration_pbar.set_postfix({
            'test_acc': f'{test_accuracy:.4f}',
            'test_loss': f'{test_loss:.4f}'
        })
    
    iteration_pbar.close()
    
    # 最终评估
    print("\n" + "=" * 80)
    print("最终模型评估")
    print("=" * 80)
    test_loss, test_accuracy = validation(main_model, test_dl, main_criterion)
    print(f"最终测试准确率: {test_accuracy:.4f}")
    
    # 保存模型
    model_path = args.model_path
    save_model(main_model, max_values, inputs, outputs, model_path)
    
    print("\n训练完成！")


# ============================================================================
# Detection Mode
# ============================================================================
def detection_mode(args):
    """检测模式"""
    print("=" * 80)
    print("检测模式")
    print("=" * 80)
    
    # 加载模型
    model, max_values, inputs, outputs = load_model(args.model_path, device)
    
    # 检测
    results = detect(model, args.data_path, max_values, args.output_path)
    
    print("\n检测完成！")


# ============================================================================
# Main Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='网络入侵检测系统 - 支持训练和检测模式')
    
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'detect'],
                       help='运行模式: train (训练) 或 detect (检测)')
    
    # 训练模式参数
    parser.add_argument('--data_path', type=str, default='data',
                       help='数据文件路径 (训练模式: 包含KDDTrain+.csv和KDDTest+.csv的目录; 检测模式: 要检测的CSV文件路径)')
    parser.add_argument('--model_path', type=str, default='model.pth',
                       help='模型文件路径 (训练模式: 保存路径; 检测模式: 加载路径)')
    
    # 训练特定参数
    parser.add_argument('--epochs', type=int, default=20,
                       help='每个节点的训练轮数 (默认: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='学习率 (默认: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='动量 (默认: 0.9)')
    parser.add_argument('--slices', type=int, default=2,
                       help='联邦学习节点数量 (默认: 2)')
    parser.add_argument('--iterations', type=int, default=5,
                       help='联邦学习迭代次数 (默认: 5)')
    parser.add_argument('--smote', action='store_true',
                       help='使用SMOTE进行数据平衡 (默认: False)')
    
    # 检测模式参数
    parser.add_argument('--output_path', type=str, default=None,
                       help='检测结果输出路径 (检测模式，默认: 自动生成)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'detect':
        if not os.path.exists(args.model_path):
            print(f"错误: 模型文件不存在: {args.model_path}")
            return
        if not os.path.exists(args.data_path):
            print(f"错误: 数据文件不存在: {args.data_path}")
            return
        detection_mode(args)


if __name__ == '__main__':
    main()
