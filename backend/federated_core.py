"""
联邦学习核心功能模块
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
import copy
from typing import Dict, List, Tuple
from config import Config
from model import Net2nn


class FederatedTrainer:
    """联邦学习训练器"""
    
    def __init__(self, num_clients=None):
        """
        初始化联邦训练器
        
        Args:
            num_clients: 客户端数量
        """
        self.num_clients = num_clients or Config.NUM_CLIENTS
        self.global_model = None
        self.client_models = {}
        self.client_optimizers = {}
        self.client_criterions = {}
        self.training_history = {
            'rounds': [],
            'global_accuracy': [],
            'global_loss': [],
            'client_accuracies': {i: [] for i in range(self.num_clients)},
            'client_losses': {i: [] for i in range(self.num_clients)}
        }
        
    def initialize_models(self, input_dim, output_dim):
        """
        初始化全局模型和所有客户端模型
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
        """
        # 创建全局模型
        self.global_model = Net2nn(inputs=input_dim, outputs=output_dim)
        
        # 为每个客户端创建模型
        for i in range(self.num_clients):
            # 创建模型
            model = Net2nn(inputs=input_dim, outputs=output_dim)
            # 复制全局模型权重
            model.load_state_dict(self.global_model.state_dict())
            self.client_models[i] = model
            
            # 创建优化器
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                momentum=Config.MOMENTUM
            )
            self.client_optimizers[i] = optimizer
            
            # 创建损失函数
            self.client_criterions[i] = nn.CrossEntropyLoss()
        
        print(f"✓ 初始化完成: 1个全局模型 + {self.num_clients}个客户端模型")
    
    def train_client(self, client_id, train_loader, num_epochs=1):
        """
        训练单个客户端的模型
        
        Args:
            client_id: 客户端ID
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        model = self.client_models[client_id]
        optimizer = self.client_optimizers[client_id]
        criterion = self.client_criterions[client_id]
        
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for epoch in range(num_epochs):
            for data, target in train_loader:
                # 前向传播
                output = model(data)
                loss = criterion(output, target)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                total_samples += len(target)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_model(self, model, test_loader):
        """
        验证模型性能
        
        Args:
            model: 要验证的模型
            test_loader: 测试数据加载器
            
        Returns:
            loss: 平均损失
            accuracy: 准确率
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                total_samples += len(target)
        
        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def get_confusion_matrix(self, model, test_loader):
        """
        计算混淆矩阵和指标
        
        Args:
            model: 模型
            test_loader: 测试数据加载器
            
        Returns:
            metrics: 包含混淆矩阵、精确率、召回率的字典
        """
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                prediction = output.argmax(dim=1)
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend(prediction.cpu().numpy())
        
        # 计算指标
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'confusion_matrix': cm.tolist(),
            'precision': float(precision),
            'recall': float(recall),
            'accuracy': float(accuracy)
        }
    
    def federated_averaging(self):
        """
        联邦平均算法 (FedAvg)
        聚合所有客户端模型的权重到全局模型
        """
        # 获取第一个客户端模型的权重形状
        first_model = self.client_models[0]
        
        # 初始化平均权重
        avg_weights = {
            'fc1_weight': torch.zeros_like(first_model.fc1.weight.data),
            'fc1_bias': torch.zeros_like(first_model.fc1.bias.data),
            'fc2_weight': torch.zeros_like(first_model.fc2.weight.data),
            'fc2_bias': torch.zeros_like(first_model.fc2.bias.data),
            'fc3_weight': torch.zeros_like(first_model.fc3.weight.data),
            'fc3_bias': torch.zeros_like(first_model.fc3.bias.data),
        }
        
        # 累加所有客户端的权重
        with torch.no_grad():
            for client_id in range(self.num_clients):
                model = self.client_models[client_id]
                
                avg_weights['fc1_weight'] += model.fc1.weight.data.clone()
                avg_weights['fc1_bias'] += model.fc1.bias.data.clone()
                avg_weights['fc2_weight'] += model.fc2.weight.data.clone()
                avg_weights['fc2_bias'] += model.fc2.bias.data.clone()
                avg_weights['fc3_weight'] += model.fc3.weight.data.clone()
                avg_weights['fc3_bias'] += model.fc3.bias.data.clone()
            
            # 计算平均值
            for key in avg_weights:
                avg_weights[key] /= self.num_clients
            
            # 更新全局模型
            self.global_model.fc1.weight.data = avg_weights['fc1_weight'].clone()
            self.global_model.fc1.bias.data = avg_weights['fc1_bias'].clone()
            self.global_model.fc2.weight.data = avg_weights['fc2_weight'].clone()
            self.global_model.fc2.bias.data = avg_weights['fc2_bias'].clone()
            self.global_model.fc3.weight.data = avg_weights['fc3_weight'].clone()
            self.global_model.fc3.bias.data = avg_weights['fc3_bias'].clone()
    
    def distribute_global_model(self):
        """
        将全局模型的权重分发给所有客户端
        """
        global_state = self.global_model.state_dict()
        
        for client_id in range(self.num_clients):
            self.client_models[client_id].load_state_dict(
                copy.deepcopy(global_state)
            )
    
    def federated_training_round(self, client_data_loaders, test_loader, 
                                 current_round):
        """
        执行一轮完整的联邦学习
        
        Args:
            client_data_loaders: 字典，每个客户端的DataLoader
            test_loader: 测试数据加载器
            current_round: 当前轮次
            
        Returns:
            round_metrics: 本轮的性能指标
        """
        print(f"\n=== 联邦学习第 {current_round} 轮 ===")
        
        # 1. 各客户端本地训练
        client_metrics = []
        for client_id in range(self.num_clients):
            train_loader = client_data_loaders[client_id]
            loss, acc = self.train_client(client_id, train_loader, num_epochs=1)
            
            client_metrics.append({
                'client_id': client_id,
                'loss': loss,
                'accuracy': acc
            })
            
            print(f"  客户端 {client_id}: Loss={loss:.4f}, Accuracy={acc:.4f}")
        
        # 2. 聚合模型（FedAvg）
        print("  聚合模型中...")
        self.federated_averaging()
        
        # 3. 评估全局模型
        global_loss, global_acc = self.validate_model(self.global_model, test_loader)
        print(f"  全局模型: Loss={global_loss:.4f}, Accuracy={global_acc:.4f}")
        
        # 4. 分发全局模型
        self.distribute_global_model()
        
        # 5. 记录历史
        self.training_history['rounds'].append(current_round)
        self.training_history['global_accuracy'].append(global_acc)
        self.training_history['global_loss'].append(global_loss)
        
        for cm in client_metrics:
            client_id = cm['client_id']
            self.training_history['client_accuracies'][client_id].append(cm['accuracy'])
            self.training_history['client_losses'][client_id].append(cm['loss'])
        
        return {
            'round': current_round,
            'global_loss': global_loss,
            'global_accuracy': global_acc,
            'client_metrics': client_metrics
        }
    
    def train(self, client_data_loaders, test_loader, num_rounds):
        """
        执行完整的联邦学习训练
        
        Args:
            client_data_loaders: 客户端数据加载器字典
            test_loader: 测试数据加载器
            num_rounds: 训练轮数
            
        Returns:
            training_history: 训练历史
        """
        print(f"\n开始联邦学习训练 ({num_rounds} 轮)...")
        
        for round_num in range(1, num_rounds + 1):
            self.federated_training_round(
                client_data_loaders, 
                test_loader, 
                round_num
            )
        
        print("\n✓ 训练完成！")
        return self.training_history
    
    def get_training_history(self):
        """获取训练历史"""
        return self.training_history
    
    def save_global_model(self, path):
        """保存全局模型"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'training_history': self.training_history
        }, path)
        print(f"✓ 全局模型已保存到 {path}")
    
    def load_global_model(self, path):
        """加载全局模型"""
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        print(f"✓ 全局模型已从 {path} 加载")


def train_centralized_model(train_loader, test_loader, num_epochs=20):
    """
    训练中心化模型（用于对比）
    
    Args:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_epochs: 训练轮数
        
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    # 获取输入输出维度
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1]
    output_dim = Config.OUTPUT_DIM
    
    # 创建模型
    model = Net2nn(inputs=input_dim, outputs=output_dim)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM
    )
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    print("\n开始中心化训练...")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            prediction = output.argmax(dim=1)
            correct += prediction.eq(target).sum().item()
            total += len(target)
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # 测试
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                prediction = output.argmax(dim=1)
                correct += prediction.eq(target).sum().item()
                total += len(target)
        
        test_loss /= len(test_loader)
        test_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    
    print("✓ 中心化训练完成！")
    return model, history


if __name__ == "__main__":
    print("联邦学习核心模块加载成功！")
