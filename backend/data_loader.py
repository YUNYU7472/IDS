"""
数据加载和预处理模块
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from config import Config


class KDDDataLoader:
    """NSL-KDD数据集加载器"""
    
    def __init__(self, data_path=None):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path or Config.DATA_DIR
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        
    def load_data(self, filename, has_difficulty=True):
        """
        加载NSL-KDD数据集
        
        Args:
            filename: 文件名（如'KDDTrain+.csv'）
            has_difficulty: 文件是否包含难度列（最后一列）
            
        Returns:
            df: 加载的DataFrame
        """
        import os
        filepath = os.path.join(self.data_path, filename)
        
        # 读取CSV文件
        df = pd.read_csv(filepath, header=None)
        
        # 如果有难度列，删除它
        if has_difficulty:
            df = df.iloc[:, :-1]
        
        # 设置列名
        df.columns = Config.COLUMN_NAMES
        
        return df
    
    def encode_labels(self, df):
        """
        将攻击类型编码为数字标签
        
        Args:
            df: 包含threat_type列的DataFrame
            
        Returns:
            df: 编码后的DataFrame
        """
        df = df.copy()
        
        # 映射攻击类型到标签
        for attack_type, label in Config.LABEL_MAPPING.items():
            df.loc[df['threat_type'] == attack_type, 'threat_type'] = label
        
        # 转换为整数
        df['threat_type'] = df['threat_type'].astype(int)
        
        return df
    
    def preprocess_features(self, df, fit=True):
        """
        预处理特征：编码分类特征和归一化
        
        Args:
            df: 输入DataFrame
            fit: 是否拟合编码器和缩放器（训练集用True，测试集用False）
            
        Returns:
            df: 预处理后的DataFrame
        """
        df = df.copy()
        
        # 1. 编码分类特征
        for col in Config.CATEGORICAL_FEATURES:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # 处理测试集中可能出现的新类别
                    df[col] = df[col].astype(str)
                    classes = self.label_encoders[col].classes_
                    df[col] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in classes else -1
                    )
        
        # 2. 分离特征和标签
        if 'threat_type' in df.columns:
            y = df.pop('threat_type')
            X = df
        else:
            X = df
            y = None
        
        # 3. 归一化数值特征
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # 转换回DataFrame
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        if y is not None:
            df_scaled['threat_type'] = y.values
        
        return df_scaled
    
    def prepare_data(self, train_file='KDDTrain+.csv', test_file='KDDTest+.csv', 
                     use_smote=False):
        """
        准备训练和测试数据
        
        Args:
            train_file: 训练数据文件名
            test_file: 测试数据文件名
            use_smote: 是否使用SMOTE过采样
            
        Returns:
            X_train, y_train, X_test, y_test: PyTorch张量
        """
        # 加载数据
        print("加载训练数据...")
        df_train = self.load_data(train_file)
        print("加载测试数据...")
        df_test = self.load_data(test_file)
        
        # 编码标签
        print("编码标签...")
        df_train = self.encode_labels(df_train)
        df_test = self.encode_labels(df_test)
        
        # 预处理特征
        print("预处理特征...")
        df_train = self.preprocess_features(df_train, fit=True)
        df_test = self.preprocess_features(df_test, fit=False)
        
        # 分离特征和标签
        y_train = df_train.pop('threat_type').values.astype(int)
        X_train = df_train.values
        
        y_test = df_test.pop('threat_type').values.astype(int)
        X_test = df_test.values
        
        # SMOTE过采样（可选）
        if use_smote:
            print("应用SMOTE过采样...")
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        
        print(f"\n数据集统计:")
        print(f"训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"测试集: X={X_test.shape}, y={y_test.shape}")
        print(f"\n训练集类别分布:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {Config.get_attack_name(label)}: {count}")
        
        # 转换为PyTorch张量
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        return X_train, y_train, X_test, y_test
    
    def create_data_loaders(self, X_train, y_train, X_test, y_test, 
                           batch_size=None):
        """
        创建PyTorch DataLoader
        
        Args:
            X_train, y_train, X_test, y_test: 数据张量
            batch_size: 批次大小
            
        Returns:
            train_loader, test_loader: DataLoader对象
        """
        batch_size = batch_size or Config.BATCH_SIZE
        
        # 创建Dataset
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def split_data_for_clients(self, X_train, y_train, num_clients=None):
        """
        将训练数据分割给多个客户端（联邦学习用）
        
        Args:
            X_train, y_train: 训练数据
            num_clients: 客户端数量
            
        Returns:
            client_data: 字典，包含每个客户端的数据
        """
        num_clients = num_clients or Config.NUM_CLIENTS
        
        # 将数据和标签转换为numpy数组（如果是张量）
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.numpy()
        
        # 按类别分组
        client_data = {}
        
        # 获取所有唯一类别
        unique_labels = np.unique(y_train)
        
        for client_id in range(num_clients):
            X_client_list = []
            y_client_list = []
            
            # 为每个类别分配数据切片
            for label in unique_labels:
                # 获取该类别的所有样本
                label_indices = np.where(y_train == label)[0]
                
                # 计算该客户端的数据范围
                start_idx = int(len(label_indices) * client_id / num_clients)
                end_idx = int(len(label_indices) * (client_id + 1) / num_clients)
                
                # 选择数据切片
                client_indices = label_indices[start_idx:end_idx]
                
                X_client_list.append(X_train[client_indices])
                y_client_list.append(y_train[client_indices])
            
            # 合并该客户端的所有数据
            X_client = np.vstack(X_client_list)
            y_client = np.concatenate(y_client_list)
            
            # 转换为张量
            X_client = torch.tensor(X_client, dtype=torch.float32)
            y_client = torch.tensor(y_client, dtype=torch.long)
            
            client_data[f'client_{client_id}'] = {
                'X': X_client,
                'y': y_client,
                'size': len(y_client)
            }
            
            print(f"客户端 {client_id}: {len(y_client)} 个样本")
            unique, counts = np.unique(y_client.numpy(), return_counts=True)
            for label, count in zip(unique, counts):
                print(f"  {Config.get_attack_name(label)}: {count}")
        
        return client_data
    
    def preprocess_single_sample(self, sample_dict):
        """
        预处理单个样本（用于实时检测）
        
        Args:
            sample_dict: 包含特征的字典
            
        Returns:
            preprocessed_tensor: 预处理后的张量
        """
        # 将字典转换为DataFrame
        df = pd.DataFrame([sample_dict])
        
        # 确保所有列都存在
        for col in Config.COLUMN_NAMES[:-1]:  # 除了threat_type
            if col not in df.columns:
                df[col] = 0
        
        # 预处理
        df = self.preprocess_features(df, fit=False)
        
        # 转换为张量
        tensor = torch.tensor(df.values, dtype=torch.float32)
        
        return tensor


if __name__ == "__main__":
    # 测试代码
    print("测试数据加载器...")
    
    loader = KDDDataLoader()
    
    # 测试数据准备
    try:
        X_train, y_train, X_test, y_test = loader.prepare_data()
        print("\n✓ 数据加载成功！")
        
        # 测试DataLoader创建
        train_loader, test_loader = loader.create_data_loaders(
            X_train, y_train, X_test, y_test
        )
        print(f"✓ DataLoader创建成功！")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  测试批次数: {len(test_loader)}")
        
        # 测试客户端数据分割
        client_data = loader.split_data_for_clients(X_train, y_train, num_clients=3)
        print(f"\n✓ 客户端数据分割成功！")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
