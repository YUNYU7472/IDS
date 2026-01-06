"""
配置文件 - 统一管理所有系统参数
"""
import os

class Config:
    """系统配置类"""
    
    # ============ 路径配置 ============
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # ============ 模型参数 ============
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    MOMENTUM = 0.9
    
    # 模型结构
    INPUT_DIM = 122  # NSL-KDD特征维度
    HIDDEN_DIM = 200
    OUTPUT_DIM = 5   # 5类分类：normal, DoS, U2R, R2L, Probe
    
    # ============ 联邦学习参数 ============
    NUM_CLIENTS = 5  # 默认客户端数量
    AGGREGATION_METHOD = 'fedavg'
    MIN_CLIENTS = 2  # 最少参与训练的客户端数
    
    # ============ 攻击类型映射 ============
    ATTACK_MAPPING = {
        0: 'Normal',
        1: 'DoS',      # Denial of Service
        2: 'U2R',      # User to Root
        3: 'R2L',      # Remote to Local
        4: 'Probe'     # Probing
    }
    
    # 攻击类型到标签的映射（用于数据预处理）
    LABEL_MAPPING = {
        'normal': 0,
        # DoS attacks
        'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
        # U2R attacks
        'buffer_overflow': 2, 'loadmodule': 2, 'perl': 2, 'rootkit': 2,
        # R2L attacks
        'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3,
        'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3,
        # Probe attacks
        'ipsweep': 4, 'nmap': 4, 'portsweep': 4, 'satan': 4
    }
    
    # ============ 数据预处理 ============
    # NSL-KDD 数据集列名
    COLUMN_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'threat_type'
    ]
    
    # 分类特征（需要one-hot编码）
    CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
    
    # ============ API配置 ============
    API_HOST = '0.0.0.0'
    API_PORT = 8000
    
    # CORS配置
    CORS_ORIGINS = ["*"]
    
    # ============ 训练状态 ============
    TRAINING_STATUS = {
        'IDLE': 'idle',           # 空闲
        'TRAINING': 'training',   # 训练中
        'AGGREGATING': 'aggregating',  # 聚合中
        'COMPLETED': 'completed', # 已完成
        'ERROR': 'error'          # 错误
    }
    
    @classmethod
    def get_attack_name(cls, label):
        """根据标签获取攻击类型名称"""
        return cls.ATTACK_MAPPING.get(label, 'Unknown')
    
    @classmethod
    def get_label_from_attack(cls, attack_type):
        """根据攻击类型获取标签"""
        return cls.LABEL_MAPPING.get(attack_type.lower(), -1)
