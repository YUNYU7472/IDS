"""
FastAPI 后端服务器
提供入侵检测和联邦学习管理的REST API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np
import threading
import os

from config import Config
from model import Net2nn, load_model, save_model
from data_loader import KDDDataLoader
from federated_core import FederatedTrainer

# 创建FastAPI应用
app = FastAPI(
    title="联邦学习入侵检测系统",
    description="基于NSL-KDD数据集的联邦学习网络入侵检测API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
class SystemState:
    def __init__(self):
        self.global_model = None
        self.data_loader = None
        self.federated_trainer = None
        self.training_status = Config.TRAINING_STATUS['IDLE']
        self.training_progress = 0
        self.current_round = 0
        self.total_rounds = 0
        self.training_history = None
        self.test_data = None
        self.client_data_loaders = None
        self.test_loader = None
        
state = SystemState()

# ==================== Pydantic模型定义 ====================

class DetectionRequest(BaseModel):
    """检测请求"""
    features: List[float]
    
class DetectionResponse(BaseModel):
    """检测响应"""
    prediction: str
    label: int
    confidence: float
    probabilities: Dict[str, float]

class TrainingConfig(BaseModel):
    """训练配置"""
    num_clients: int = 5
    num_rounds: int = 10
    use_smote: bool = False

class TrainingStatus(BaseModel):
    """训练状态"""
    status: str
    progress: float
    current_round: int
    total_rounds: int
    message: Optional[str] = None

class MetricsResponse(BaseModel):
    """指标响应"""
    rounds: List[int]
    global_accuracy: List[float]
    global_loss: List[float]
    client_accuracies: Dict[int, List[float]]

# ==================== 辅助函数 ====================

def initialize_system():
    """初始化系统"""
    if state.data_loader is None:
        print("初始化数据加载器...")
        state.data_loader = KDDDataLoader()
        
    if state.global_model is None:
        # 尝试加载已有模型
        model_path = os.path.join(Config.MODEL_DIR, 'global_model.pth')
        if os.path.exists(model_path):
            print(f"加载已有模型: {model_path}")
            state.global_model = load_model(model_path)
        else:
            print("创建新模型...")
            state.global_model = Net2nn()

def preprocess_for_detection(features):
    """预处理单个样本用于检测"""
    # 确保特征数量正确
    if len(features) != Config.INPUT_DIM:
        raise ValueError(f"特征维度错误: 期望 {Config.INPUT_DIM}, 得到 {len(features)}")
    
    # 转换为张量
    tensor = torch.tensor([features], dtype=torch.float32)
    return tensor

def run_federated_training(config: TrainingConfig):
    """后台运行联邦学习训练"""
    try:
        state.training_status = Config.TRAINING_STATUS['TRAINING']
        state.current_round = 0
        state.total_rounds = config.num_rounds
        
        # 准备数据
        print("准备训练数据...")
        X_train, y_train, X_test, y_test = state.data_loader.prepare_data(
            use_smote=config.use_smote
        )
        
        # 分割客户端数据
        print("分割客户端数据...")
        client_data = state.data_loader.split_data_for_clients(
            X_train, y_train, num_clients=config.num_clients
        )
        
        # 创建DataLoader
        state.client_data_loaders = {}
        for client_id in range(config.num_clients):
            client_dataset = torch.utils.data.TensorDataset(
                client_data[f'client_{client_id}']['X'],
                client_data[f'client_{client_id}']['y']
            )
            state.client_data_loaders[client_id] = torch.utils.data.DataLoader(
                client_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True
            )
        
        # 创建测试DataLoader
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        state.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        )
        
        # 初始化联邦训练器
        print("初始化联邦训练器...")
        state.federated_trainer = FederatedTrainer(num_clients=config.num_clients)
        input_dim = X_train.shape[1]
        state.federated_trainer.initialize_models(input_dim, Config.OUTPUT_DIM)
        
        # 执行训练
        print("开始联邦学习训练...")
        for round_num in range(1, config.num_rounds + 1):
            state.current_round = round_num
            state.training_progress = (round_num / config.num_rounds) * 100
            
            state.federated_trainer.federated_training_round(
                state.client_data_loaders,
                state.test_loader,
                round_num
            )
        
        # 保存训练历史和模型
        state.training_history = state.federated_trainer.get_training_history()
        state.global_model = state.federated_trainer.global_model
        
        # 保存模型
        model_path = os.path.join(Config.MODEL_DIR, 'global_model.pth')
        state.federated_trainer.save_global_model(model_path)
        
        state.training_status = Config.TRAINING_STATUS['COMPLETED']
        print("✓ 联邦学习训练完成！")
        
    except Exception as e:
        state.training_status = Config.TRAINING_STATUS['ERROR']
        print(f"✗ 训练错误: {e}")
        import traceback
        traceback.print_exc()

# ==================== API端点 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    print("=" * 60)
    print("联邦学习入侵检测系统启动中...")
    print("=" * 60)
    initialize_system()
    print("✓ 系统初始化完成")
    print(f"API文档: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    print("=" * 60)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "联邦学习入侵检测系统",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": state.global_model is not None
    }

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_intrusion(request: DetectionRequest):
    """
    检测单个网络流量样本
    
    Args:
        request: 包含特征的检测请求
        
    Returns:
        检测结果（攻击类型、置信度等）
    """
    try:
        # 初始化系统（如果需要）
        initialize_system()
        
        if state.global_model is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        # 预处理
        input_tensor = preprocess_for_detection(request.features)
        
        # 预测
        pred_class, probs = state.global_model.predict(input_tensor)
        
        # 构造响应
        label = pred_class.item()
        prediction = Config.get_attack_name(label)
        confidence = probs[0, label].item()
        
        # 各类别概率
        probabilities = {
            Config.get_attack_name(i): float(probs[0, i].item())
            for i in range(Config.OUTPUT_DIM)
        }
        
        return DetectionResponse(
            prediction=prediction,
            label=label,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fl/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    启动联邦学习训练
    
    Args:
        config: 训练配置
        
    Returns:
        训练任务信息
    """
    # 检查是否已在训练
    if state.training_status == Config.TRAINING_STATUS['TRAINING']:
        raise HTTPException(status_code=400, detail="训练正在进行中")
    
    # 在后台启动训练
    background_tasks.add_task(run_federated_training, config)
    
    return {
        "message": "联邦学习训练已启动",
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "use_smote": config.use_smote
    }

@app.get("/api/fl/status", response_model=TrainingStatus)
async def get_training_status():
    """
    获取训练状态
    
    Returns:
        当前训练状态
    """
    return TrainingStatus(
        status=state.training_status,
        progress=state.training_progress,
        current_round=state.current_round,
        total_rounds=state.total_rounds,
        message=f"训练轮次 {state.current_round}/{state.total_rounds}"
    )

@app.get("/api/fl/metrics", response_model=MetricsResponse)
async def get_training_metrics():
    """
    获取训练指标（准确率、损失等）
    
    Returns:
        训练历史数据
    """
    if state.training_history is None:
        raise HTTPException(status_code=404, detail="暂无训练历史数据")
    
    return MetricsResponse(
        rounds=state.training_history['rounds'],
        global_accuracy=state.training_history['global_accuracy'],
        global_loss=state.training_history['global_loss'],
        client_accuracies=state.training_history['client_accuracies']
    )

@app.post("/api/fl/stop")
async def stop_training():
    """停止训练"""
    if state.training_status != Config.TRAINING_STATUS['TRAINING']:
        raise HTTPException(status_code=400, detail="没有正在进行的训练")
    
    # TODO: 实现训练停止逻辑
    state.training_status = Config.TRAINING_STATUS['IDLE']
    
    return {"message": "训练已停止"}

@app.get("/api/config")
async def get_config():
    """获取系统配置"""
    return {
        "learning_rate": Config.LEARNING_RATE,
        "num_epochs": Config.NUM_EPOCHS,
        "batch_size": Config.BATCH_SIZE,
        "num_clients": Config.NUM_CLIENTS,
        "attack_types": Config.ATTACK_MAPPING
    }

# 前端相关代码已移除 - 专注于后端开发
# 前端文件保留在 frontend/ 目录，后续需要时可重新集成

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )
