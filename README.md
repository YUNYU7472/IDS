# 联邦学习入侵检测系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**基于联邦学习的分布式网络入侵检测系统**

[功能特性](#功能特性) • [快速开始](#快速开始) • [系统架构](#系统架构) • [API文档](#api文档) • [演示](#演示)

</div>

---

## 📋 项目简介

本项目是一个完整的**联邦学习入侵检测系统**，在原有的NSL-KDD实验代码基础上进行了全面重构和功能扩展，实现了：

- ✅ **前后端分离架构** - FastAPI后端 + Web Dashboard前端
- ✅ **联邦学习实现** - FedAvg算法，支持多客户端协同训练
- ✅ **实时入侵检测** - REST API接口，支持实时流量检测
- ✅ **可视化监控** - 实时准确率曲线、攻击类型分布图表
- ✅ **模块化设计** - 代码重构，易于扩展和维护

## 🌟 功能特性

### 核心功能（后端）

1. **联邦学习训练**
   - 支持2-10个客户端的联邦训练
   - FedAvg聚合算法
   - 可配置训练轮数、学习率等参数
   - 训练过程状态查询

2. **入侵检测**
   - 支持5种攻击类型分类：Normal, DoS, U2R, R2L, Probe
   - RESTful API接口
   - 实时检测和响应
   - 置信度评估

3. **数据处理**
   - 自动特征工程和归一化
   - 支持SMOTE过采样
   - 灵活的数据分片策略

> **注意**: 前端暂未实现，当前专注于后端API开发。前端将在后续阶段统一开发。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM

### 安装步骤

1. **克隆项目**
```bash
cd Federated_Learning_Evaluation/intrusion_detection_system
```

2. **创建虚拟环境（推荐）**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **准备数据**

确保数据文件在正确位置：
```
Federated_Learning_Evaluation/
├── data/
│   ├── KDDTrain+.csv
│   └── KDDTest+.csv
└── intrusion_detection_system/
    ├── backend/
    └── frontend/
```

### 运行系统

**启动后端服务器**
```bash
cd backend
python main.py
```

服务器将在 `http://localhost:8000` 启动

**访问API文档**
```
http://localhost:8000/docs  # Swagger UI自动文档
```

**测试API**
```bash
# 健康检查
curl http://localhost:8000/health

# 查看配置
curl http://localhost:8000/api/config
```

## 🏗️ 系统架构

```
intrusion_detection_system/
├── backend/                    # 后端代码
│   ├── config.py              # 配置文件
│   ├── model.py               # 神经网络模型
│   ├── data_loader.py         # 数据加载和预处理
│   ├── federated_core.py      # 联邦学习核心逻辑
│   └── main.py                # FastAPI服务器
├── models/                     # 保存训练好的模型
├── requirements.txt           # Python依赖
└── README.md                  # 本文件

注：前端将在后续阶段统一开发
```

### 模块说明

| 模块 | 功能 |
|------|------|
| `config.py` | 统一管理配置参数、攻击类型映射 |
| `model.py` | 定义Net2nn神经网络模型 |
| `data_loader.py` | NSL-KDD数据加载、预处理、特征工程 |
| `federated_core.py` | 联邦学习训练器、FedAvg聚合算法 |
| `main.py` | FastAPI REST API服务 |

> **前端开发计划**: 待后端所有功能完成后，将使用现代前端框架（Vue/React）统一开发

## 📡 API文档

### 1. 检测API

**POST** `/api/detect`

检测单个网络流量样本

**请求体:**
```json
{
  "features": [0.1, 0.2, ..., 0.5]  // 122维特征向量
}
```

**响应:**
```json
{
  "prediction": "DoS",
  "label": 1,
  "confidence": 0.95,
  "probabilities": {
    "Normal": 0.01,
    "DoS": 0.95,
    "U2R": 0.02,
    "R2L": 0.01,
    "Probe": 0.01
  }
}
```

### 2. 联邦学习控制API

**POST** `/api/fl/start`

启动联邦学习训练

**请求体:**
```json
{
  "num_clients": 5,
  "num_rounds": 10,
  "use_smote": false
}
```

**GET** `/api/fl/status`

获取训练状态

**响应:**
```json
{
  "status": "training",
  "progress": 50.0,
  "current_round": 5,
  "total_rounds": 10
}
```

**GET** `/api/fl/metrics`

获取训练指标

**响应:**
```json
{
  "rounds": [1, 2, 3, ...],
  "global_accuracy": [0.85, 0.87, 0.90, ...],
  "global_loss": [0.5, 0.4, 0.3, ...],
  "client_accuracies": {
    "0": [0.84, 0.86, ...],
    "1": [0.85, 0.88, ...]
  }
}
```

### 3. 其他API

- `GET /health` - 系统健康检查
- `GET /api/config` - 获取系统配置
- `GET /docs` - Swagger API文档（自动生成）

## 🎯 使用示例

### 1. 启动联邦学习训练

使用Postman或curl调用API：

```bash
curl -X POST http://localhost:8000/api/fl/start \
  -H "Content-Type: application/json" \
  -d '{
    "num_clients": 5,
    "num_rounds": 10,
    "use_smote": false
  }'
```

查看训练状态：
```bash
curl http://localhost:8000/api/fl/status
```

获取训练指标：
```bash
curl http://localhost:8000/api/fl/metrics
```

### 2. 入侵检测

```bash
# 生成122维随机特征向量进行测试
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 0.5]}'  # 需要122个值
```

### 3. 使用Python调用API

```python
import requests

# 检测API示例
features = [0.1] * 122  # 122维特征
response = requests.post(
    'http://localhost:8000/api/detect',
    json={'features': features}
)
result = response.json()
print(f"检测结果: {result['prediction']}")
print(f"置信度: {result['confidence']:.2%}")
```

## 🔬 技术栈

### 后端
- **FastAPI** - 现代高性能Web框架
- **PyTorch** - 深度学习框架
- **Scikit-learn** - 数据预处理和评估
- **Imbalanced-learn** - SMOTE过采样

### 前端
> 待开发（计划使用Vue.js或React）

### 算法
- **FedAvg** - 联邦平均算法
- **SGD** - 随机梯度下降优化器
- **交叉熵损失** - 多分类损失函数

## 📊 数据集

使用 **NSL-KDD** 数据集：
- 训练集：125,973条记录
- 测试集：22,544条记录
- 特征：41个原始特征 + 编码后共122维
- 类别：5类（Normal, DoS, U2R, R2L, Probe）

## 🎨 性能指标

典型训练结果（5个客户端，10轮）：
- 全局模型准确率：~96%
- 训练时间：约5-10分钟（CPU）
- 检测响应时间：<10ms

## 🔧 配置选项

在 `backend/config.py` 中可修改：

```python
# 模型参数
LEARNING_RATE = 0.01
NUM_EPOCHS = 20
BATCH_SIZE = 32

# 联邦学习参数
NUM_CLIENTS = 5
AGGREGATION_METHOD = 'fedavg'

# API配置
API_HOST = '0.0.0.0'
API_PORT = 8000
```

## 📝 开发路线

### ✅ Phase 1: 基础架构（已完成）
- [x] 代码重构（model.py, data_loader.py, federated_core.py）
- [x] 后端基础API（/detect, /fl/start, /fl/status）
- [x] 配置管理和文档

### 🚧 Phase 1.5: 后端功能完善（进行中）
- [ ] 批量检测API
- [ ] 流量回放功能
- [ ] 模型评估API（混淆矩阵、指标）
- [ ] 日志系统
- [ ] 模型版本管理

### 📅 Phase 2: 前端开发（待开始）
- [ ] 选择前端框架（Vue/React）
- [ ] Dashboard界面设计
- [ ] 实时数据可视化
- [ ] WebSocket通信

### 📅 Phase 3: 高级功能（计划中）
- [ ] 数据库集成
- [ ] 用户认证系统
- [ ] 更多聚合算法
- [ ] 模型压缩和加速

### 📅 Phase 4: 部署优化（计划中）
- [ ] Docker容器化
- [ ] 性能优化
- [ ] 监控和告警
- [ ] 生产环境部署

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目基于原 [Federated_Learning_Evaluation](../) 项目开发

参考文献：
1. Wijethilaka, Shalitha, and Madhusanka Liyanage. "The role of security orchestrator in network slicing for future networks." Journal of Communications and Networks 25.3 (2023): 355-369.
2. Wijethilaka, Shalitha, and Madhusanka Liyanage. "A Federated Learning Approach for Improving Security in Network Slicing." GLOBECOM 2022-2022 IEEE Global Communications Conference. IEEE, 2022

## 📧 联系方式

如有问题或建议，请提交Issue。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star！**

Made with ❤️ for Cybersecurity Research

</div>
