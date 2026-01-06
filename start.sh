#!/bin/bash

# 联邦学习入侵检测系统启动脚本

echo "=========================================="
echo "  联邦学习入侵检测系统"
echo "  Federated Learning Intrusion Detection"
echo "=========================================="
echo ""

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"
echo ""

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  建议在虚拟环境中运行"
    echo "创建虚拟环境: python3 -m venv venv"
    echo "激活虚拟环境: source venv/bin/activate"
    echo ""
fi

# 检查依赖
echo "检查依赖..."
if ! pip show fastapi > /dev/null 2>&1; then
    echo "❌ 依赖未安装，正在安装..."
    pip install -r requirements.txt
else
    echo "✓ 依赖已安装"
fi
echo ""

# 检查数据文件
echo "检查数据文件..."
data_dir="../data"
if [ -f "$data_dir/KDDTrain+.csv" ] && [ -f "$data_dir/KDDTest+.csv" ]; then
    echo "✓ 数据文件已就绪"
else
    echo "❌ 数据文件未找到"
    echo "请确保以下文件存在:"
    echo "  - $data_dir/KDDTrain+.csv"
    echo "  - $data_dir/KDDTest+.csv"
    exit 1
fi
echo ""

# 创建模型目录
if [ ! -d "models" ]; then
    echo "创建模型目录..."
    mkdir -p models
fi

# 启动后端服务
echo "=========================================="
echo "  启动后端服务..."
echo "=========================================="
echo ""
echo "API服务地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "注意: 前端待开发，当前仅提供API服务"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

cd backend
python3 main.py
