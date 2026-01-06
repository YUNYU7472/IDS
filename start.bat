@echo off
REM 联邦学习入侵检测系统启动脚本 (Windows)

echo ==========================================
echo   联邦学习入侵检测系统
echo   Federated Learning Intrusion Detection
echo ==========================================
echo.

REM 检查Python
echo 检查Python版本...
python --version
echo.

REM 检查依赖
echo 检查依赖...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo 依赖未安装，正在安装...
    pip install -r requirements.txt
) else (
    echo 依赖已安装
)
echo.

REM 检查数据文件
echo 检查数据文件...
if exist "..\data\KDDTrain+.csv" (
    if exist "..\data\KDDTest+.csv" (
        echo 数据文件已就绪
    ) else (
        echo 数据文件未找到: KDDTest+.csv
        pause
        exit /b 1
    )
) else (
    echo 数据文件未找到: KDDTrain+.csv
    pause
    exit /b 1
)
echo.

REM 创建模型目录
if not exist "models" (
    echo 创建模型目录...
    mkdir models
)

REM 启动服务
echo ==========================================
echo   启动后端服务...
echo ==========================================
echo.
echo API服务地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo.
echo 注意: 前端待开发，当前仅提供API服务
echo.
echo 按 Ctrl+C 停止服务
echo.

cd backend
python main.py
