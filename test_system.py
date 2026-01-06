"""
快速测试脚本 - 验证系统各模块是否正常工作
"""
import sys
import os

# 添加backend目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """测试所有模块是否能正常导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    try:
        from config import Config
        print("✓ config.py - 配置模块")
        
        from model import Net2nn, create_model
        print("✓ model.py - 模型模块")
        
        from data_loader import KDDDataLoader
        print("✓ data_loader.py - 数据加载模块")
        
        from federated_core import FederatedTrainer
        print("✓ federated_core.py - 联邦学习核心")
        
        print("\n所有模块导入成功！")
        return True
    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """测试模型创建和前向传播"""
    print("\n" + "=" * 60)
    print("测试模型...")
    print("=" * 60)
    
    try:
        import torch
        from model import create_model
        from config import Config
        
        # 创建模型
        model = create_model()
        print(f"✓ 模型创建成功")
        print(f"  输入维度: {Config.INPUT_DIM}")
        print(f"  隐藏层: {Config.HIDDEN_DIM}")
        print(f"  输出维度: {Config.OUTPUT_DIM}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, Config.INPUT_DIM)
        output = model(dummy_input)
        print(f"✓ 前向传播成功")
        print(f"  输出形状: {output.shape}")
        
        # 测试预测
        pred_class, probs = model.predict(dummy_input)
        print(f"✓ 预测功能正常")
        print(f"  预测类别: {pred_class.item()}")
        print(f"  置信度: {probs[0, pred_class.item()].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """测试配置"""
    print("\n" + "=" * 60)
    print("测试配置...")
    print("=" * 60)
    
    try:
        from config import Config
        
        print(f"✓ 攻击类型映射:")
        for label, name in Config.ATTACK_MAPPING.items():
            print(f"  {label}: {name}")
        
        print(f"\n✓ 训练参数:")
        print(f"  学习率: {Config.LEARNING_RATE}")
        print(f"  批次大小: {Config.BATCH_SIZE}")
        print(f"  训练轮数: {Config.NUM_EPOCHS}")
        print(f"  客户端数: {Config.NUM_CLIENTS}")
        
        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 60)
    print("检查依赖包...")
    print("=" * 60)
    
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'sklearn',
        'fastapi',
        'uvicorn',
        'imblearn'
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            all_ok = False
    
    return all_ok

def main():
    """主测试函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "联邦学习入侵检测系统 - 快速测试" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 运行所有测试
    results = {
        "依赖检查": check_dependencies(),
        "模块导入": test_imports(),
        "配置测试": test_config(),
        "模型测试": test_model()
    }
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s} : {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！系统已就绪。")
        print("\n运行以下命令启动系统:")
        print("  ./start.sh        (Linux/Mac)")
        print("  start.bat         (Windows)")
        print("  cd backend && python main.py")
    else:
        print("✗ 部分测试失败，请检查错误信息并安装缺失的依赖。")
        print("\n安装依赖:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
