"""
神经网络模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class Net2nn(nn.Module):
    """
    两层隐藏层的全连接神经网络
    用于NSL-KDD入侵检测分类任务
    """
    def __init__(self, inputs=None, outputs=None):
        super(Net2nn, self).__init__()
        
        # 使用配置文件中的参数，或使用传入的参数
        self.inputs = inputs or Config.INPUT_DIM
        self.outputs = outputs or Config.OUTPUT_DIM
        self.hidden = Config.HIDDEN_DIM
        
        # 定义网络层
        self.fc1 = nn.Linear(self.inputs, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.outputs)
        
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        """
        预测方法 - 返回类别和概率
        
        Args:
            x: 输入特征张量
            
        Returns:
            predicted_class: 预测的类别
            probabilities: 各类别的概率
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1)
        return predicted_class, probabilities
    
    def get_weights(self):
        """获取模型权重（用于联邦学习）"""
        return {
            'fc1_weight': self.fc1.weight.data.clone(),
            'fc1_bias': self.fc1.bias.data.clone(),
            'fc2_weight': self.fc2.weight.data.clone(),
            'fc2_bias': self.fc2.bias.data.clone(),
            'fc3_weight': self.fc3.weight.data.clone(),
            'fc3_bias': self.fc3.bias.data.clone(),
        }
    
    def set_weights(self, weights):
        """设置模型权重（用于联邦学习）"""
        with torch.no_grad():
            self.fc1.weight.data = weights['fc1_weight'].data.clone()
            self.fc1.bias.data = weights['fc1_bias'].data.clone()
            self.fc2.weight.data = weights['fc2_weight'].data.clone()
            self.fc2.bias.data = weights['fc2_bias'].data.clone()
            self.fc3.weight.data = weights['fc3_weight'].data.clone()
            self.fc3.bias.data = weights['fc3_bias'].data.clone()


def create_model(input_dim=None, output_dim=None):
    """
    创建模型实例的工厂函数
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        
    Returns:
        model: Net2nn实例
    """
    model = Net2nn(inputs=input_dim, outputs=output_dim)
    return model


def save_model(model, path):
    """
    保存模型到文件
    
    Args:
        model: 模型实例
        path: 保存路径
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': model.inputs,
        'output_dim': model.outputs
    }, path)
    print(f"Model saved to {path}")


def load_model(path):
    """
    从文件加载模型
    
    Args:
        path: 模型文件路径
        
    Returns:
        model: 加载的模型实例
    """
    checkpoint = torch.load(path)
    model = Net2nn(
        inputs=checkpoint['input_dim'],
        outputs=checkpoint['output_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    # 测试代码
    print("测试模型创建...")
    model = create_model()
    print(f"模型结构:\n{model}")
    
    # 测试前向传播
    dummy_input = torch.randn(1, Config.INPUT_DIM)
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试预测
    pred_class, probs = model.predict(dummy_input)
    print(f"\n预测类别: {pred_class.item()}")
    print(f"各类别概率: {probs.numpy()}")
