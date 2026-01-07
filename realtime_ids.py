#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时入侵检测系统 (Real-time Intrusion Detection System)
基于联邦学习模型的实时网络流量检测服务
"""

import sys
import os
import argparse
import json
import time
import hashlib
import logging
import signal
from datetime import datetime
from collections import deque, defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# 复用原模块的功能
import kdd_federated_learning as kfl

# ============================================================================
# 全局配置
# ============================================================================
__version__ = "1.0.0"
VERSION_HASH = hashlib.md5(__version__.encode()).hexdigest()[:8]

# 强制使用CPU设备（即使有GPU）
DEVICE = torch.device("cpu")

# 日志格式配置
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# ============================================================================
# 日志配置
# ============================================================================
def setup_logging(log_level: str = "INFO"):
    """配置日志系统"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stderr
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# 统计器
# ============================================================================
class StatisticsCollector:
    """收集和统计检测性能指标"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.alert_counts = defaultdict(int)
        self.total_count = 0
        self.start_time = time.time()
        self.last_report_time = time.time()
        
    def record(self, latency_ms: float, threat_label: int, threat_name: str):
        """记录一次检测结果"""
        self.latencies.append(latency_ms)
        self.total_count += 1
        if threat_label != 0:  # 非正常流量
            self.alert_counts[threat_name] += 1
    
    def get_stats(self) -> Dict:
        """获取当前统计信息"""
        if not self.latencies:
            return {
                'qps': 0.0,
                'p50_latency': 0.0,
                'p95_latency': 0.0,
                'p99_latency': 0.0,
                'mean_latency': 0.0,
                'total_count': self.total_count,
                'alert_count': sum(self.alert_counts.values()),
                'alert_breakdown': dict(self.alert_counts),
                'uptime_seconds': time.time() - self.start_time
            }
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            'qps': self.total_count / (time.time() - self.start_time) if self.total_count > 0 else 0.0,
            'p50_latency': sorted_latencies[int(n * 0.50)] if n > 0 else 0.0,
            'p95_latency': sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            'p99_latency': sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
            'mean_latency': np.mean(sorted_latencies),
            'total_count': self.total_count,
            'alert_count': sum(self.alert_counts.values()),
            'alert_breakdown': dict(self.alert_counts),
            'uptime_seconds': time.time() - self.start_time
        }
    
    def should_report(self, interval: float) -> bool:
        """检查是否应该输出统计报告"""
        now = time.time()
        if now - self.last_report_time >= interval:
            self.last_report_time = now
            return True
        return False


# ============================================================================
# 推理引擎
# ============================================================================
class InferenceEngine:
    """实时推理引擎"""
    
    def __init__(self, model_path: str, batch_size: int = 1):
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = None
        self.max_values = None
        self.input_dim = None
        self.output_dim = None
        self.model_hash = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和相关参数"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        logger.info(f"正在加载模型: {self.model_path}")
        
        # 计算模型文件哈希
        with open(self.model_path, 'rb') as f:
            self.model_hash = hashlib.md5(f.read()).hexdigest()[:16]
        
        # 使用原模块的load_model函数，但强制使用CPU
        checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
        
        self.model = kfl.Net2nn(checkpoint['inputs'], checkpoint['outputs'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(DEVICE)  # 强制CPU
        self.model.eval()
        
        self.max_values = checkpoint['max_values']
        self.input_dim = checkpoint['inputs']
        self.output_dim = checkpoint['outputs']
        
        logger.info(f"模型加载成功: 输入维度={self.input_dim}, 输出类别={self.output_dim}")
        logger.info(f"模型哈希: {self.model_hash}")
    
    def preprocess_single(self, raw_data: str, input_format: str = 'auto') -> Optional[torch.Tensor]:
        """
        预处理单条输入数据
        
        Args:
            raw_data: 原始CSV格式字符串（逗号分隔）
            input_format: 'csv41', 'csv42', 'csv43', 'auto'
        
        Returns:
            预处理后的tensor，如果格式错误返回None
        """
        try:
            # 解析CSV行
            parts = [p.strip() for p in raw_data.split(',')]
            n_cols = len(parts)
            
            # 自动检测格式或使用指定格式
            if input_format == 'auto':
                if n_cols == 41:
                    # 41列：无标签
                    data = parts
                elif n_cols == 42:
                    # 42列：含threat_type
                    data = parts[:-1]
                elif n_cols == 43:
                    # 43列：含threat_type和difficulty
                    data = parts[:-2]
                else:
                    logger.warning(f"无法自动识别格式: {n_cols}列，期望41/42/43列")
                    return None
            elif input_format == 'csv41':
                if n_cols != 41:
                    logger.warning(f"格式不匹配: 期望41列，实际{n_cols}列")
                    return None
                data = parts
            elif input_format == 'csv42':
                if n_cols != 42:
                    logger.warning(f"格式不匹配: 期望42列，实际{n_cols}列")
                    return None
                data = parts[:-1]  # 移除threat_type
            elif input_format == 'csv43':
                if n_cols != 43:
                    logger.warning(f"格式不匹配: 期望43列，实际{n_cols}列")
                    return None
                data = parts[:-2]  # 移除threat_type和difficulty
            else:
                logger.error(f"未知的输入格式: {input_format}")
                return None
            
            # 转换为DataFrame（单行）
            df = pd.DataFrame([data], columns=kfl.colnames[:-1])  # 不包括threat_type
            
            # 添加虚拟的threat_type列（用于兼容预处理函数）
            df['threat_type'] = 0
            
            # 使用原模块的预处理函数
            x_data = kfl.preprocess_new_data(df, self.max_values, kfl.colnames)
            
            # 转换为tensor（单样本）
            x_tensor = torch.tensor(x_data, dtype=torch.float32).to(DEVICE)
            
            return x_tensor
            
        except Exception as e:
            logger.error(f"预处理失败: {e}, 数据: {raw_data[:100]}...")
            return None
    
    def infer(self, x_tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """
        执行推理
        
        Returns:
            (预测标签, 置信度, 所有类别概率)
        """
        with torch.no_grad():
            output = self.model(x_tensor)
            probs = F.softmax(output, dim=-1)
            pred_label = output.argmax(dim=-1).item()
            confidence = probs[0][pred_label].item()
            probs_array = probs[0].cpu().numpy()
        
        return pred_label, confidence, probs_array


# ============================================================================
# 输入读取器
# ============================================================================
class InputReader:
    """输入数据读取器（支持stdin和tail-file）"""
    
    def __init__(self, source_type: str, source_path: Optional[str] = None):
        self.source_type = source_type
        self.source_path = source_path
        self.file_handle = None
        self._setup()
    
    def _setup(self):
        """设置输入源"""
        if self.source_type == 'stdin':
            self.file_handle = sys.stdin
            logger.info("输入源: 标准输入 (stdin)")
        elif self.source_type == 'tail-file':
            if not self.source_path:
                raise ValueError("tail-file模式需要指定文件路径")
            if not os.path.exists(self.source_path):
                raise FileNotFoundError(f"文件不存在: {self.source_path}")
            self.file_handle = open(self.source_path, 'r', encoding='utf-8')
            # 移动到文件末尾（只读取新追加的内容）
            self.file_handle.seek(0, 2)  # 移动到文件末尾
            logger.info(f"输入源: 文件尾监听 (tail-file: {self.source_path})")
        else:
            raise ValueError(f"未知的输入源类型: {self.source_type}")
    
    def read_line(self, timeout: float = 0.1) -> Optional[str]:
        """读取一行数据（非阻塞）"""
        try:
            if self.source_type == 'stdin':
                # stdin可能有缓冲，使用readline
                line = self.file_handle.readline()
                if not line:
                    return None
                return line.strip()
            elif self.source_type == 'tail-file':
                # 检查文件是否有新内容
                line = self.file_handle.readline()
                if not line:
                    return None
                return line.strip()
        except Exception as e:
            logger.error(f"读取输入失败: {e}")
            return None
    
    def close(self):
        """关闭输入源"""
        if self.file_handle and self.source_type == 'tail-file':
            self.file_handle.close()


# ============================================================================
# 输出器
# ============================================================================
class OutputWriter:
    """结果输出器（支持stdout和文件）"""
    
    def __init__(self, output_path: Optional[str] = None, output_format: str = 'jsonl'):
        self.output_path = output_path
        self.output_format = output_format
        self.file_handle = None
        
        if output_path:
            self.file_handle = open(output_path, 'w', encoding='utf-8')
            logger.info(f"输出目标: 文件 ({output_path})")
        else:
            self.file_handle = sys.stdout
            logger.info("输出目标: 标准输出 (stdout)")
    
    def write_result(self, result: Dict):
        """写入一条检测结果"""
        if self.output_format == 'jsonl':
            json_str = json.dumps(result, ensure_ascii=False)
            self.file_handle.write(json_str + '\n')
            self.file_handle.flush()
        elif self.output_format == 'csv':
            # CSV格式：timestamp,src,threat_type,confidence,latency_ms
            csv_line = f"{result['timestamp']},{result.get('src', 'NA')},{result['threat_type']},{result['confidence']:.4f},{result['latency_ms']:.2f}\n"
            self.file_handle.write(csv_line)
            self.file_handle.flush()
    
    def close(self):
        """关闭输出文件"""
        if self.output_path and self.file_handle:
            self.file_handle.close()


# ============================================================================
# 主程序
# ============================================================================
def health_check(model_path: str):
    """健康检查：打印模型信息后退出"""
    try:
        engine = InferenceEngine(model_path, batch_size=1)
        print(f"\n{'='*80}")
        print("实时入侵检测系统 - 健康检查")
        print(f"{'='*80}")
        print(f"版本: {__version__} (hash: {VERSION_HASH})")
        print(f"模型路径: {model_path}")
        print(f"模型哈希: {engine.model_hash}")
        print(f"输入维度: {engine.input_dim}")
        print(f"输出类别数: {engine.output_dim}")
        print(f"运行设备: {DEVICE}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"{'='*80}\n")
        print("✓ 模型加载成功，系统就绪")
        sys.exit(0)
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        sys.exit(1)


def parse_single_line(line: str, input_format: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析输入行，提取特征数据和源IP（如果存在）
    
    Returns:
        (特征数据CSV字符串, 源IP或None)
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None
    
    parts = [p.strip() for p in line.split(',')]
    
    # 尝试提取源IP（通常在前几列，但NSL-KDD格式中没有标准IP字段）
    # 这里假设如果有IP格式的数据，可能在某个位置
    src = None
    for part in parts[:5]:  # 检查前5列
        if '.' in part and part.count('.') == 3:
            try:
                # 验证是否为IP地址格式
                octets = part.split('.')
                if all(0 <= int(o) <= 255 for o in octets):
                    src = part
                    break
            except:
                pass
    
    return line, src


def main():
    parser = argparse.ArgumentParser(
        description='实时入侵检测系统 - 基于联邦学习的实时网络流量检测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例命令:
  # 从stdin读取
  cat data.csv | python realtime_ids.py --stdin --model-path ckpt/model.pth
  
  # 监听文件追加
  python realtime_ids.py --tail-file /var/log/network.log --model-path ckpt/model.pth --output results.jsonl
  
  # 指定输出文件
  python realtime_ids.py --stdin --model-path ckpt/model.pth --output results.csv --format csv
        """
    )
    
    # 输入源（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--stdin', action='store_true',
                            help='从标准输入读取（管道/SSH/agent上报）')
    input_group.add_argument('--tail-file', type=str, metavar='PATH',
                            help='持续监听文件尾部追加（tail -f模式）')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, default='ckpt/model.pth',
                       help='模型文件路径 (默认: ckpt/model.pth)')
    
    # 推理参数
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批处理大小 (默认: 1，实时场景建议为1)')
    
    # 输入格式
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'csv41', 'csv42', 'csv43'],
                       help='输入数据格式 (默认: auto自动检测)')
    
    # 输出参数
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (默认: stdout)')
    parser.add_argument('--output-format', type=str, default='jsonl',
                       choices=['jsonl', 'csv'],
                       help='输出格式 (默认: jsonl)')
    
    # 日志和统计
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    parser.add_argument('--stats-interval', type=float, default=10.0,
                       help='统计报告间隔（秒）(默认: 10.0)')
    
    # 特殊模式
    parser.add_argument('--dry-run', action='store_true',
                       help='干运行模式：只解析不推理（用于排查输入格式）')
    parser.add_argument('--health-check', action='store_true',
                       help='健康检查：打印模型信息后退出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    global logger
    logger = setup_logging(args.log_level)
    
    # 健康检查模式
    if args.health_check:
        health_check(args.model_path)
    
    # 初始化组件
    try:
        # 加载模型
        engine = InferenceEngine(args.model_path, batch_size=args.batch_size)
        
        # 设置输入源
        source_type = 'stdin' if args.stdin else 'tail-file'
        source_path = None if args.stdin else args.tail_file
        reader = InputReader(source_type, source_path)
        
        # 设置输出
        writer = OutputWriter(args.output, args.output_format)
        
        # 统计器
        stats = StatisticsCollector(window_size=1000)
        
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        sys.exit(1)
    
    # 启动信息
    logger.info("=" * 80)
    logger.info("实时入侵检测系统启动")
    logger.info("=" * 80)
    logger.info(f"版本: {__version__} (hash: {VERSION_HASH})")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"模型哈希: {engine.model_hash}")
    logger.info(f"输入维度: {engine.input_dim}")
    logger.info(f"输出类别数: {engine.output_dim}")
    logger.info(f"运行设备: {DEVICE} (强制CPU模式)")
    logger.info(f"输入模式: {source_type}")
    if source_path:
        logger.info(f"监听文件: {source_path}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"输入格式: {args.format}")
    logger.info(f"输出格式: {args.output_format}")
    logger.info(f"统计间隔: {args.stats_interval}秒")
    logger.info("=" * 80)
    logger.info("开始实时检测...")
    logger.info("")
    
    # 信号处理（优雅退出）
    def signal_handler(sig, frame):
        logger.info("\n收到退出信号，正在关闭...")
        stats_final = stats.get_stats()
        logger.info("=" * 80)
        logger.info("最终统计报告")
        logger.info("=" * 80)
        logger.info(f"总检测数: {stats_final['total_count']}")
        logger.info(f"总告警数: {stats_final['alert_count']}")
        logger.info(f"平均QPS: {stats_final['qps']:.2f}")
        logger.info(f"平均延迟: {stats_final['mean_latency']:.2f}ms")
        logger.info(f"P95延迟: {stats_final['p95_latency']:.2f}ms")
        logger.info(f"运行时长: {stats_final['uptime_seconds']:.1f}秒")
        logger.info("告警分类统计:")
        for threat_type, count in stats_final['alert_breakdown'].items():
            logger.info(f"  {threat_type}: {count}")
        logger.info("=" * 80)
        reader.close()
        writer.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 主循环
    try:
        while True:
            # 读取输入
            line = reader.read_line()
            if line is None:
                time.sleep(0.01)  # 避免CPU空转
                # 检查是否需要输出统计
                if stats.should_report(args.stats_interval):
                    stats_data = stats.get_stats()
                    if stats_data['total_count'] > 0:
                        logger.info("=" * 80)
                        logger.info(f"统计报告 [总检测: {stats_data['total_count']}, 告警: {stats_data['alert_count']}]")
                        logger.info(f"QPS: {stats_data['qps']:.2f} | "
                                  f"延迟: 均值={stats_data['mean_latency']:.2f}ms, "
                                  f"P50={stats_data['p50_latency']:.2f}ms, "
                                  f"P95={stats_data['p95_latency']:.2f}ms, "
                                  f"P99={stats_data['p99_latency']:.2f}ms")
                        if stats_data['alert_breakdown']:
                            logger.info("告警分类:")
                            for threat_type, count in stats_data['alert_breakdown'].items():
                                logger.info(f"  {threat_type}: {count}")
                        logger.info("=" * 80)
                continue
            
            # 解析输入
            raw_data, src = parse_single_line(line, args.format)
            if raw_data is None:
                continue
            
            # 干运行模式
            if args.dry_run:
                logger.info(f"[DRY-RUN] 解析成功: {len(raw_data.split(','))}列, 源: {src or 'NA'}")
                continue
            
            # 预处理
            start_time = time.time()
            x_tensor = engine.preprocess_single(raw_data, args.format)
            if x_tensor is None:
                continue
            
            # 推理
            try:
                pred_label, confidence, probs = engine.infer(x_tensor)
                latency_ms = (time.time() - start_time) * 1000
                
                # 获取威胁类型名称
                threat_name = kfl.THREAT_LABELS.get(pred_label, f'未知({pred_label})')
                
                # 记录统计
                stats.record(latency_ms, pred_label, threat_name)
                
                # 构建结果
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'src': src or 'NA',
                    'threat_type': threat_name,
                    'threat_label': int(pred_label),
                    'confidence': float(confidence),
                    'latency_ms': float(latency_ms),
                    'model_version': engine.model_hash
                }
                
                # 输出结果
                writer.write_result(result)
                
                # 如果是告警，在stderr输出（便于区分）
                if pred_label != 0:
                    logger.warning(f"🚨 告警: {threat_name} (置信度: {confidence:.2%}, 延迟: {latency_ms:.2f}ms, 源: {src or 'NA'})")
                
            except Exception as e:
                logger.error(f"推理失败: {e}, 数据: {raw_data[:100]}...")
                continue
            
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        logger.error(f"运行时错误: {e}", exc_info=True)
        reader.close()
        writer.close()
        sys.exit(1)


if __name__ == '__main__':
    main()

