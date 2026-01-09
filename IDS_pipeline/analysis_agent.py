#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 分析代理 (Analysis Agent):
定期读取蜜罐日志，聚合攻击上下文，调用 LLM (DeepSeek) 生成可解释性安全报告。
[修复版] 兼容 Python 3.8/3.9 的类型注解
"""

import argparse
import json
import time
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union # [修改] 引入兼容类型

try:
    import requests
except ImportError:
    print("错误: 缺少 requests 库。请运行: pip install requests")
    sys.exit(1)

# ---------- 配置与常量 ----------
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

# ---------- 日志配置 ----------
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger("AnalysisAgent")

# ---------- 核心功能 ----------

# [修改] 使用 Optional[str] 替代 str | None
def get_api_key(args_key: Optional[str]) -> str:
    if args_key:
        return args_key
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    
    logger.warning("未检测到 API Key。请设置环境变量 DEEPSEEK_API_KEY 或使用 --api-key 参数。")
    return ""

def read_new_logs(filepath: str, last_position: int) -> tuple: # Python 3.8元组类型提示较复杂，直接简化或移除
    if not os.path.exists(filepath):
        return [], 0
    
    new_records = []
    current_size = os.path.getsize(filepath)
    
    if current_size < last_position:
        last_position = 0
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.seek(last_position)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    new_records.append(record)
                except json.JSONDecodeError:
                    continue
            new_position = f.tell()
    except Exception as e:
        logger.error(f"读取日志文件失败: {e}")
        return [], last_position

    return new_records, new_position

def call_deepseek_api(logs: List[Dict], api_key: str, api_url: str, model: str) -> str:
    if not api_key:
        return "无法生成报告：缺失 API Key。"
        
    minified_logs = []
    for log in logs:
        simplified = {
            "time": log.get("ts_ms"),
            "src": f"{log.get('client_ip')}:{log.get('client_port')}",
            "type": log.get("record_type", "unknown"),
        }
        
        if "preview" in log:
             simplified["payload"] = log["preview"]
        elif "honeypot_observed" in log:
             obs = log["honeypot_observed"]
             simplified["payload"] = obs.get("preview", "") or obs.get("preview_b64", "")
        
        if "ids_result" in log:
            simplified["ids_decision"] = log.get("decision")
            simplified["threat"] = log.get("ids_result", {}).get("pred_name") or log.get("ids_result", {}).get("threat_type_str")

        minified_logs.append(simplified)

    log_snippet = json.dumps(minified_logs, indent=2, ensure_ascii=False)
    
    prompt = f"""
你是一名资深网络安全分析师。这是蜜罐系统在过去一段时间内捕获的 {len(logs)} 条攻击/探测日志片段：

{log_snippet}

请根据上述日志进行简要的威胁分析（中文回答）。报告结构如下：
1. **态势概览**: 攻击源IP分布、主要攻击时间段。
2. **攻击分析**: 
   - 尝试分析攻击者的意图（如: 扫描、WebShell探测、SQL注入、SSH暴力破解等）。
   - 如果有 payload，请解释其含义。
3. **防御建议**: 针对这些特定行为的阻断或加固建议。

请保持客观、专业。
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个网络安全专家助手。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3, 
        "stream": False
    }

    try:
        logger.info(f"正在向 DeepSeek 发送 {len(logs)} 条记录进行分析...")
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        content = result['choices'][0]['message']['content']
        return content
    except Exception as e:
        logger.error(f"API 调用失败: {e}")
        return f"AI 分析暂时不可用: {e}"

def main():
    parser = argparse.ArgumentParser(description="IDS AI Analysis Agent")
    # [修改] 默认文件名改为 honeypot_hits.jsonl 以匹配 honeypot.py
    parser.add_argument("--log", default="honeypot_hits.jsonl", help="蜜罐日志文件路径")
    parser.add_argument("--interval", type=int, default=10, help="分析周期（秒）") # [建议] 缩短默认周期方便测试
    parser.add_argument("--batch-size", type=int, default=20, help="每次分析的最大日志条数")
    parser.add_argument("--api-key", help="DeepSeek API Key")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="LLM API Endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    
    args = parser.parse_args()
    
    logger.info(f"AI 分析代理已启动。监控日志: {args.log}")
    
    # [修改] 启动时不跳过历史日志，方便测试刚才的 curl 攻击
    last_pos = 0
    if os.path.exists(args.log):
        # 如果你想跳过旧日志，取消下面两行的注释
        # last_pos = os.path.getsize(args.log)
        # logger.info(f"跳过历史日志，从字节偏移量 {last_pos} 开始监控...")
        pass
    
    api_key = get_api_key(args.api_key)
    if not api_key:
        logger.warning(">>> 警告: 未配置 API Key，将仅打印日志聚合结果，跳过 AI 分析 <<<")

    try:
        while True:
            # 1.获取新日志
            new_logs, next_pos = read_new_logs(args.log, last_pos)
            
            if new_logs:
                last_pos = next_pos
                logger.info(f"捕获到 {len(new_logs)} 条新记录。")
                
                # 2. 批次处理
                analyze_batch = new_logs[-args.batch_size:] 
                
                # 3. 调用 AI
                if api_key:
                    report = call_deepseek_api(analyze_batch, api_key, args.api_url, args.model)
                    print("\n" + "="*40 + " INTELLIGENCE REPORT " + "="*40)
                    print(report)
                    print("="*101 + "\n")
                else:
                    logger.info("跳过 AI 调用 (无 Key)")
            
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("停止监控。")

if __name__ == "__main__":
    main()