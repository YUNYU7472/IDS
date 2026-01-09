#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版实时推理脚本（仅保留：加载模型、从 stdin 读取 CSV/JSONL、推理、写入 CSV、输出一次统计）

使用说明（示例）：
  head -n 50 data/KDDTest+.jsonl | python realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv
  head -n 50 data/KDDTest+.csv  | python realtime_ids.py --input-type csv  --model-path ckpt/model.pth --output out.csv

设计原则：
- 只支持 stdin 输入（JSONL 或 CSV）
- 输出为 CSV 文件（默认 detection_results.csv）
- 特征严格以 checkpoint['max_values'].index 顺序为准
- 遇到缺失或无法转 float 的字段，跳过该行并计入 bad_rows
- 只在 EOF 时输出一次最终统计
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import time
import csv
from io import StringIO
from typing import Iterator, List, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

import logging
from response_actions import maybe_redirect_to_honeypot

import torch.nn as nn

# KDD 列名（用于 CSV 解析，与训练时列顺序一致）
colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'threat_type']

# Inlined simple Net2nn model (3-layer MLP)
class Net2nn(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 常量
DEFAULT_OUTPUT = 'detection_results.csv'
DEVICE = torch.device('cpu')


# ---------- logging setup ----------
class MillisecondFormatter(logging.Formatter):
    """Formatter that truncates microseconds to milliseconds for timestamps"""
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        ms = int(record.msecs)
        return f"{t}.{ms:03d}"


def setup_logging(level_name: str = 'INFO', quiet: bool = False) -> logging.Logger:
    level = logging.getLevelName(level_name)
    if quiet:
        level = logging.WARNING
    logger = logging.getLogger('realtime_ids')
    logger.setLevel(level)
    # clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(level)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ch.setFormatter(MillisecondFormatter(fmt))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


# global logger (will be initialized in main)
logger: logging.Logger | None = None


def load_checkpoint_model(model_path: str):
    """加载 checkpoint，返回 (model, feature_names, inputs, outputs, max_values, model_hash)

    要求 checkpoint 包含: model_state_dict, max_values, inputs, outputs
    特征顺序由 max_values.index 决定
    """
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        # 兼容旧版 torch（没有 weights_only 参数）
        checkpoint = torch.load(model_path, map_location='cpu')

    # 基本字段校验
    for key in ('model_state_dict', 'max_values', 'inputs', 'outputs'):
        if key not in checkpoint:
            raise KeyError(f"checkpoint 缺少字段: {key}")

    # feature_names 提取逻辑（兼容 Series 或 dict）
    max_values = checkpoint['max_values']
    if hasattr(max_values, 'index'):
        feature_names = list(max_values.index)
    elif isinstance(max_values, dict):
        feature_names = list(max_values.keys())
    else:
        # 尝试直接迭代
        feature_names = list(max_values)

    inputs = int(checkpoint['inputs'])
    # outputs = int(checkpoint['outputs'])
    outputs = int(checkpoint['outputs'])

    if len(feature_names) != inputs:
        raise ValueError(f"特征数量不匹配: len(max_values.index)={len(feature_names)} vs inputs={inputs}")

    model = Net2nn(inputs, outputs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # compute sha256 prefix
    try:
        import hashlib
        with open(model_path, 'rb') as f:
            sha = hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        sha = 'unknown'

    # max_values 提取：
    # 下游 predict_one 使用时会转 numpy，我们需要支持 dict 查找或 Series 查找
    # 必须确保 max_values 能被转换为 numpy array 且顺序正确。
    
    if isinstance(max_values, dict):
         # dict, convert based on feature_names order
         max_v_arr = np.array([float(max_values[f]) for f in feature_names])
    elif hasattr(max_values, 'values'):
         # likely pandas Series
         max_v_arr = max_values.values.astype(float)
    else:
         # fallback (e.g. list or unknown iterable), assume order matches
         max_v_arr = np.array(list(max_values)).astype(float)
         
    return model, feature_names, inputs, outputs, max_v_arr, sha


def parse_csv_stream(stdin, feature_names: List[str]) -> Iterator[Tuple[np.ndarray, dict, str]]:
    """从 stdin 读取整个 CSV，逐行 yield (arr_or_none, meta, error)

    返回三元组：
      - arr (np.ndarray) 或 None
      - meta dict（如 {'src_ip':..., 'proto':...}）
      - error: None 或 reason 字符串 (missing_field / parse_error / dim_mismatch)
    """
    data = stdin.read()
    if not data:
        return
        yield  # type: ignore

    df = pd.read_csv(StringIO(data), header=None, dtype=str)

    # 处理 43 列（可能多出 difficulty）
    if df.shape[1] == 43:
        df = df.iloc[:, :-1]

    # 保证列数为 41 或 42（含 threat_type）
    if df.shape[1] >= 42:
        # 取前 42 列
        df = df.iloc[:, :42]
        df.columns = colnames
    elif df.shape[1] == 41:
        df.columns = colnames[:-1]
        df['threat_type'] = 'normal'
    else:
        # 不符合预期的列数：逐行尝试解析（按逗号分割）
        for line in data.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) in (41, 42, 43):
                row_df = pd.read_csv(StringIO(','.join(parts if len(parts) != 43 else parts[:-1]) + '\n'), header=None, dtype=str)
                if row_df.shape[1] == 41:
                    row_df.columns = colnames[:-1]
                    row_df['threat_type'] = 'normal'
                elif row_df.shape[1] == 42:
                    row_df.columns = colnames
                else:
                    continue
                arr, reason = _extract_numeric_row_with_reason(row_df.iloc[0], feature_names)
                meta = _extract_meta_from_row(row_df.iloc[0])
                yield arr, meta, reason
        return

    # 现在 df 的列名已对齐，按原始行迭代并从中提取 feature 与 meta
    for _, orig_row in df.iterrows():
        feat_series = orig_row.reindex(feature_names)
        arr, reason = _extract_numeric_row_with_reason(feat_series, feature_names)
        meta = _extract_meta_from_row(orig_row)
        yield arr, meta, reason


def parse_jsonl_stream(stdin, feature_names: List[str]) -> Iterator[Tuple[np.ndarray, dict, str]]:
    """逐行读取 JSONL，每行解析为 (arr_or_none, meta, error)"""
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                yield None, {}, 'parse_error'
                continue

            meta = _extract_meta_from_json(obj)

            # 情形A: {'features': {...}} 或 {'features': [...]}
            if 'features' in obj:
                feats = obj['features']
                if isinstance(feats, dict):
                    vals = []
                    missing = False
                    for name in feature_names:
                        if name not in feats:
                            missing = True
                            break
                        vals.append(feats[name])
                    if missing:
                        yield None, meta, 'missing_field'
                        continue
                elif isinstance(feats, list):
                    if len(feats) != len(feature_names):
                        yield None, meta, 'dim_mismatch'
                        continue
                    vals = feats
                else:
                    yield None, meta, 'parse_error'
                    continue
            else:
                # 情形B: 顶层字段 dict
                vals = []
                missing = False
                for name in feature_names:
                    if name not in obj:
                        missing = True
                        break
                    vals.append(obj[name])
                if missing:
                    yield None, meta, 'missing_field'
                    continue

            # 尝试转 float
            try:
                # 首先处理可能的 None 或非数值
                safe_float_vals = []
                for x in vals:
                    if x is None:
                        safe_float_vals.append(0.0)
                    else:
                        safe_float_vals.append(float(x))
                arr = np.array(safe_float_vals, dtype=float)
                # 将可能的 NaN/Inf 替换为 0
                arr[~np.isfinite(arr)] = 0.0
            except Exception:
                yield None, meta, 'parse_error'
                continue

            yield arr, meta, None

        except json.JSONDecodeError:
            yield None, {}, 'parse_error'
            continue


def _extract_numeric_row_with_reason(row: pd.Series, feature_names: List[str]) -> Tuple[np.ndarray | None, str | None]:
    """从 pandas Series 中按 feature_names 提取并转为 np.ndarray；返回 (arr, reason)

    reason 为 None 表示成功，否则为 'missing_field' 或 'parse_error'
    """
    vals = []
    for name in feature_names:
        v = row.get(name)
        if pd.isna(v):
            return None, 'missing_field'
        try:
            f = float(v)
        except Exception:
            return None, 'parse_error'
        vals.append(f)
    return np.array(vals, dtype=float), None


def _extract_meta_from_row(row: pd.Series) -> dict:
    """从 CSV 行中提取可能的元信息（src_ip, dst_ip, proto等）"""
    meta = {}
    # 尝试在前 5 列找 IP 地址
    for val in row.iloc[:5].astype(str):
        if isinstance(val, str) and val.count('.') == 3:
            parts = val.split('.')
            if len(parts) == 4:
                try:
                    if all(0 <= int(p) <= 255 for p in parts):
                        meta['src_ip'] = val
                        break
                except Exception:
                    pass
    # protocol_type 列可能存在
    if 'protocol_type' in row.index:
        proto = row.get('protocol_type')
        if pd.notna(proto):
            meta['proto'] = str(proto)
    return meta


def _extract_meta_from_json(obj: dict) -> dict:
    """从 JSONL 顶层对象提取可能的元信息，保留常见键名"""
    keys = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'proto', 'protocol', 'src', 'dst']
    meta = {}
    for k in keys:
        if k in obj:
            meta[k] = obj[k]
    # normalize some aliases
    if 'protocol' in meta and 'proto' not in meta:
        meta['proto'] = meta['protocol']
    if 'src' in meta and 'src_ip' not in meta:
        meta['src_ip'] = meta['src']
    if 'dst' in meta and 'dst_ip' not in meta:
        meta['dst_ip'] = meta['dst']
    # remove original aliases
    meta.pop('protocol', None)
    meta.pop('src', None)
    meta.pop('dst', None)
    return meta


def predict_one(model: torch.nn.Module, x: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """对单个样本推理，返回 (label, confidence, probs_array)"""
    model.eval()
    x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(x_tensor)
        probs = F.softmax(out, dim=-1)
        if probs.ndim == 2:
            probs = probs[0]
        pred = int(torch.argmax(out, dim=-1).item())
        conf = float(probs[pred].item())
        probs_array = probs.cpu().numpy()
    return pred, conf, probs_array


def main():
    global logger
    parser = argparse.ArgumentParser(description='简化实时推理脚本')
    parser.add_argument('--input-type', required=False, choices=['csv', 'jsonl'], default='jsonl',
                        help='输入类型 (csv 或 jsonl)，默认 jsonl')
    parser.add_argument('--model-path', required=True, type=str,
                        help='模型 checkpoint 路径（必填）')
    parser.add_argument('--output', required=False, type=str, default=DEFAULT_OUTPUT,
                        help='输出 CSV 文件路径（默认: detection_results.csv）')
    parser.add_argument('--quiet', action='store_true', help='安静模式：只输出告警和最终统计（等价于 --log-level WARNING）')
    parser.add_argument('--log-level', required=False, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='日志级别，默认 INFO')

    args = parser.parse_args()

    logger = setup_logging(args.log_level, quiet=args.quiet)

    model, feature_names, inputs, outputs, max_values, model_hash = load_checkpoint_model(args.model_path)

    # 启动摘要（一次性 INFO，中文化）
    CH_LABELS = {0: '正常', 1: '拒绝服务(DoS)', 2: '越权提权(U2R)', 3: '远程登录(R2L)', 4: '探测(Probe)'}
    logger.info(
        f"启动成功 版本=lite 模型路径={args.model_path} 模型哈希={model_hash} 设备=CPU "
        f"输入类型={args.input_type} 输出文件={args.output} 期望特征维度={len(feature_names)} "
        f"模型输入维度={inputs} 标签映射={CH_LABELS} 读完EOF自动退出=是"
    )

    # 验证 inputs
    if inputs != len(feature_names):
        logger.error("模型 inputs 与 feature_names 长度不一致")
        raise RuntimeError("模型 inputs 与 feature_names 长度不一致")

    # 处理 max_values，替换 0 为 1 防止除零
    maxv = pd.Series(max_values).reindex(feature_names).astype(float).replace(0.0, 1.0).values

    total_rows = 0
    ok_rows = 0
    bad_rows = 0
    alerts = 0
    per_label_counts: dict = {}
    latencies: list = []
    bad_reasons: dict = {}

    # CSV header: include optional meta columns consistently
    base_fields = ['idx', 'pred_label', 'pred_name', 'confidence', 'latency_ms']
    meta_fields = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'proto']
    fieldnames = base_fields + meta_fields

    out_f = open(args.output, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()

    start_time = time.time()

    # Choose parser
    if args.input_type == 'csv':
        generator = parse_csv_stream(sys.stdin, feature_names)
    else:
        generator = parse_jsonl_stream(sys.stdin, feature_names)

    out_idx = 0
    for entry in generator:
        total_rows += 1
        arr, meta, reason = entry
        if reason is not None:
            bad_rows += 1
            bad_reasons[reason] = bad_reasons.get(reason, 0) + 1
            continue

        if arr is None:
            bad_rows += 1
            bad_reasons['unknown'] = bad_reasons.get('unknown', 0) + 1
            continue

        if arr.shape[0] != len(feature_names):
            bad_rows += 1
            bad_reasons['dim_mismatch'] = bad_reasons.get('dim_mismatch', 0) + 1
            continue

        # 归一化（严格按照 max_values）
        try:
            x = arr.astype(float) / maxv
        except Exception:
            bad_rows += 1
            bad_reasons['parse_error'] = bad_reasons.get('parse_error', 0) + 1
            continue
        
        # 归一化后可能再次产生 NaN/Inf (例如除以了0), 再次进行清洗
        x[~np.isfinite(x)] = 0.0

        # 如果有 NaN/Inf，不再跳过，而是视为0 (因为已经清洗了，这里理论上不会触发，但保留作为最终守门员)
        if not np.isfinite(x).all():
             # 双重保险，如果还无法清洗，才丢弃
            bad_rows += 1
            bad_reasons['nan_inf'] = bad_reasons.get('nan_inf', 0) + 1
            continue

        t0 = time.time()
        pred, conf, probs = predict_one(model, x)
        latency_ms = (time.time() - t0) * 1000.0

        ok_rows += 1
        out_idx += 1
        latencies.append(latency_ms)

        if pred != 0:
            alerts += 1
        per_label_counts[pred] = per_label_counts.get(pred, 0) + 1

        # 使用中文标签映射（已在启动时声明）
        pred_name = CH_LABELS.get(pred, '未知')

        # 写 CSV
        rowd = {k: '' for k in fieldnames}
        rowd['idx'] = out_idx
        rowd['pred_label'] = pred
        rowd['pred_name'] = pred_name
        rowd['confidence'] = f"{conf:.6f}"
        rowd['latency_ms'] = f"{latency_ms:.2f}"
        for m in meta_fields:
            if m in meta:
                rowd[m] = meta[m]
        writer.writerow(rowd)

        # 告警日志（只在 pred != 0），中文化输出
        if pred != 0:
            parts = [f"告警 序号={out_idx}", f"类型={pred_name}", f"置信度={conf*100:.2f}%", f"延迟={latency_ms:.2f}ms"]
            # add meta key-values if present (only show non-empty); 优先显示 proto
            if 'proto' in meta:
                parts.append(f"proto={meta['proto']}")
            for k in ['src_ip', 'dst_ip', 'src_port', 'dst_port']:
                if k in meta:
                    parts.append(f"{k}={meta[k]}")
            # include topk when DEBUG
            if logger.level == logging.DEBUG:
                # top3
                topk = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)[:3]
                tk = ','.join([f"{i}:{p*100:.2f}%" for i, p in topk])
                parts.append(f"topk={tk}")

            # maybe redirect to honeypot (dry-run); do not print to stdout
            try:
                # include human-readable pred_name in meta for downstream beacon payload
                meta["pred_name"] = pred_name
                ra = maybe_redirect_to_honeypot(meta=meta, pred=pred, conf=conf)
                if logger.level == logging.DEBUG:
                    logger.debug(f"response_action={ra}")
            except Exception:
                logger.exception("response_actions failed")

            logger.warning(' '.join(parts))

        # DEBUG 每条样本日志（中文化）
        if logger.level == logging.DEBUG:
            logger.debug(f"样本 序号={out_idx} 预测={pred} 置信度={conf:.6f} 延迟={latency_ms:.2f}ms")

    # 结束，写统计并退出
    elapsed = time.time() - start_time

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    throughput = ok_rows / elapsed if elapsed > 0 else 0.0

    # 只在结束时输出一次最终统计（中文化）
    # 把 per_label_counts 的 key 转为中文名
    label_cn_map = {0: '正常', 1: '拒绝服务(DoS)', 2: '越权提权(U2R)', 3: '远程登录(R2L)', 4: '探测(Probe)'}
    per_label_cn = {label_cn_map.get(k, f'未知({k})'): v for k, v in per_label_counts.items()}

    logger.info(
        f"最终统计 总输入={total_rows} 有效检测={ok_rows} 无效行={bad_rows} 告警数={alerts} "
        f"类别统计={per_label_cn} 平均延迟={avg_latency:.3f}ms 吞吐={throughput:.3f}条/秒 "
        f"输出文件={args.output} 耗时={elapsed:.3f}s"
    )

    if bad_reasons and bad_rows > 0:
        reason_map = {
            'missing_field': '缺失字段',
            'parse_error': '解析失败',
            'dim_mismatch': '维度不匹配',
            'nan_inf': '出现NaN/Inf',
            'unknown': '未知原因'
        }
        bad_reasons_cn = {reason_map.get(k, k): v for k, v in bad_reasons.items()}
        logger.info(f"无效行原因统计: {bad_reasons_cn}")

    out_f.close()


if __name__ == '__main__':
    main()
