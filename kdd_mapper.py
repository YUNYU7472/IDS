#!/usr/bin/env python3
"""
kdd_mapper.py — 中间映射层：将 lite_agent 的 JSONL 输出聚合为伪 KDD 样本（JSONL -> JSONL）。

Usage examples:

Standalone:
  python lite_agent.py --interval 0.2 --max-rows 50 | \
    python kdd_mapper.py --model-path ckpt/model.pth --window-sec 2.0 --max-rows 5 > kdd_stream.jsonl

Full pipeline:
  python lite_agent.py --interval 0.2 | \
    python kdd_mapper.py --model-path ckpt/model.pth --window-sec 2.0 | \
    python realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv --log-level INFO

Behavior summary:
- 读取 stdin 每行 JSON（由 lite_agent.py 输出），维护固定滑窗（--window-sec），基于窗口首尾样本做差/聚合，输出一条伪 KDD 样本（JSONL）到 stdout。
- stdout 仅写 JSONL，所有日志写 stderr（使用 logging）。
- 读取模型 checkpoint (torch.load) 获取 `feature_names = list(checkpoint['max_values'].index)` 并确保输出的 features 包含所有这些名字。
- 对于无法直接映射的 KDD 特征，使用明确的 proxy 策略（置0或基于 net/host 指标的近似），并在启动日志里列出哪些字段为 proxy。
- EOF 时打印一次中文汇总统计并退出。

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from collections import deque, Counter
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
except Exception as e:
    # We will raise later if model loading is required; keep import error informative.
    torch = None


# ------------------------- Helpers -------------------------

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            if math.isnan(x):
                return default
            return float(x)
        return float(str(x))
    except Exception:
        return default




# ------------------------- Window Aggregator -------------------------

class WindowAggregator:
    """Maintains a tumbling window (based on first->last ts difference) and emits aggregated records.

    Implementation is intentionally simple (tumbling windows): accumulate until
    (last_ts - first_ts) >= window_sec (or EOF then optionally emit if samples >= min_samples).
    """

    def __init__(self, window_sec: float, min_samples: int, logger: logging.Logger):
        self.window_sec = float(window_sec)
        self.min_samples = int(min_samples)
        self.logger = logger

        self.buf: deque = deque()  # each entry: raw parsed sample dict

        # statistics
        self.total_input_samples = 0
        self.emitted_samples = 0
        self.dropped_windows = 0
        self.drop_reasons: Counter = Counter()
        self.window_compute_times: List[float] = []

    def add_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a sample and possibly emit an aggregated KDD-like dict if window time reached.

        Returns aggregated dict (not JSON) or None.
        """
        ts_ms = sample.get("ts_ms")
        if ts_ms is None:
            # ignore malformed sample
            self.logger.debug("skipping sample without ts_ms")
            return None

        self.buf.append(sample)
        self.total_input_samples += 1

        # check if window ready
        first_ts = self.buf[0]["ts_ms"]
        last_ts = self.buf[-1]["ts_ms"]
        if (last_ts - first_ts) / 1000.0 >= self.window_sec:
            return self._emit_window()
        return None

    def finalize(self) -> Optional[Dict[str, Any]]:
        """Call on EOF: may emit last partial window if it meets min_samples."""
        if len(self.buf) >= self.min_samples:
            return self._emit_window(final=True)
        elif len(self.buf) > 0:
            self.logger.warning("final window dropped: too few samples (%d < %d)", len(self.buf), self.min_samples)
            self.dropped_windows += 1
            self.drop_reasons["too_few_samples"] += 1
        return None

    def _emit_window(self, final: bool = False) -> Optional[Dict[str, Any]]:
        start = time.perf_counter()
        samples = list(self.buf)
        self.buf.clear()

        if len(samples) < self.min_samples:
            self.logger.warning("window dropped: too few samples (%d < %d)", len(samples), self.min_samples)
            self.dropped_windows += 1
            self.drop_reasons["too_few_samples"] += 1
            return None

        first, last = samples[0], samples[-1]
        try:
            agg = self._aggregate(samples, first, last)
        except ValueError as e:
            # e.g., counter wrap detected
            self.logger.warning("window dropped: %s", str(e))
            self.dropped_windows += 1
            self.drop_reasons[str(e)] += 1
            return None
        elapsed = time.perf_counter() - start
        self.window_compute_times.append(elapsed)
        self.emitted_samples += 1
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("emitted window: samples=%d duration=%.3fs compute=%.4fs", len(samples), (last['ts_ms']-first['ts_ms'])/1000.0, elapsed)
        return agg

    def _aggregate(self, samples: List[Dict[str, Any]], first: Dict[str, Any], last: Dict[str, Any]) -> Dict[str, Any]:
        # basic per-sample metric extractors
        def g_num(sample: Dict[str, Any], *path_candidates) -> float:
            for p in path_candidates:
                if isinstance(p, tuple):
                    d = sample
                    ok = True
                    for k in p:
                        if not isinstance(d, dict) or k not in d:
                            ok = False
                            break
                        d = d[k]
                    if ok:
                        return safe_float(d)
                else:
                    # string: try features then meta.net then meta.host
                    if 'features' in sample and isinstance(sample['features'], dict) and p in sample['features']:
                        return safe_float(sample['features'].get(p))
                    if 'meta' in sample and isinstance(sample['meta'], dict):
                        # try meta.net
                        net = sample['meta'].get('net')
                        if isinstance(net, dict) and p in net:
                            return safe_float(net.get(p))
                        # try host
                        host = sample['meta'].get('host')
                        if isinstance(host, dict) and p in host:
                            return safe_float(host.get(p))
            return 0.0

        # time
        duration = (last['ts_ms'] - first['ts_ms']) / 1000.0

        # rx/tx bytes: prefer meta.net.{rx_bytes, tx_bytes} else features
        rx_first = g_num(first, 'rx_bytes')
        rx_last = g_num(last, 'rx_bytes')
        tx_first = g_num(first, 'tx_bytes')
        tx_last = g_num(last, 'tx_bytes')

        rx_delta = rx_last - rx_first
        tx_delta = tx_last - tx_first
        if rx_delta < 0 or tx_delta < 0:
            raise ValueError('counter_wrap')

        # CPU percent: try features.cpu_percent then meta.host.cpu_percent
        cpu_vals = []
        mem_vals = []
        tcp_est_vals = []
        tcp_listen_vals = []
        tcp_time_wait_vals = []
        tcp_syn_recv_vals = []
        tcp_syn_sent_vals = []

        for s in samples:
            cpu_vals.append(g_num(s, 'cpu_percent'))
            mem_vals.append(g_num(s, 'mem_used_mb'))
            tcp_est_vals.append(g_num(s, ('meta', 'net', 'tcp_established')))
            tcp_listen_vals.append(g_num(s, ('meta', 'net', 'tcp_listen')))
            tcp_time_wait_vals.append(g_num(s, ('meta', 'net', 'tcp_time_wait')))
            tcp_syn_recv_vals.append(g_num(s, ('meta', 'net', 'tcp_syn_recv')))
            tcp_syn_sent_vals.append(g_num(s, ('meta', 'net', 'tcp_syn_sent')))

        def avg_safe(xs):
            xs2 = [safe_float(x) for x in xs]
            return float(statistics.mean(xs2)) if xs2 else 0.0

        cpu_avg = avg_safe(cpu_vals)
        try:
            cpu_std = float(statistics.pstdev([safe_float(x) for x in cpu_vals])) if len(cpu_vals) > 1 else 0.0
        except Exception:
            cpu_std = 0.0
        cpu_max = float(max([safe_float(x) for x in cpu_vals])) if cpu_vals else 0.0

        mem_delta = mem_vals[-1] - mem_vals[0] if mem_vals else 0.0

        tcp_est_avg = avg_safe(tcp_est_vals)
        tcp_est_max = float(max([safe_float(x) for x in tcp_est_vals])) if tcp_est_vals else 0.0
        tcp_listen_avg = avg_safe(tcp_listen_vals)
        tcp_listen_max = float(max([safe_float(x) for x in tcp_listen_vals])) if tcp_listen_vals else 0.0
        tcp_time_wait_avg = avg_safe(tcp_time_wait_vals)
        tcp_syn_recv_avg = avg_safe(tcp_syn_recv_vals)
        tcp_syn_sent_avg = avg_safe(tcp_syn_sent_vals)

        # build a minimal meta object (from last sample's meta, but prune)
        meta_out = {
            'window_sec': self.window_sec,
            'samples': len(samples),
            'host': {},
            'net': {}
        }
        last_meta = last.get('meta', {}) if isinstance(last.get('meta', {}), dict) else {}
        # copy small selection
        for k in ('hostname', 'host', 'id'):
            if k in last_meta.get('host', {}):
                meta_out['host'][k] = last_meta['host'][k]
        # copy some net metrics (last) to help downstream inspection
        if 'net' in last_meta and isinstance(last_meta['net'], dict):
            for k in ('rx_bytes', 'tx_bytes', 'tcp_established'):
                if k in last_meta['net']:
                    meta_out['net'][k] = last_meta['net'][k]

        # computed proxies / approximations
        computed = {
            'duration': float(duration),
            'src_bytes': float(rx_delta),
            'dst_bytes': float(tx_delta),
            'cpu_avg': float(cpu_avg),
            'cpu_std': float(cpu_std),
            'cpu_max': float(cpu_max),
            'mem_delta': float(mem_delta),
            'tcp_est_avg': float(tcp_est_avg),
            'tcp_est_max': float(tcp_est_max),
            'tcp_listen_avg': float(tcp_listen_avg),
            'tcp_listen_max': float(tcp_listen_max),
            'tcp_time_wait_avg': float(tcp_time_wait_avg),
            'tcp_syn_recv_avg': float(tcp_syn_recv_avg),
            'tcp_syn_sent_avg': float(tcp_syn_sent_avg),
        }

        return {'ts_ms': last['ts_ms'], 'computed': computed, 'meta': meta_out}


# ------------------------- KDD Mapping -------------------------

class KDDMapper:
    """Maps aggregator outputs to KDD feature_names (from checkpoint).

    Uses explicit proxy logic described in the spec.
    """

    # mapping semantics: for some KDD feature names we have direct or proxy mappings; others default to 0.
    APPROX_PROXY_FEATURES = {
        # proxies (derived from host/net metrics)
        'duration', 'src_bytes', 'dst_bytes',
        'count', 'srv_count',
        'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate',
        'dst_host_count', 'dst_host_srv_count',
        'wrong_fragment', 'urgent'
    }

    def __init__(self, feature_names: List[str], logger: logging.Logger):
        self.feature_names = list(feature_names)
        self.logger = logger

        # define which KDD features we can approximate and how
        # startup: summarize proxies
        will_proxy = [f for f in self.feature_names if f in self.APPROX_PROXY_FEATURES]
        unavailable = [f for f in self.feature_names if f not in will_proxy and f not in ('duration','src_bytes','dst_bytes') and not self._is_direct_mapped(f)]
        proxy_list = sorted(will_proxy + unavailable)
        self.logger.info("模型输入维度=%d", len(self.feature_names))
        # 避免超长列表刷屏：显示数量 + 前10个 + 省略提示
        if len(proxy_list) > 10:
            proxy_display = ', '.join(proxy_list[:10]) + f', ... 其余{len(proxy_list) - 10}个省略'
        else:
            proxy_display = ', '.join(proxy_list)
        self.logger.info("代理/近似或置零的特征数量=%d (%s)", len(proxy_list), proxy_display)

    def _is_direct_mapped(self, name: str) -> bool:
        # direct mapped ones we compute above
        return name in ('duration', 'src_bytes', 'dst_bytes')

    def map(self, agg: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        computed = agg['computed']
        proxy_features: List[str] = []

        # precompute some useful intermediate values
        tcp_est_avg = computed.get('tcp_est_avg', 0.0)
        tcp_syn_recv_avg = computed.get('tcp_syn_recv_avg', 0.0)
        tcp_syn_sent_avg = computed.get('tcp_syn_sent_avg', 0.0)
        tcp_listen_avg = computed.get('tcp_listen_avg', 0.0)

        count_proxy = tcp_est_avg + tcp_syn_recv_avg + tcp_syn_sent_avg
        srv_count_proxy = tcp_listen_avg

        serror_rate = (tcp_syn_recv_avg / (count_proxy if count_proxy > 0 else 1.0))
        srv_serror_rate = (tcp_syn_recv_avg / (srv_count_proxy if srv_count_proxy > 0 else 1.0))

        dst_host_count = computed.get('tcp_est_max', 0.0)
        dst_host_srv_count = computed.get('tcp_listen_max', 0.0)

        # this dict holds candidate values for common KDD fields
        candidate = {
            'duration': computed.get('duration', 0.0),
            'src_bytes': computed.get('src_bytes', 0.0),
            'dst_bytes': computed.get('dst_bytes', 0.0),
            'wrong_fragment': 0.0,
            'urgent': 0.0,
            'count': count_proxy,
            'srv_count': srv_count_proxy,
            'serror_rate': serror_rate,
            'srv_serror_rate': srv_serror_rate,
            'rerror_rate': 0.0,
            'srv_rerror_rate': 0.0,
            'same_srv_rate': 0.0,
            'diff_srv_rate': 0.0,
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            # additionally include some host metrics that may exist in feature_names
            'cpu_avg': computed.get('cpu_avg', 0.0),
            'cpu_std': computed.get('cpu_std', 0.0),
            'cpu_max': computed.get('cpu_max', 0.0),
            'mem_delta': computed.get('mem_delta', 0.0),
            'tcp_time_wait_avg': computed.get('tcp_time_wait_avg', 0.0)
        }

        # build final features ensuring ordering and presence
        features_out: Dict[str, float] = {}
        for name in self.feature_names:
            if name in candidate:
                val = safe_float(candidate[name], 0.0)
                # if the used value is an approximation for a nominal feature, mark as proxy
                if name in self.APPROX_PROXY_FEATURES:
                    proxy_features.append(name)
            else:
                # unknown, set 0 and mark as proxy
                val = 0.0
                proxy_features.append(name)
            features_out[name] = float(val)

        # deduplicate proxy_features
        proxy_features = sorted(set(proxy_features))

        return features_out, proxy_features


# ------------------------- Main CLI -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='kdd_mapper: aggregate lite_agent JSONL to pseudo-KDD JSONL')
    p.add_argument('--model-path', required=True, help='checkpoint path (e.g., ckpt/model.pth)')
    p.add_argument('--window-sec', type=float, default=2.0, help='sliding window length in seconds (default 2.0)')
    p.add_argument('--min-samples', type=int, default=2, help='minimum samples per window to emit (default 2)')
    p.add_argument('--max-rows', type=int, default=0, help='maximum output rows to emit (0 means unlimited)')
    p.add_argument('--log-level', default='INFO', help='log level (DEBUG/INFO/WARNING/ERROR)')
    p.add_argument('--quiet', action='store_true', help='equivalent to --log-level WARNING')
    return p.parse_args()


def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stderr, level=lvl, format='%(asctime)s [%(levelname)s] kdd_mapper: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger('kdd_mapper')


def load_checkpoint_feature_names(path: str, logger: logging.Logger) -> List[str]:
    if torch is None:
        logger.error('torch not available; cannot load checkpoint')
        raise SystemExit(2)
    try:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # 兼容旧版 torch（没有 weights_only 参数）
            ckpt = torch.load(path, map_location='cpu')

    except Exception as e:
        logger.exception('failed to load checkpoint: %s', e)
        raise SystemExit(2)
    if 'max_values' not in ckpt:
        logger.error("checkpoint missing 'max_values' key; cannot infer feature_names")
        raise SystemExit(2)
    mv = ckpt['max_values']
    # attempt to extract index-compatible names without pandas dependency
    if hasattr(mv, 'index'):
        try:
            feature_names = list(mv.index)
            return [str(x) for x in feature_names]
        except Exception:
            # fallback
            pass
    # maybe it's a dict/Series-like
    if isinstance(mv, dict):
        return [str(x) for x in mv.keys()]
    # last resort
    try:
        return [str(x) for x in list(mv)]
    except Exception:
        logger.error('unable to interpret checkpoint["max_values"] to list of names')
        raise SystemExit(2)


def main():
    args = parse_args()
    if args.quiet:
        args.log_level = 'WARNING'
    logger = setup_logging(args.log_level)

    logger.info('kdd_mapper 启动')
    logger.info('参数: model_path=%s window_sec=%.3f min_samples=%d max_rows=%d', args.model_path, args.window_sec, args.min_samples, args.max_rows)

    feature_names = load_checkpoint_feature_names(args.model_path, logger)
    logger.info('checkpoint加载成功：特征数=%d', len(feature_names))

    # Initialize classes
    aggregator = WindowAggregator(window_sec=args.window_sec, min_samples=args.min_samples, logger=logger)
    mapper = KDDMapper(feature_names=feature_names, logger=logger)

    # read stdin line by line
    out_count = 0
    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                logger.debug('skipping invalid json line')
                continue

            # validate incoming lite_agent format: must contain ts_ms & features dict
            if 'ts_ms' not in obj or 'features' not in obj or not isinstance(obj['features'], dict):
                logger.debug('skipping object missing ts_ms or features')
                continue

            agg_produced = aggregator.add_sample(obj)
            if agg_produced is not None:
                # map to kdd features
                features_out, proxy_feats = mapper.map(agg_produced)
                out_obj = {
                    'ts_ms': agg_produced['ts_ms'],
                    'features': features_out,
                    'meta': dict(agg_produced['meta'])
                }
                out_obj['meta']['proxy_features'] = proxy_feats

                # write JSONL to stdout only (wrap to handle downstream pipe closure)
                try:
                    sys.stdout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
                    sys.stdout.flush()
                except BrokenPipeError:
                    logger.info('下游管道已关闭（BrokenPipe），kdd_mapper 停止输出并退出')
                    try:
                        sys.stdout.close()
                    except Exception:
                        pass
                    return

                out_count += 1
                if args.max_rows > 0 and out_count >= args.max_rows:
                    logger.info('达到 --max-rows=%d，退出', args.max_rows)
                    break
    except KeyboardInterrupt:
        logger.info('被用户中断')

    # EOF: finalize
    final_agg = aggregator.finalize()
    if final_agg is not None and (args.max_rows == 0 or out_count < args.max_rows):
        features_out, proxy_feats = mapper.map(final_agg)
        out_obj = {
            'ts_ms': final_agg['ts_ms'],
            'features': features_out,
            'meta': dict(final_agg['meta'])
        }
        out_obj['meta']['proxy_features'] = proxy_feats
        try:
            sys.stdout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
            sys.stdout.flush()
        except BrokenPipeError:
            logger.info('下游管道已关闭（BrokenPipe），kdd_mapper 停止输出并退出')
            try:
                sys.stdout.close()
            except Exception:
                pass
            return
        out_count += 1

    # final statistics (one-time Chinese summary to stderr)
    total_in = aggregator.total_input_samples
    total_out = aggregator.emitted_samples
    dropped = aggregator.dropped_windows
    reason_list = ', '.join([f'{k}:{v}' for k, v in aggregator.drop_reasons.items()]) if aggregator.drop_reasons else '无'
    avg_comp = (sum(aggregator.window_compute_times) / len(aggregator.window_compute_times)) if aggregator.window_compute_times else 0.0

    logger.info('汇总: 输入采样=%d 产出样本=%d 丢弃窗口=%d 丢弃原因=%s 平均窗口耗时=%.6fs',
                total_in, total_out, dropped, reason_list, avg_comp)


if __name__ == '__main__':
    main()
