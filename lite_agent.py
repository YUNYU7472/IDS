#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级 Agent：持续采集主机实时信息并以 JSONL 输出到 stdout（每行一个 JSON）

用法示例：
# 方案A：Agent -> 管道 -> realtime_ids 推理
python lite_agent.py --interval 0.2 --log-level WARNING | \
  python realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv --log-level INFO

# 测试：只跑 10 条
python lite_agent.py --interval 0.1 --max-rows 10 > host_stream.jsonl

输出要求：
- 每次采样一行 JSON（print(json.dumps(obj, ensure_ascii=False), flush=True)），stdout 只输出 JSONL
- 日志通过 stderr（logging）输出
- 支持 psutil 优先，否则降级到 /proc 或标准库

"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import time
import sys
import os
from typing import Optional, Dict, Any

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


# ---------- logging setup (stderr) ----------
class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        ms = int(record.msecs)
        return f"{t}.{ms:03d}"


def setup_logging(level_name: str = 'INFO') -> logging.Logger:
    level = logging.getLevelName(level_name)
    logger = logging.getLogger('lite_agent')
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers.clear()
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(level)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ch.setFormatter(MillisecondFormatter(fmt))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


# ---------- helpers for /proc parsing ----------

def _read_proc_stat() -> Optional[Dict[str, int]]:
    try:
        with open('/proc/stat', 'r') as f:
            for line in f:
                if line.startswith('cpu '):
                    parts = line.split()
                    # cpu user nice system idle iowait irq softirq steal guest guest_nice
                    vals = [int(x) for x in parts[1:]]
                    total = sum(vals)
                    idle = vals[3] if len(vals) > 3 else 0
                    return {'total': total, 'idle': idle}
    except Exception:
        return None
    return None


def _read_proc_net_dev() -> Optional[Dict[str, int]]:
    try:
        rx = 0
        tx = 0
        with open('/proc/net/dev', 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:
                if ':' not in line:
                    continue
                iface, rest = line.split(':', 1)
                parts = rest.split()
                if len(parts) >= 9:
                    rx += int(parts[0])
                    tx += int(parts[8])
        return {'rx_bytes': rx, 'tx_bytes': tx}
    except Exception:
        return None


def _count_tcp_states() -> Dict[str, int]:
    # mapping hex -> name
    hex_map = {
        '01': 'ESTABLISHED',
        '02': 'SYN_SENT',
        '03': 'SYN_RECV',
        '06': 'TIME_WAIT',
        '08': 'CLOSE_WAIT',
        '0A': 'LISTEN'
    }
    counts = {v: 0 for v in hex_map.values()}
    def _parse_file(path: str):
        try:
            with open(path, 'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        st = parts[3]
                        st = st.upper()
                        if st in hex_map:
                            counts[hex_map[st]] += 1
        except Exception:
            pass

    _parse_file('/proc/net/tcp')
    _parse_file('/proc/net/tcp6')
    return counts


def _count_udp_sockets() -> int:
    cnt = 0
    for path in ('/proc/net/udp', '/proc/net/udp6'):
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                cnt += max(0, len(lines) - 1)
        except Exception:
            pass
    return cnt


# ---------- Sampler class ----------
class Sampler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.hostname = socket.gethostname()
        self.use_psutil = _HAS_PSUTIL
        self.prev_stat = _read_proc_stat() if not self.use_psutil else None
        self.prev_net = _read_proc_net_dev() if not self.use_psutil else None
        self.prev_time = time.monotonic()
        # prime psutil cpu counter if available
        if self.use_psutil:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                self.use_psutil = False
        if self.use_psutil:
            self.logger.debug('使用 psutil 采集')
        else:
            self.logger.debug('未检测到 psutil，将使用 /proc 降级采集（若不可用会尽量使用标准库）')

    def sample(self) -> Dict[str, Any]:
        ts_ms = int(time.time() * 1000)
        host = self._sample_host()
        net = self._sample_net()
        top = self._sample_top()

        # features 占位，确保全部为数值（避免字符串）
        features = {
            'cpu_percent': float(host.get('cpu_percent')) if host.get('cpu_percent') is not None else 0.0,
            'mem_used_mb': float(host.get('mem_used_mb') or 0),
            'rx_bytes': float(net.get('rx_bytes') or 0)
        }

        meta = {'host': host, 'net': net}
        if top is not None:
            meta['top'] = top

        obj = {
            'ts_ms': ts_ms,
            'features': features,
            'meta': meta
        }
        return obj

    def _sample_host(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {'hostname': self.hostname}
        # CPU
        if self.use_psutil:
            try:
                cpu = psutil.cpu_percent(interval=None)
                res['cpu_percent'] = float(round(cpu, 2))
            except Exception:
                res['cpu_percent'] = None
        else:
            # try /proc/stat diff
            cur = _read_proc_stat()
            now = time.monotonic()
            if cur and self.prev_stat:
                total_delta = cur['total'] - self.prev_stat['total']
                idle_delta = cur['idle'] - self.prev_stat['idle']
                if total_delta > 0:
                    cpu_percent = 100.0 * (1.0 - (idle_delta / total_delta))
                    res['cpu_percent'] = float(round(cpu_percent, 2))
                else:
                    res['cpu_percent'] = None
            else:
                res['cpu_percent'] = None
            if cur:
                self.prev_stat = cur
            self.prev_time = now

        # loadavg
        try:
            load1, load5, load15 = os.getloadavg()
            res['load1'] = float(load1)
            res['load5'] = float(load5)
            res['load15'] = float(load15)
        except Exception:
            res['load1'] = None
            res['load5'] = None
            res['load15'] = None

        # memory
        try:
            if self.use_psutil:
                vm = psutil.virtual_memory()
                res['mem_total_mb'] = int(vm.total / 1024 / 1024)
                res['mem_used_mb'] = int((vm.total - vm.available) / 1024 / 1024)
                res['mem_percent'] = float(round(vm.percent, 2))
            else:
                # /proc/meminfo
                meminfo = {}
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            k, v = line.split(':', 1)
                            meminfo[k.strip()] = int(v.split()[0])
                    # values in kB
                    total_kb = meminfo.get('MemTotal')
                    free_kb = meminfo.get('MemFree', 0) + meminfo.get('Buffers', 0) + meminfo.get('Cached', 0)
                    if total_kb:
                        total_mb = int(total_kb / 1024)
                        used_mb = int((total_kb - free_kb) / 1024)
                        res['mem_total_mb'] = total_mb
                        res['mem_used_mb'] = used_mb
                        res['mem_percent'] = round(100.0 * used_mb / (total_mb if total_mb else 1), 2)
                    else:
                        res['mem_total_mb'] = None
                        res['mem_used_mb'] = None
                        res['mem_percent'] = None
                except Exception:
                    res['mem_total_mb'] = None
                    res['mem_used_mb'] = None
                    res['mem_percent'] = None
        except Exception:
            res['mem_total_mb'] = None
            res['mem_used_mb'] = None
            res['mem_percent'] = None

        # proc count
        try:
            if self.use_psutil:
                res['proc_count'] = int(len(psutil.pids()))
            else:
                # count directories in /proc that are digits
                cnt = 0
                try:
                    for name in os.listdir('/proc'):
                        if name.isdigit():
                            cnt += 1
                    res['proc_count'] = cnt
                except Exception:
                    res['proc_count'] = None
        except Exception:
            res['proc_count'] = None

        return res

    def _sample_net(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        if self.use_psutil:
            try:
                io = psutil.net_io_counters()
                res['rx_bytes'] = int(io.bytes_recv)
                res['tx_bytes'] = int(io.bytes_sent)
            except Exception:
                res['rx_bytes'] = None
                res['tx_bytes'] = None
        else:
            cur = _read_proc_net_dev()
            if cur:
                res['rx_bytes'] = cur['rx_bytes']
                res['tx_bytes'] = cur['tx_bytes']
            else:
                res['rx_bytes'] = None
                res['tx_bytes'] = None
        # tcp states
        try:
            tcp_states = _count_tcp_states()
            res['tcp_established'] = tcp_states.get('ESTABLISHED', 0)
            res['tcp_listen'] = tcp_states.get('LISTEN', 0)
            res['tcp_time_wait'] = tcp_states.get('TIME_WAIT', 0)
            res['tcp_syn_sent'] = tcp_states.get('SYN_SENT', 0)
            res['tcp_syn_recv'] = tcp_states.get('SYN_RECV', 0)
            res['tcp_close_wait'] = tcp_states.get('CLOSE_WAIT', 0)
        except Exception:
            res['tcp_established'] = None
            res['tcp_listen'] = None
            res['tcp_time_wait'] = None
            res['tcp_syn_sent'] = None
            res['tcp_syn_recv'] = None
            res['tcp_close_wait'] = None

        try:
            res['udp_sockets'] = _count_udp_sockets()
        except Exception:
            res['udp_sockets'] = None

        return res

    def _sample_top(self) -> Optional[Dict[str, Any]]:
        if not self.use_psutil:
            return None
        try:
            # find process with highest cpu_percent
            procs = []
            for p in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_info']):
                info = p.info
                # cpu_percent might be 0 on first call; call once to update
                if info.get('cpu_percent') is None:
                    continue
                procs.append(info)
            if not procs:
                return None
            procs.sort(key=lambda x: x.get('cpu_percent', 0.0), reverse=True)
            top = procs[0]
            return {
                'pid': int(top.get('pid')),
                'name': str(top.get('name')) if top.get('name') else None,
                'cpu_percent': float(round(top.get('cpu_percent', 0.0), 2)),
                'mem_mb': int(top.get('memory_info').rss / 1024 / 1024) if top.get('memory_info') else None
            }
        except Exception:
            return None


# ---------- main loop ----------

def main():
    parser = argparse.ArgumentParser(description='轻量级主机采样 Agent -> 输出 JSONL 到 stdout')
    parser.add_argument('--interval', type=float, default=0.5, help='采样间隔（秒），默认 0.5')
    parser.add_argument('--max-rows', type=int, default=0, help='最多采样条数，达到后自动退出（0 表示不限制）')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.debug(f"psutil present: {_HAS_PSUTIL}")

    sampler = Sampler(logger)
    row_count = 0
    samples = 0
    start = time.time()
    sample_times = []

    try:
        while True:
            t0 = time.monotonic()
            s0 = time.time()
            obj = sampler.sample()
            # stdout must only contain JSONL lines
            try:
                print(json.dumps(obj, ensure_ascii=False), flush=True)
            except BrokenPipeError:
                # 下游退出/管道关闭：安静结束，避免堆栈污染日志
                logger.info("下游管道已关闭（BrokenPipe），采集端停止输出并退出")
                return
            s1 = time.time()
            sample_times.append(s1 - s0)
            samples += 1
            row_count += 1

            if args.max_rows and row_count >= args.max_rows:
                logger.info(f"达到最大采样条数={args.max_rows}，准备退出")
                break

            # periodic logging
            if samples % 20 == 0:
                avg = sum(sample_times[-20:]) / min(len(sample_times), 20)
                logger.info(f"已采样 {samples} 条，最近平均采样耗时={avg*1000:.2f}ms")

            # sleep until next interval, accounting for work time
            elapsed = time.monotonic() - t0
            to_sleep = args.interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    except KeyboardInterrupt:
        logger.info("收到中断信号，准备退出")
    finally:
        total_time = time.time() - start
        avg_sample = (sum(sample_times) / len(sample_times)) if sample_times else 0.0
        logger.info(f"退出汇总 采样条数={samples} 运行时长={total_time:.3f}s 平均采样耗时={avg_sample*1000:.3f}ms")


if __name__ == '__main__':
    main()
