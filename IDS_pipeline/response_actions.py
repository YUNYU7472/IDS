#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蜜罐重定向响应模块（dry-run, 仅记录与日志）
- 仅使用标准库
- 通过 JSONL 文件追加写入每次决策（取证链）
- 提供 CLI --demo 用于本地演示
"""
from __future__ import annotations

import sys
import time
import json
import logging
import uuid
import hashlib
import base64
import http.client
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# module-level cooldown store: src_ip -> last_ts_ms
_last_trigger_ts: Dict[str, int] = {}


@dataclass
class ResponseConfig:
    enabled: bool = True
    confidence_threshold: float = 0.80
    cooldown_sec: int = 30
    mode: str = "dry-run"  # 仅支持 dry-run（不要执行 iptables），但可以生成 cmd 字符串
    honeypot_ip: str = "127.0.0.1"
    honeypot_port: int = 8080
    watch_dport: int = 80
    log_path: str = "honeypot_redirect.log"  # JSONL，追加写
    whitelist_ips: List[str] = field(default_factory=list)
    label_policy: Dict[int, bool] = field(default_factory=lambda: {1: True, 2: True, 3: True, 4: True})

    # Beacon settings (向 honeypot 发送事件通知)
    beacon_enabled: bool = True
    beacon_timeout_sec: float = 0.8
    beacon_path: str = "/beacon"
    beacon_max_chars: int = 2048
    beacon_retry: int = 1


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("response_actions")
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers.clear()
    ch = logging.StreamHandler(stream=sys.stderr)

    class MillisecondFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            ct = self.converter(record.created)
            t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
            ms = int(record.msecs)
            return f"{t}.{ms:03d}"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ch.setFormatter(MillisecondFormatter(fmt))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


_logger = setup_logger()  # module-level logger


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def maybe_redirect_to_honeypot(meta: Dict[str, Any], pred: int, conf: float, cfg: ResponseConfig | None = None) -> Dict[str, Any]:
    """决定是否（dry-run）重定向到蜜罐，并记录 JSONL 日志。

    返回包含 ts_ms, action, reason, mode, cmd, logged, src_ip, dst_ip, src_port, dst_port, proto, pred, conf, honeypot_ip, honeypot_port
    并确保不向 stdout 打印，记录到 stderr logger 及 JSONL 文件（追加）。
    所有异常须被捕获，不能抛出。
    """
    if cfg is None:
        cfg = ResponseConfig()

    now_ms = int(time.time() * 1000)

    # normalize meta fields
    src_ip = str(meta.get("src_ip") or meta.get("src") or "unknown")
    dst_ip = str(meta.get("dst_ip") or meta.get("dst") or "unknown")
    proto = str(meta.get("proto") or meta.get("protocol") or "")
    src_port_raw = meta.get("src_port", "")
    dst_port_raw = meta.get("dst_port", "")

    src_port = _safe_int(src_port_raw) if src_port_raw not in (None, "") else None
    dst_port = _safe_int(dst_port_raw) if dst_port_raw not in (None, "") else None

    # generate stable event_id immediately to avoid NameError in exceptions
    event_id = f"{now_ms}-{src_ip}-{pred}-{uuid.uuid4().hex[:6]}"

    result: Dict[str, Any] = {
        "event_id": event_id,
        "ts_ms": now_ms,
        "action": "none",
        "reason": "not_triggered",
        "mode": cfg.mode,
        "cmd": "",
        "logged": False,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port if src_port is not None else "",
        "dst_port": dst_port if dst_port is not None else "",
        "proto": proto,
        "pred": pred,
        "conf": float(conf),
        "honeypot_ip": cfg.honeypot_ip,
        "honeypot_port": cfg.honeypot_port,
    }

    try:
        if not cfg.enabled:
            result["reason"] = "disabled"
            _logger.debug("response disabled by config")
            # write log and return
        elif pred == 0 or not cfg.label_policy.get(pred, False):
            result["reason"] = "label_policy"
            _logger.debug("label_policy prevents response")
        elif conf < cfg.confidence_threshold:
            result["reason"] = "low_confidence"
            _logger.debug("confidence below threshold")
        elif src_ip in cfg.whitelist_ips:
            result["reason"] = "whitelisted"
            _logger.debug("src_ip whitelisted")
        elif cfg.watch_dport and dst_port is not None and dst_port != cfg.watch_dport:
            result["reason"] = "port_not_match"
            _logger.debug("dst_port not matching watch_dport")
        else:
            # cooldown logic (special-case anonymous or NA src_ips)
            if src_ip in ("NA", "unknown", ""):
                cooldown_key = f"{src_ip}:{pred}:{dst_port if dst_port is not None else ''}"
            else:
                cooldown_key = src_ip

            last_ts = _last_trigger_ts.get(cooldown_key)
            if last_ts is not None and (now_ms - last_ts) < (cfg.cooldown_sec * 1000):
                result["reason"] = "cooldown"
                _logger.debug("cooldown not elapsed for key=%s (event=%s)", cooldown_key, result.get("event_id", "NA"))
            else:
                # trigger redirect (dry-run), generate cmd string but don't execute
                result["action"] = "redirect"
                result["reason"] = "triggered"
                result["mode"] = "dry-run"
                # a human-readable cmd-like explanation
                result["cmd"] = (
                    f"# dry-run: redirect traffic from {src_ip} destined to port {dst_port if dst_port is not None else cfg.watch_dport} "
                    f"to honeypot {cfg.honeypot_ip}:{cfg.honeypot_port} (mode=dry-run)"
                )
                _last_trigger_ts[cooldown_key] = now_ms
                _logger.info("dry-run redirect prepared for %s -> %s:%s (key=%s, event=%s)", src_ip, cfg.honeypot_ip, cfg.honeypot_port, cooldown_key, result.get("event_id", "NA"))

        # write JSONL log (append)
        # Set logged=True in the payload to reflect the intent to log; if write fails we'll update logged=False and attempt a fallback record
        result["logged"] = True
        try:
            line = json.dumps(result, ensure_ascii=False) + "\n"
            with open(cfg.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            # mark failure and try to write a small failure record to an alternate file so there is an audit trail
            result["logged"] = False
            result["reason"] = (result.get("reason") or "") + "|log_failed"
            _logger.exception("failed to write redirect log for event=%s: %s", result.get("event_id", "NA"), e)
            try:
                fallback_path = cfg.log_path + ".failed"
                with open(fallback_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                _logger.warning("wrote fallback log to %s for event=%s", fallback_path, result.get("event_id", "NA"))
            except Exception:
                _logger.exception("failed to write fallback redirect log for event=%s", result.get("event_id", "NA"))
        # if redirect, optionally send beacon to honeypot
        if result.get("action") == "redirect" and cfg.beacon_enabled:
            try:
                # prepare agent_capture: prefer agent_capture or agent_preview in meta
                ac = None
                if isinstance(meta.get("agent_capture"), str):
                    ac = meta.get("agent_capture")
                elif isinstance(meta.get("agent_preview"), str):
                    ac = meta.get("agent_preview")
                else:
                    # fallback: small safe summary of meta
                    try:
                        ac = json.dumps({k: meta.get(k) for k in ("src_ip", "dst_ip", "src_port", "dst_port", "proto")}, ensure_ascii=False)
                    except Exception:
                        ac = str(meta)[:cfg.beacon_max_chars]

                # truncate
                if ac is None:
                    ac = ""
                if len(ac) > cfg.beacon_max_chars:
                    ac_trunc = ac[:cfg.beacon_max_chars]
                else:
                    ac_trunc = ac

                # compute hash and base64
                try:
                    ac_bytes = ac_trunc.encode("utf-8", errors="replace")
                    ac_hash = hashlib.sha256(ac_bytes).hexdigest()
                    ac_b64 = base64.b64encode(ac_bytes).decode("ascii")
                except Exception:
                    ac_hash = ""
                    ac_b64 = ""

                payload = {
                    "event_id": event_id,
                    "ts_ms": now_ms,
                    "agent_capture": ac_trunc,
                    "agent_capture_b64": ac_b64,
                    "agent_capture_hash": ac_hash,
                    "meta": {k: meta.get(k) for k in ("src_ip", "dst_ip", "src_port", "dst_port", "proto")},
                    "ids_result": {"pred_label": pred, "pred_name": meta.get("pred_name", ""), "confidence": float(conf)},
                    "decision": {"action": result.get("action"), "reason": result.get("reason"), "mode": result.get("mode"), "cmd": result.get("cmd")},
                }

                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

                attempts = cfg.beacon_retry + 1
                backoff = 0.2
                beacon_sent = False
                beacon_status = None
                for attempt in range(attempts):
                    try:
                        conn = http.client.HTTPConnection(cfg.honeypot_ip, cfg.honeypot_port, timeout=cfg.beacon_timeout_sec)
                        conn.request("POST", cfg.beacon_path, body=body, headers={"Content-Type": "application/json"})
                        resp = conn.getresponse()
                        beacon_status = resp.status
                        # read and close to free connection
                        try:
                            resp.read()
                        except Exception:
                            pass
                        conn.close()
                        beacon_sent = True
                        _logger.info("beacon sent event_id=%s to %s:%s%s status=%s", event_id, cfg.honeypot_ip, cfg.honeypot_port, cfg.beacon_path, beacon_status)
                        break
                    except Exception as e:
                        _logger.debug("beacon attempt %d failed: %s", attempt + 1, e)
                        time.sleep(backoff)
                        backoff *= 2

                result["beacon_sent"] = beacon_sent
                result["beacon_status"] = beacon_status

            except Exception:
                _logger.exception("exception while sending beacon for event_id=%s", event_id)
                result["beacon_sent"] = False
                result["beacon_status"] = "error"

    except Exception as e:
        # any unexpected error shouldn't crash caller
        result["action"] = "none"
        result["reason"] = "error"
        result["cmd"] = ""
        result["logged"] = False
        _logger.exception("exception in maybe_redirect_to_honeypot: %s", e)

    return result


def explain() -> str:
    return (
        "检测→响应闭环：系统在检测到高置信度的攻击行为后（检测），依据策略判断是否触发重定向（响应），"
        "以 dry-run 方式生成可解释的重定向指令并追加写入 JSONL 日志（取证），整个流程形成可审计的证据链。"
    )


def _demo_run():
    cfg = ResponseConfig()
    _logger.info("演示模式开始：将创建若干模拟事件并写入 %s", cfg.log_path)

    samples = [
        ({"src_ip": "1.2.3.4", "dst_ip": "10.0.0.1", "dst_port": 80, "proto": "tcp"}, 1, 0.95),
        ({"src_ip": "5.6.7.8", "dst_ip": "10.0.0.1", "dst_port": 22, "proto": "tcp"}, 2, 0.99),
        ({"src_ip": "127.0.0.1", "dst_ip": "10.0.0.1", "dst_port": 80}, 1, 0.50),
    ]

    for meta, pred, conf in samples:
        res = maybe_redirect_to_honeypot(meta=meta, pred=pred, conf=conf, cfg=cfg)
        _logger.info("demo result: src=%s pred=%s conf=%.2f action=%s reason=%s logged=%s", meta.get("src_ip"), pred, conf, res.get("action"), res.get("reason"), res.get("logged"))

    _logger.info("演示结束，请查看日志文件：%s（JSONL，每行一个事件）", cfg.log_path)


def _cli():
    if "--demo" in sys.argv:
        _demo_run()
    else:
        print(explain(), file=sys.stderr)


if __name__ == '__main__':
    _cli()
