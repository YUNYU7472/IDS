#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
低交互蜜罐服务（课设级实现，使用标准库）

功能：
- 监听 TCP（默认 0.0.0.0:8080），并发处理连接
- 尽量读取一小段数据（默认最多 4096 字节，超时默认 2s）
- 根据收到的数据判断是否像 HTTP，请求返回合适的简短响应并关闭连接
- 每次连接写一行 JSON 到日志文件（JSONL）便于取证

注意：这是低交互诱捕（do not provide real services），不要用于替代真实服务。
"""
from __future__ import annotations

import argparse
import socket
import socketserver
import threading
import time
import json
import logging
import base64
import itertools
from typing import Tuple


# ---------- logger setup (millisecond timestamps) ----------
class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        ms = int(record.msecs)
        return f"{t}.{ms:03d}"


def setup_logger(level_name: str = "INFO") -> logging.Logger:
    level = logging.getLevelName(level_name)
    logger = logging.getLogger("honeypot")
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ch.setFormatter(MillisecondFormatter(fmt))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


logger = setup_logger()


# ---------- global state ----------
_file_lock = threading.Lock()
_counter = itertools.count(1)
_stats_lock = threading.Lock()
_stats = {
    "total_connections": 0,
    "beacon_count": 0,
    "raw_count": 0,
    "write_ok": 0,
    "write_fail": 0,
    "log_success": 0,
    "errors": 0,
}


# ---------- helper functions ----------
HTTP_METHODS = ("GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS")


def _is_http_preview(preview: str) -> bool:
    s = preview.lstrip().upper()
    for m in HTTP_METHODS:
        if s.startswith(m + " "):
            return True
    return False


def _preview_safe(b: bytes, max_chars: int = 200) -> str:
    # use latin-1 to preserve raw bytes in a JSON-safe str; limit length
    try:
        s = b.decode("latin-1", errors="replace")
    except Exception:
        s = base64.b64encode(b).decode("ascii")
    return s[:max_chars]


def _preview_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def explain() -> str:
    return (
        "低交互蜜罐：监听并接受被重定向/引流过来的可疑连接，尽量收集请求/探测的原始 bytes 并写入 JSONL 日志以便取证。"
        "与 response_actions.py 结合时，response_actions 可将检测到的攻击者引流到本蜜罐，从而形成“检测→响应→诱捕取证”的闭环证据链。"
    )


# ---------- TCP handler ----------
class HoneypotHandler(socketserver.BaseRequestHandler):
    def handle(self):
        args = self.server.args  # type: ignore[attr-defined]
        ts_ms = int(time.time() * 1000)
        conn_id = f"{ts_ms}-{next(_counter)}"

        client_ip, client_port = self.client_address[0], self.client_address[1]
        try:
            local_ip, local_port = self.request.getsockname()
        except Exception:
            local_ip, local_port = "", ""

        # increment total connections
        with _stats_lock:
            _stats["total_connections"] += 1

        try:
            # set read timeout
            try:
                self.request.settimeout(args.read_timeout)
            except Exception:
                pass

            # read up to max_bytes (single recv for simplicity)
            try:
                raw = self.request.recv(args.max_bytes)
            except socket.timeout:
                raw = b""
            except Exception:
                # in case socket already closed
                raw = b""

            bytes_received = len(raw)
            preview = _preview_safe(raw, max_chars=200)
            kind = "http" if _is_http_preview(preview) else "tcp"

            # detect beacon (HTTP POST to beacon path with JSON body)
            is_beacon = False
            beacon_payload = None
            if kind == "http":
                try:
                    # split raw bytes into head and body to preserve UTF-8 body encoding
                    sep = b"\r\n\r\n"
                    if sep in raw:
                        head_bytes, body_bytes = raw.split(sep, 1)
                    else:
                        head_bytes = raw
                        body_bytes = b""

                    # decode headers with latin-1 (safe) and parse
                    head = head_bytes.decode("latin-1", errors="replace")
                    first_line = head.splitlines()[0] if head else ""
                    parts = first_line.split()
                    method = parts[0] if len(parts) > 0 else ""
                    path = parts[1].split("?", 1)[0] if len(parts) > 1 else ""
                    headers = {}
                    for line in head.splitlines()[1:]:
                        if ":" in line:
                            k, v = line.split(":", 1)
                            headers[k.strip().lower()] = v.strip()

                    # if beacon, decode body as UTF-8 (to preserve Chinese) then parse JSON
                    if method.upper() == "POST" and path == args.beacon_path and "content-type" in headers and "application/json" in headers["content-type"]:
                        try:
                            body_str = body_bytes.decode("utf-8", errors="replace")
                            beacon_payload = json.loads(body_str)
                            is_beacon = True
                        except Exception as e:
                            logger.exception("failed to parse beacon JSON body: %s", e)
                            is_beacon = False
                except Exception as e:
                    logger.exception("exception while detecting beacon: %s", e)
                    is_beacon = False

            if is_beacon:
                # prepare unified event
                payload = beacon_payload or {}
                unified = {
                    "event_id": payload.get("event_id") or conn_id,
                    "ts_ms": payload.get("ts_ms") or int(time.time() * 1000),
                    "agent_capture": payload.get("agent_capture"),
                    "ids_result": payload.get("ids_result"),
                    "decision": payload.get("decision"),
                    "meta": payload.get("meta"),
                    "honeypot_observed": {
                        "client_ip": client_ip,
                        "client_port": client_port,
                        "local_ip": local_ip,
                        "local_port": local_port,
                        "bytes_received": bytes_received,
                        "kind": kind,
                        "preview": preview,
                        "preview_b64": _preview_b64(raw) if bytes_received else "",
                    },
                    "record_type": "beacon",
                }
                # append to JSONL
                try:
                    with _file_lock:
                        with open(args.log, "a", encoding="utf-8") as f:
                            f.write(json.dumps(unified, ensure_ascii=False) + "\n")
                    with _stats_lock:
                        _stats["beacon_count"] += 1
                        _stats["write_ok"] += 1
                    # respond 200 OK
                    resp_body = b"OK"
                    resp = b"HTTP/1.1 200 OK\r\n" + f"Content-Length: {len(resp_body)}\r\nConnection: close\r\n\r\n".encode("ascii") + resp_body
                    try:
                        self.request.sendall(resp)
                    except Exception:
                        pass
                    logger.info("beacon received conn=%s client=%s:%s event=%s", conn_id, client_ip, client_port, unified.get("event_id"))
                except Exception as e:
                    with _stats_lock:
                        _stats["beacon_count"] += 1
                        _stats["write_fail"] += 1
                        _stats["errors"] += 1
                    logger.exception("failed to write beacon unified event for conn=%s: %s", conn_id, e)
                # done handling beacon
            else:
                # proceed with original response and raw logging
                if kind == "http":
                    body = (args.http_response or "<html><body><h1>200 OK</h1></body></html>")
                    if isinstance(body, str):
                        body_bytes = body.encode("utf-8", errors="replace")
                    else:
                        body_bytes = body
                    resp = (b"HTTP/1.1 200 OK\r\n" + f"Content-Length: {len(body_bytes)}\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n".encode("ascii") + body_bytes)
                else:
                    banner = args.banner or "SSH-2.0-OpenSSH_8.2p1"
                    if not banner.endswith("\r\n"):
                        banner_bytes = (banner + "\r\n").encode("ascii", errors="replace")
                    else:
                        banner_bytes = banner.encode("ascii", errors="replace")
                    resp = banner_bytes

                # send response (best-effort)
                try:
                    self.request.sendall(resp)
                except Exception:
                    pass

                # build log entry
                entry = {
                    "conn_id": conn_id,
                    "ts_ms": ts_ms,
                    "client_ip": client_ip,
                    "client_port": client_port,
                    "local_ip": local_ip,
                    "local_port": local_port,
                    "bytes_received": bytes_received,
                    "preview": preview,
                    "preview_b64": _preview_b64(raw) if bytes_received else "",
                    "kind": kind,
                    "record_type": "raw",
                }

                # append to JSONL (thread-safe)
                try:
                    with _file_lock:
                        with open(args.log, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    with _stats_lock:
                        _stats["raw_count"] += 1
                        _stats["write_ok"] += 1
                    logger.info("logged conn=%s client=%s:%s kind=%s bytes=%d", conn_id, client_ip, client_port, kind, bytes_received)
                except Exception as e:
                    with _stats_lock:
                        _stats["raw_count"] += 1
                        _stats["write_fail"] += 1
                        _stats["errors"] += 1
                    logger.exception("failed to write log for conn=%s: %s", conn_id, e)
        except Exception as e:
            with _stats_lock:
                _stats["errors"] += 1
            logger.exception("error handling connection %s from %s:%s: %s", conn_id, client_ip, client_port, e)
        finally:
            # best-effort close
            try:
                self.request.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.request.close()
            except Exception:
                pass


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------- CLI / main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="低交互蜜罐服务（仅用于诱捕/取证，不提供真实服务）")
    p.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    p.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")
    p.add_argument("--log", default="honeypot_hits.jsonl", help="JSONL 日志文件路径（默认 honeypot_hits.jsonl）")
    p.add_argument("--read-timeout", type=float, default=2.0, help="socket 读取超时（秒，默认 2）")
    p.add_argument("--max-bytes", type=int, default=4096, help="单次读取最大字节数（默认 4096）")
    p.add_argument("--banner", default=None, help="非 HTTP 连接返回的 banner（默认示例）")
    p.add_argument("--http-response", default=None, help="自定义 HTTP 响应 body（可选）")
    p.add_argument("--beacon-path", default="/beacon", help="beacon 路径（默认 /beacon）")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    return p.parse_args()


def main():
    args = parse_args()

    # reset logger level
    global logger
    logger = setup_logger(args.log_level)

    logger.info("honeypot 启动：%s:%s log=%s beacon_path=%s", args.host, args.port, args.log, args.beacon_path)
    logger.info("配置 read_timeout=%s max_bytes=%s", args.read_timeout, args.max_bytes)

    server = ThreadedTCPServer((args.host, args.port), HoneypotHandler)
    server.args = args  # attach args

    try:
        server_thread = threading.Thread(target=server.serve_forever, name="HoneypotMain", daemon=True)
        server_thread.start()
        logger.info("监听中，按 Ctrl+C 停止")
        # wait until interrupted
        while server_thread.is_alive():
            server_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass

        # print summary stats to stderr via logger
        with _stats_lock:
            total = _stats.get("total_connections", 0)
            beacon_count = _stats.get("beacon_count", 0)
            raw_count = _stats.get("raw_count", 0)
            write_ok = _stats.get("write_ok", 0)
            write_fail = _stats.get("write_fail", 0)
            errors = _stats.get("errors", 0)
        logger.info("已关闭：总连接=%d 信标=%d 原始=%d 写入成功=%d 写入失败=%d 错误=%d", total, beacon_count, raw_count, write_ok, write_fail, errors)


if __name__ == '__main__':
    main()
