import streamlit as st
import pandas as pd
from datetime import datetime


def _nav_bar():
    st.markdown(
        """
        <div class="ghids-nav">
            <div class="ghids-brand">GHIDS</div>
            <div class="ghids-nav-right">
                <a class="ghids-nav-btn" href="/?page=home">首页</a>
                <a class="ghids-nav-btn" href="/?page=realtime">实时入侵检测</a>
                <a class="ghids-nav-btn" href="/?page=evidence">决策证据链</a>
                <a class="ghids-nav-btn" href="/?page=honeypot">蜜罐重定向</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_style():
    st.markdown(
        """
        <style>
        /* 隐藏Streamlit默认顶栏 & 移除组件白框 */
        header[data-testid="stHeader"] {display: none;}
        div[data-testid="stVerticalBlock"] > div {background: transparent !important; box-shadow: none !important;}
        div[data-testid="stHorizontalBlock"] > div {background: transparent !important; box-shadow: none !important;}

        /* 全局字体美化 */
        * { font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif !important; }

        /* 页面内边距 */
        .block-container {padding-top: 0.8rem; padding-bottom: 1.2rem; padding-left: 2rem; padding-right: 2rem;}

        /* NAV BAR */
        .ghids-nav {
          width: 100%;
          background: #0B5ED7;
          border-radius: 12px;
          padding: 12px 20px;
          display:flex;
          align-items:center;
          justify-content:space-between;
          box-shadow: 0 8px 24px rgba(11,94,215,0.2);
          margin-bottom: 20px;
        }
        .ghids-brand {
          color: #ffffff;
          font-weight: 700;
          font-size: 24px;
          letter-spacing: 0.8px;
          margin: 0;
          line-height: 1;
        }
        .ghids-nav-right{ display:flex; gap: 16px; }
        .ghids-nav-btn{
          border: 1px solid rgba(255,255,255,0.0);
          background: rgba(255,255,255,0.0);
          color: rgba(255,255,255,0.95);
          padding: 10px 16px;
          border-radius: 10px;
          font-weight: 600;
          font-size: 15px;
          cursor: pointer;
          user-select: none;
          transition: all 0.2s ease;
          text-decoration: none;
          display: inline-block;
        }
        .ghids-nav-btn:hover{
          background: rgba(255,255,255,0.2);
          border: 1px solid rgba(255,255,255,0.3);
          transform: translateY(-1px);
        }
        .ghids-nav-btn:visited { color: rgba(255,255,255,0.95); }

        /* section title */
        .section-title{
          font-size: 20px;
          font-weight: 700;
          color: #0F172A;
          margin: 20px 0 12px 0;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .section-title::before {
          content: "";
          width: 4px;
          height: 22px;
          background: #0B5ED7;
          border-radius: 2px;
        }

        /* KPI / 通用卡片 */
        .stat-card {
          display: flex;
          flex-direction: column;
          gap: 2px;
          padding: 10px 16px;
          background: linear-gradient(145deg, rgba(11,94,215,0.05), rgba(11,94,215,0.02));
          border: 1px solid rgba(11,94,215,0.12);
          border-radius: 12px;
          margin-bottom: 8px;
          transition: transform 0.2s ease;
        }
        .stat-card:hover { transform: translateY(-2px); }
        .stat-label { font-size: 14px; font-weight: 600; color: rgba(15,23,42,0.75); }
        .stat-value { font-size: 22px; font-weight: 700; color: #0B5ED7; letter-spacing: 0.5px; }
        .stat-unit { font-size: 13px; color: rgba(15,23,42,0.5); margin-left: 4px; }

        /* 隐藏 Streamlit 默认 sidebar + 控件（兜底） */
        section[data-testid="stSidebar"], div[data-testid="collapsedControl"], button[data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarNav"], div[data-testid="stToolbar"], div[data-testid="stDecoration"]
        { display: none !important; }

        /* badge / chip */
        .badge {
          display: inline-flex;
          align-items: center;
          padding: 2px 8px;
          border-radius: 999px;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.3px;
          border: 1px solid rgba(148,163,184,0.35);
          background: rgba(148,163,184,0.10);
          color: #475569;
        }
        .badge-ok { background: rgba(34,197,94,0.12); color: #15803D; border: 1px solid rgba(34,197,94,0.35); }
        .badge-warn { background: rgba(245,158,11,0.12); color: #B45309; border: 1px solid rgba(245,158,11,0.35); }
        .badge-danger { background: rgba(239,68,68,0.10); color: #B91C1C; border: 1px solid rgba(239,68,68,0.35); }
        .chip-row { display:flex; flex-wrap:wrap; gap:8px; margin-top: 8px; }
        .chip {
          display:inline-flex;
          align-items:center;
          gap:6px;
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 600;
          color: rgba(15,23,42,0.85);
          background: rgba(15,23,42,0.04);
          border: 1px solid rgba(15,23,42,0.08);
        }
        .chip b { color: #0F172A; font-weight: 800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_mock_data():
    now_ms = int(datetime.now().timestamp() * 1000)

    # 核心：只保留“第4节点 response_actions.py 必须解释的东西”
    event = {
        "event_id": "evt-20250109-0012",
        "ts_ms": now_ms,
        "action": "redirect",          # none / redirect
        "reason": "triggered",         # triggered / cooldown / low_confidence / whitelisted / port_not_match / label_policy / disabled
        "mode": "dry-run",
        "pred_name": "DoS",
        "conf": 0.942,
        "src_ip": "192.168.12.45",
        "dst_ip": "10.0.3.17",
        "dst_port": 8080,
        "proto": "TCP",
        "honeypot_ip": "10.0.9.23",
        "honeypot_port": 18080,
        "beacon_sent": True,
        "beacon_status": 200,
        "log_path": "honeypot_redirect.log",
    }

    # 关键：把“规则栈”精简成 3 件事：门槛、拦截点、触发点
    # 这里展示“为什么触发/为什么不触发”，不要把实时页已有的“事件流/趋势”搬过来
    policy = {
        "enabled": True,
        "label_policy": ["DoS", "Probe", "R2L", "U2R"],
        "confidence_threshold": 0.80,
        "cooldown_sec": 30,
        "watch_dport": 80,  # 你们 response_actions 里默认是 80；这里故意展示“被配置为关注端口”
        "whitelist_ips": ["10.0.0.8", "10.0.0.9"],
    }

    # 让这个 mock 看起来“像真实解释链”：明确指出“决定性规则”
    decisive = {
        "rule": "综合触发",
        "explain": (
            "enabled=True 且 label_policy 命中；conf≥threshold；未命中 whitelist；"
            "端口满足 watch_dport；cooldown 已过 → 输出 redirect（dry-run）。"
        ),
        "gates": [
            {"name": "enabled", "result": True, "detail": "response_actions.enabled=True"},
            {"name": "label_policy", "result": True, "detail": f"{event['pred_name']} ∈ allow-list"},
            {"name": "confidence", "result": True, "detail": f"{event['conf']} ≥ {policy['confidence_threshold']}"},
            {"name": "whitelist", "result": True, "detail": f"{event['src_ip']} not in whitelist"},
            {"name": "port", "result": True, "detail": f"dst_port={event['dst_port']} matches watch_dport"},
            {"name": "cooldown", "result": True, "detail": f"elapsed 38s ≥ {policy['cooldown_sec']}s"},
        ],
    }

    cmd = (
        f"# dry-run: redirect traffic from {event['src_ip']} destined to port {event['dst_port']} "
        f"to honeypot {event['honeypot_ip']}:{event['honeypot_port']} (mode=dry-run)"
    )

    # “审计闭环”只给一个 JSONL 样例（截断字段），避免和实时页重复
    audit_line = {
        "event_id": event["event_id"],
        "ts_ms": event["ts_ms"],
        "action": event["action"],
        "reason": event["reason"],
        "mode": event["mode"],
        "cmd": cmd,
        "src_ip": event["src_ip"],
        "dst_ip": event["dst_ip"],
        "dst_port": event["dst_port"],
        "pred_name": event["pred_name"],
        "conf": event["conf"],
        "logged": True,
        "beacon_sent": event["beacon_sent"],
        "beacon_status": event["beacon_status"],
    }

    # Beacon 展示只要“可追溯要素”，不展示冗长 b64/hashes
    beacon_min = {
        "event_id": event["event_id"],
        "ts_ms": event["ts_ms"],
        "meta": {"src_ip": event["src_ip"], "dst_ip": event["dst_ip"], "dst_port": event["dst_port"], "proto": event["proto"]},
        "ids_result": {"pred_name": event["pred_name"], "confidence": event["conf"]},
        "decision": {"action": event["action"], "reason": event["reason"], "mode": event["mode"]},
    }

    gates_df = pd.DataFrame(
        [
            {
                "Gate": g["name"],
                "Result": "PASS" if g["result"] else "BLOCK",
                "Detail": g["detail"],
            }
            for g in decisive["gates"]
        ]
    )

    return event, policy, decisive, cmd, audit_line, beacon_min, gates_df


def render_evidence():
    st.set_page_config(page_title="决策证据链", layout="wide", initial_sidebar_state="collapsed")
    _inject_style()
    _nav_bar()

    event, policy, decisive, cmd, audit_line, beacon_min, gates_df = _build_mock_data()

    # =========================
    # 1) 一句话结论（只讲决策，不讲监控）
    # =========================
    st.markdown('<div class="section-title">本次决策结论</div>', unsafe_allow_html=True)

    badge_action = "badge-danger" if event["action"] == "redirect" else "badge-ok"
    badge_reason = "badge-ok" if event["reason"] == "triggered" else "badge-warn"

    a1, a2, a3, a4 = st.columns([1.3, 1.0, 1.0, 1.2])
    with a1:
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">Action（最终动作）</div>
              <div class="stat-value">{event['action']} <span class="badge {badge_action}" style="margin-left:8px;">{event['mode']}</span></div>
              <div class="stat-unit">event_id={event['event_id']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">Reason（触发原因）</div>
              <div class="stat-value">{event['reason']} <span class="badge {badge_reason}" style="margin-left:8px;">rule</span></div>
              <div class="stat-unit">第4节点响应决策输出</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">Threat Verdict（模型输出）</div>
              <div class="stat-value">{event['pred_name']}</div>
              <div class="stat-unit">confidence={event['conf']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a4:
        beacon_badge = "badge-ok" if event["beacon_sent"] else "badge-warn"
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">Audit & Beacon（闭环）</div>
              <div class="stat-value">{'OK' if event['beacon_sent'] else 'PENDING'} <span class="badge {beacon_badge}" style="margin-left:8px;">status={event['beacon_status']}</span></div>
              <div class="stat-unit">log={event['log_path']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================
    # 2) 决策门控（精简版规则栈）
    # =========================
    st.markdown('<div class="section-title">决策门控</div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1.1])
    with left:
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">决定性规则</div>
              <div style="margin-top:6px;font-size:14px;font-weight:800;color:#0F172A;">
                {decisive['rule']}
              </div>
              <div style="margin-top:6px;font-size:12px;color:rgba(15,23,42,0.72);line-height:1.7;">
                • <span style="color:#0B5ED7;font-weight:600;">触发前提</span>：
                  enabled=<b>{str(policy['enabled']).lower()}</b> 且
                  label_policy 允许 <b>{event['pred_name']}</b><br/>
                • <span style="color:#0B5ED7;font-weight:600;">置信门槛</span>：
                  conf=<b>{event['conf']}</b> ≥ threshold=<b>{policy['confidence_threshold']}</b><br/>
                • <span style="color:#0B5ED7;font-weight:600;">策略约束</span>：
                  未命中 whitelist，dst_port=<b>{event['dst_port']}</b> 命中 watch_dport，
                  冷却时间满足 <b>{policy['cooldown_sec']}s</b><br/>
                • <span style="color:#0B5ED7;font-weight:600;">结论</span>：
                  输出 action=<b>{event['action']}</b>（mode=<b>{event['mode']}</b>），
                  同步写入审计日志并发送 beacon 以支撑取证与追溯。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="stat-card" style="margin-top:10px;">
              <div class="stat-label">关键阈值</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:-2px;">
              <div class="stat-card" style="margin-bottom:0;">
                <div class="stat-label">confidence_threshold</div>
                <div class="stat-value">{policy['confidence_threshold']}</div>
                <div class="stat-unit">conf 必须 ≥ 阈值</div>
              </div>
              <div class="stat-card" style="margin-bottom:0;">
                <div class="stat-label">cooldown_sec</div>
                <div class="stat-value">{policy['cooldown_sec']}<span class="stat-unit">s</span></div>
                <div class="stat-unit">防止频繁触发</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        # 用小表格呈现门控结果：比时间线更“干净”
        st.markdown(
            """
            <div class="stat-card">
              <div class="stat-label">门控检查结果</div>
              <div style="font-size:12px;color:rgba(15,23,42,0.70);margin-top:6px;">
                PASS 表示该检查未阻断；若发生 BLOCK，则此处将成为最终 reason 的来源。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gates_view = gates_df.copy()
        gates_view = gates_view.dropna(how="all")
        gates_view = gates_view[gates_view["Gate"].astype(str).str.strip() != ""]
        gates_view = gates_view.reset_index(drop=True)
        st.dataframe(
            gates_view,
            use_container_width=True,
            height=280,
            hide_index=True,
        )

    # =========================
    # 3) 审计闭环（只给“证据落点”与“最小payload”）
    # =========================
    st.markdown('<div class="section-title" style="margin-top: 10px;">审计闭环</div>', unsafe_allow_html=True)

    b1, b2 = st.columns([1.0, 1.0])
    with b1:
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">JSONL 审计记录</div>
              <div style="font-size:12px;color:rgba(15,23,42,0.62);margin-top:6px;line-height:1.7;">
                写入：<span style="color:#0B5ED7;font-weight:800;">{event['log_path']}</span>（每行一个事件；字段可用于离线复现与问责审计）
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(audit_line, language="json")

    with b2:
        st.markdown(
            """
            <div class="stat-card">
              <div class="stat-label">Beacon（发往蜜罐的最小闭环 payload）</div>
              <div style="font-size:12px;color:rgba(15,23,42,0.62);margin-top:6px;line-height:1.7;">
                不展示冗长 b64/hash，仅保留
                <span style="color:#0B5ED7;font-weight:800;">可追溯关键字段</span>：
                <span style="color:#0B5ED7;font-weight:800;">meta + ids_result + decision</span>。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(beacon_min, language="json")


if __name__ == "__main__":
    render_evidence()
