import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


def _ensure_state():
    ss = st.session_state
    ss.setdefault("rt_running", False)
    ss.setdefault("rt_status_text", "Idle")
    ss.setdefault("rt_mode", "—")
    ss.setdefault("rt_seed_offset", 0)

    ss.setdefault("kdd_input", False)
    ss.setdefault("os_input", False)
    ss.setdefault("kdd_count", 50)
    ss.setdefault("kdd_loop", False)


def _make_mock(seed: int = 42, n_events: int = 240, n_points: int = 60):
    random.seed(seed)
    np.random.seed(seed)

    now = datetime.now()

    # timeseries
    ts = [now - timedelta(seconds=(n_points - 1 - i) * 10) for i in range(n_points)]
    base_rate = np.clip(np.random.normal(32, 6, n_points), 5, 80)
    alerts = np.clip(
        np.random.poisson(2, n_points) + (np.random.rand(n_points) < 0.15) * np.random.randint(4, 10, n_points),
        0, 30
    )
    latency = np.clip(np.random.normal(28, 7, n_points) + alerts * 0.6, 5, 120)

    ts_df = pd.DataFrame({
        "ts": [t.strftime("%H:%M:%S") for t in ts],
        "rows_per_sec": base_rate.round(2),
        "alerts_per_min": alerts.astype(int),
        "latency_ms": latency.round(2),
    })

    # events
    preds = ["正常", "DoS", "Probe", "R2L", "U2R"]
    weights = [0.72, 0.14, 0.09, 0.04, 0.01]
    protos = ["TCP", "UDP", "ICMP"]
    actions = ["无操作", "重定向至蜜罐", "阻断", "限流"]

    def rand_ip():
        return f"{random.randint(10, 172)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    rows = []
    for i in range(n_events):
        p = random.choices(preds, weights=weights, k=1)[0]
        conf = round(random.uniform(0.65, 0.99) if p != "正常" else random.uniform(0.55, 0.93), 3)
        lat = round(max(3.0, random.gauss(26, 8) + (0 if p == "正常" else random.uniform(6, 22))), 2)
        proto = random.choice(protos)
        act = "无操作" if p == "正常" else random.choices(actions[1:], weights=[0.6, 0.25, 0.15], k=1)[0]
        rows.append({
            "idx": i + 1,
            "ts": (now - timedelta(seconds=(n_events - 1 - i) * 3)).strftime("%H:%M:%S"),
            "pred_name": p,
            "confidence": conf,
            "latency_ms": lat,
            "src_ip": rand_ip(),
            "dst_ip": rand_ip(),
            "proto": proto,
            "action": act,
        })
    events_df = pd.DataFrame(rows)

    kpi = {
        "rows_per_sec": float(ts_df["rows_per_sec"].iloc[-1]),
        "alerts": int((events_df["pred_name"] != "正常").sum()),
        "avg_latency": float(events_df["latency_ms"].mean()),
        "bad_rows": int(np.random.poisson(3)),
        "redirect_count": int((events_df["action"] == "重定向至蜜罐").sum()),
    }

    agent_sample = {
        "ts_ms": int(now.timestamp() * 1000),
        "features": {
            "cpu_percent": round(random.uniform(8, 72), 2),
            "mem_used_mb": round(random.uniform(1200, 9800), 1),
            "disk_used_pct": round(random.uniform(18, 88), 1),
            "rx_bytes": int(random.uniform(2e5, 6e6)),
            "tx_bytes": int(random.uniform(2e5, 6e6)),
        },
        "meta": {"host": "mock-host", "net": {"iface": "eth0"}},
    }

    mapper_sample = {
        "ts_ms": int(now.timestamp() * 1000),
        "window_sec": 2,
        "min_samples": 2,
        "feature_dim": 122,
        "meta": {
            "proxy_features": [
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_rerror_rate",
                "srv_serror_rate", "srv_rerror_rate"
            ]
        }
    }

    validator_stats = {
        "missing_field": int(np.random.poisson(1)),
        "parse_error": int(np.random.poisson(1)),
        "dim_mismatch": int(np.random.poisson(0)),
        "nan_inf": int(np.random.poisson(0)),
        "bad_examples": [
            {"reason": "missing_field", "sample": {"ts_ms": int(now.timestamp() * 1000), "features": {"cpu_percent": 12.3}}},
            {"reason": "parse_error", "sample": "not a json line"},
        ],
    }

    return {
        "ts_df": ts_df,
        "events_df": events_df,
        "kpi": kpi,
        "agent_sample": agent_sample,
        "mapper_sample": mapper_sample,
        "validator_stats": validator_stats,
    }


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


def render_realtime():
    st.set_page_config(page_title="实时入侵检测", layout="wide", initial_sidebar_state="collapsed")
    _ensure_state()
    ss = st.session_state

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

        /* KPI card */
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
        .stat-value { font-size: 26px; font-weight: 700; color: #0B5ED7; letter-spacing: 0.5px; }
        .stat-unit { font-size: 13px; color: rgba(15,23,42,0.5); margin-top: -4px; }

        /* 控制行：对齐优化 */
        .ctrl-label { font-size: 13px; color: rgba(15,23,42,0.60); margin-bottom: 4px; }
        .ctrl-wrap { padding: 12px 14px; border: 1px solid rgba(15,23,42,0.08); border-radius: 14px; background: #fff;
                     box-shadow: 0 6px 18px rgba(15,23,42,0.05); }
        /* 让 toggle/slider 更紧凑 */
        div[data-testid="stToggleSwitch"] { margin-top: -4px; }
        div[data-testid="stSlider"] { margin-top: -6px; }

        /* 表格美化 */
        .stDataFrame { border-radius: 12px; border: 1px solid rgba(15, 23, 42, 0.08); box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04); }
        .stDataFrame th { background: #f8fafc !important; font-weight: 600 !important; color: #0F172A !important; }

        /* 隐藏 Streamlit 默认 sidebar + 控件（兜底） */
        section[data-testid="stSidebar"], div[data-testid="collapsedControl"], button[data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarNav"], div[data-testid="stToolbar"], div[data-testid="stDecoration"]
        { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _nav_bar()

    # ---------- 控制台行 ----------
    st.markdown('<div class="section-title">控制台</div>', unsafe_allow_html=True)
    def _on_kdd():
        if ss.get("kdd_input"):
            ss["os_input"] = False

    def _on_os():
        if ss.get("os_input"):
            ss["kdd_input"] = False

    c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.35, 2.4, 1.6, 1.2, 1.2], vertical_alignment="center")
    with c1:
        st.toggle("KDD数据集", key="kdd_input", on_change=_on_kdd, disabled=ss.get("os_input", False), label_visibility="visible")
    with c2:
        st.toggle("OS系统态捕捉", key="os_input", on_change=_on_os, disabled=ss.get("kdd_input", False), label_visibility="visible")
    with c3:
        st.slider("测试条数", 10, 500, step=10, key="kdd_count", disabled=not ss.get("kdd_input", False))
    with c4:
        st.toggle("管道循环输入仿真", key="kdd_loop", disabled=not ss.get("kdd_input", False))
    # start/stop enable rules
    source_selected = bool(ss.get("kdd_input")) or bool(ss.get("os_input"))
    kdd_ok = bool(ss.get("kdd_input")) and (bool(ss.get("kdd_loop")) or int(ss.get("kdd_count", 0)) >= 10)
    os_ok = bool(ss.get("os_input"))
    can_start = (not ss.get("rt_running")) and (source_selected and (kdd_ok or os_ok))
    can_stop = bool(ss.get("rt_running"))

    with c5:
        if st.button("开始检测", use_container_width=True, disabled=not can_start):
            ss["rt_running"] = True
            ss["rt_status_text"] = "Running"
            ss["rt_mode"] = "KDD数据集" if ss.get("kdd_input") else "OS系统态捕捉"
    with c6:
        if st.button("结束检测", use_container_width=True, disabled=not can_stop):
            ss["rt_running"] = False
            ss["rt_status_text"] = "Idle"
            ss["rt_mode"] = "—"

    st.caption(f"状态：**{ss.get('rt_status_text','Idle')}**  |  模式：**{ss.get('rt_mode','—')}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # mock data
    if "rt_mock" not in ss:
        ss["rt_mock"] = _make_mock(seed=42 + int(ss.get("rt_seed_offset", 0)))
    mock = ss["rt_mock"]

    # ---------- KPI ----------
    st.markdown('<div class="section-title">实时概览</div>', unsafe_allow_html=True)
    k = mock["kpi"]
    k1, k2, k3, k4, k5 = st.columns(5)

    def _kpi(col, label, value, unit=""):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}<span class="stat-unit">{unit}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _kpi(k1, "输入速率", f"{k['rows_per_sec']:.1f}", " rows/s")
    _kpi(k2, "告警数", f"{k['alerts']}", "")
    _kpi(k3, "平均延迟", f"{k['avg_latency']:.1f}", " ms")
    _kpi(k4, "解析失败", f"{k['bad_rows']}", "")
    _kpi(k5, "重定向触发", f"{k['redirect_count']}", "")

    # ---------- Trend chart ----------
    st.markdown('<div class="section-title">监测趋势</div>', unsafe_allow_html=True)
    import plotly.graph_objects as go
    ts_df = mock["ts_df"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_df["ts"], y=ts_df["rows_per_sec"], mode="lines", name="rows/s"))
    fig.add_trace(go.Scatter(x=ts_df["ts"], y=ts_df["alerts_per_min"], mode="lines", name="alerts/min"))
    fig.add_trace(go.Scatter(x=ts_df["ts"], y=ts_df["latency_ms"], mode="lines", name="latency_ms", yaxis="y2"))

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=36, b=64),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            nticks=8
        ),
        yaxis=dict(title="rows/s, alerts/min"),
        yaxis2=dict(title="latency_ms", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---------- Events table + details (click row) ----------
    st.markdown('<div class="section-title">实时事件流</div>', unsafe_allow_html=True)

    left, right = st.columns([2.2, 1.0], vertical_alignment="top")
    events = mock["events_df"].copy()

    display_cols = ["idx", "ts", "pred_name", "confidence", "latency_ms", "src_ip", "dst_ip", "proto", "action"]
    display_df = events[display_cols].copy()

    with left:
        table_state = st.dataframe(
            display_df,
            use_container_width=True,
            height=520,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

    # selection (robust across versions)
    selected_row_idx = 0
    try:
        sel = getattr(table_state, "selection", None)
        if sel and getattr(sel, "rows", None) and len(sel.rows) > 0:
            selected_row_idx = int(sel.rows[0])
    except Exception:
        selected_row_idx = 0

    # default first row if none selected
    selected_row = display_df.iloc[selected_row_idx].to_dict()

    with right:
        st.markdown(
            '<div class="section-title" style="margin-top: 0px;">事件详情</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">检测结果</div>
              <div class="stat-value">{selected_row['pred_name']}<span class="stat-unit"> / conf {selected_row['confidence']}</span></div>
              <div style="margin-top:8px;color:rgba(15,23,42,0.7);font-size:13px;">latency: {selected_row['latency_ms']} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="stat-card" style="margin-top:12px;">
              <div class="stat-label">网络元信息</div>
              <div style="font-size:13px;color:rgba(15,23,42,0.8);margin-top:8px;">
                src: <b>{selected_row['src_ip']}</b><br/>
                dst: <b>{selected_row['dst_ip']}</b><br/>
                proto: <b>{selected_row['proto']}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="stat-card" style="margin-top:12px;">
              <div class="stat-label">响应动作</div>
              <div style="font-size:13px;color:rgba(15,23,42,0.8);margin-top:8px;">
                action: <b>{selected_row['action']}</b><br/>
                reason: <span style="color:rgba(15,23,42,0.7);">mock policy decision</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:14px;">原始片段（Mock）</div>', unsafe_allow_html=True)
        st.code(
            {
                "agent": mock["agent_sample"],
                "mapper": mock["mapper_sample"],
                "output": {
                    "pred_name": selected_row["pred_name"],
                    "confidence": selected_row["confidence"],
                    "latency_ms": selected_row["latency_ms"],
                },
            },
            language="python",
        )

    # ---------- Tabs ----------
    st.markdown('<div class="section-title">数据源与映射检查</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["lite_agent 原始采样", "kdd_mapper 聚合", "realtime_ids 输出校验"])
    with t1:
        a = mock["agent_sample"]
        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, "CPU", a["features"]["cpu_percent"], "%")
        _kpi(c2, "内存", a["features"]["mem_used_mb"], " MB")
        _kpi(c3, "RX", a["features"]["rx_bytes"], "")
        _kpi(c4, "TX", a["features"]["tx_bytes"], "")
        st.code(a, language="json")
    with t2:
        m = mock["mapper_sample"]
        st.write("proxy_features（近似/置零的特征）")
        st.dataframe(pd.DataFrame({"proxy_features": m["meta"]["proxy_features"]}), use_container_width=True, height=240, hide_index=True)
        st.code(m, language="json")
    with t3:
        v = mock["validator_stats"]
        st.dataframe(pd.DataFrame([{k: v[k] for k in ["missing_field", "parse_error", "dim_mismatch", "nan_inf"]}]),
                     use_container_width=True, hide_index=True)
        st.write("bad examples（mock）")
        st.code(v["bad_examples"], language="json")


if __name__ == "__main__":
    render_realtime()