import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ===============================
# Page config
# ===============================
def render_home():
    """首页渲染函数（封装原有首页逻辑）"""
    st.set_page_config(page_title="GHIDS Dashboard", layout="wide", initial_sidebar_state="collapsed")

    # ===============================
    # Global CSS (全面美化样式)
    # ===============================
    st.markdown(
        """
    <style>
    /* 隐藏Streamlit默认顶栏 & 移除组件白框 */
    header[data-testid="stHeader"] {display: none;}
    div[data-testid="stVerticalBlock"] > div {background: transparent !important; box-shadow: none !important;}
    div[data-testid="stHorizontalBlock"] > div {background: transparent !important; box-shadow: none !important;}

    /* 全局字体美化 */
    * {
        font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif !important;
    }

    /* 页面内边距 */
    .block-container {padding-top: 0.8rem; padding-bottom: 1.2rem; padding-left: 2rem; padding-right: 2rem;}

    /* NAV BAR 优化 */
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
    .ghids-nav-right{
      display:flex;
      gap: 16px;
    }
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
    }
    .ghids-nav-btn:hover{
      background: rgba(255,255,255,0.2);
      border: 1px solid rgba(255,255,255,0.3);
      transform: translateY(-1px);
    }
    /* 为 a 标签补充样式（仅新增，不改原有样式） */
    .ghids-nav-btn {
      text-decoration: none;
      display: inline-block;
    }
    .ghids-nav-btn:visited {
      color: rgba(255,255,255,0.95);
    }

    /* KPI CARD 增大+美化 */
    .kpi-card{
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 16px;
      padding: 20px 20px 16px 20px;
      background: #ffffff;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
      height: 100%;
      transition: box-shadow 0.3s ease;
    }
    .kpi-card:hover {
      box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
    }
    .kpi-title{
      font-size: 18px;
      font-weight: 700;
      color: #0F172A;
      margin: 0 0 16px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .kpi-title::before {
      content: "";
      width: 4px;
      height: 20px;
      background: #0B5ED7;
      border-radius: 2px;
    }
    .kpi-subtle{
      color: rgba(15,23,42,0.55);
      font-size: 13px;
      margin-top: 10px;
      padding-left: 12px;
    }
    .model-update-row{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 10px;
      padding-left: 12px;
      padding-right: 12px;
    }
    .model-update-time{
      color: rgba(15,23,42,0.55);
      font-size: 13px;
    }
    .model-update-btn{
      border: 1px solid rgba(11,94,215,0.3);
      background: rgba(11,94,215,0.08);
      color: #0B5ED7;
      padding: 4px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      user-select: none;
      transition: all 0.2s ease;
      text-decoration: none;
      display: inline-block;
    }
    .model-update-btn:hover{
      background: rgba(11,94,215,0.15);
      border-color: rgba(11,94,215,0.5);
      transform: translateY(-1px);
    }
    /* Streamlit按钮美化 */
    button[kind="secondary"] {
      border: 1px solid rgba(11,94,215,0.3) !important;
      background: rgba(11,94,215,0.08) !important;
      color: #0B5ED7 !important;
      padding: 4px 12px !important;
      border-radius: 6px !important;
      font-size: 12px !important;
      font-weight: 500 !important;
      height: auto !important;
      min-height: auto !important;
    }
    button[kind="secondary"]:hover {
      background: rgba(11,94,215,0.15) !important;
      border-color: rgba(11,94,215,0.5) !important;
    }

    /* 系统运行概览 数值美化 */
    .stat-card {
      display: flex;
      flex-direction: column;
      gap: 2px;          /* 减小内部间距 */
      padding: 10px 16px;/* 上下内边距减小为10px，左右保持16px */
      background: linear-gradient(145deg, rgba(11,94,215,0.05), rgba(11,94,215,0.02));
      border: 1px solid rgba(11,94,215,0.12);
      border-radius: 12px;
      margin-bottom: 8px;/* 减小卡片间距 */
      transition: transform 0.2s ease;
    }
    .stat-card:hover {
      transform: translateY(-2px);
    }
    .stat-label {
      font-size: 14px;
      font-weight: 600;
      color: rgba(15,23,42,0.75);
    }
    .stat-value {
      font-size: 26px;
      font-weight: 700;
      color: #0B5ED7;
      letter-spacing: 0.5px;
    }
    .stat-unit {
      font-size: 13px;
      color: rgba(15,23,42,0.5);
      margin-top: -4px;
    }

    /* Events section 美化 */
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

    /* 表格美化 */
    .stDataFrame {
      border-radius: 12px;
      border: 1px solid rgba(15, 23, 42, 0.08);
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }
    .stDataFrame th {
      background: #f8fafc !important;
      font-weight: 600 !important;
      color: #0F172A !important;
    }
    
    /* 隐藏 Streamlit 默认 sidebar（两层兜底） */
    section[data-testid="stSidebar"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    div[data-testid="collapsedControl"] { display: none !important; }
    button[data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    </style>
        """,
        unsafe_allow_html=True,
    )

    # ===============================
    # NAV BAR
    # ===============================
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

    # ===============================
    # 模拟数据
    # ===============================
    model_report = {
        "timestamp": "20260107_103522",
        "final_accuracy": 0.9554912127129486,
        "final_loss": 0.13542708626632527,
        "training_time_seconds": 2039.867615222931,
        "precision": 0.9647412928192632,
        "recall": 0.941925552756787,
        "f1_score": 0.9531969128372159,
    }
    model_time_display = "2026-01-07 10:35:22"

    radar_metrics = {
        "准确率": model_report["final_accuracy"],
        "精确率": model_report["precision"],
        "召回率": model_report["recall"],
        "F1值": model_report["f1_score"],
        "稳定性(1-Loss)": 1.0 - model_report["final_loss"],
    }

    alarm_labels = ["正常", "拒绝服务(DoS)", "探测(Probe)", "远程登录(R2L)", "越权提权(U2R)"]
    alarm_values = [58, 16, 12, 9, 5]

    stats = [
        ("输入样本数", "12,480", "条"),
        ("告警总数", "3,214", "条"),
        ("触发决策数", "487", "次"),
        ("蜜罐命中数", "129", "次"),
    ]

    # ===============================
    # KPI区域（增大卡片）
    # ===============================
    k1, k2, k3 = st.columns([2.2, 2.1, 1.5], gap="large")

    # ---------- Card 1: 雷达图（增大高度） ----------
    with k1:
        st.markdown('<div class="kpi-title">最新模型概览</div>', unsafe_allow_html=True)

        radar_df = pd.DataFrame({"metric": list(radar_metrics.keys()), "value": list(radar_metrics.values())})
        radar_df_closed = pd.concat([radar_df, radar_df.iloc[[0]]], ignore_index=True)

        radar_fig = go.Figure()
        radar_fig.add_trace(
            go.Scatterpolar(
                r=radar_df_closed["value"],
                theta=radar_df_closed["metric"],
                fill="toself",
                hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
                line=dict(color="#0B5ED7", width=2),
                fillcolor="rgba(11,94,215,0.2)",
            )
        )
        radar_fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickformat=".1f", tickfont=dict(size=12)),
                angularaxis=dict(tickfont=dict(size=13))
            ),
            showlegend=False,
            height=400,  # 增大雷达图高度
            plot_bgcolor='rgba(0,0,0,0)'  # 修复：使用标准透明色值
        )
        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

        col_time, col_btn = st.columns([3, 1])
        with col_time:
            st.markdown(f'<div class="model-update-time" style="margin-top: 10px; padding-left: 12px;">模型更新时间：{model_time_display}</div>', unsafe_allow_html=True)
        with col_btn:
            st.button("拉取最新模型", key="pull_model_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Card 2: 饼图 ----------
    with k2:
        st.markdown('<div class="kpi-title">告警类型占比</div>', unsafe_allow_html=True)

        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=alarm_labels,
                    values=alarm_values,
                    hole=0.4,
                    hovertemplate="%{label}<br>占比: %{percent}<br>数量: %{value}<extra></extra>",
                    marker=dict(colors=["#36CFC9", "#FF7A45", "#722ED1", "#FFC53D", "#F5222D"]),
                    textfont=dict(size=13)
                )
            ]
        )
        pie_fig.update_layout(
            margin=dict(l=20, r=60, t=20, b=20),
            showlegend=True,
            legend=dict(orientation="v", x=1.05, y=0.5, font=dict(size=12)),
            height=400,  # 同步增大饼图高度
            plot_bgcolor='rgba(0,0,0,0)'  # 修复：使用标准透明色值
        )
        st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Card 3: 系统运行概览（美化数值） ----------
    with k3:
        st.markdown('<div class="kpi-title">系统运行概览</div>', unsafe_allow_html=True)

        for label, value, unit in stats:
            st.markdown(
                f"""
                <div class="stat-card">
                  <div class="stat-label">{label}</div>
                  <div class="stat-value">{value}</div>
                  <div class="stat-unit">{unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # 最近事件列表
    # ===============================
    st.markdown('<div class="section-title">最近事件</div>', unsafe_allow_html=True)

    np.random.seed(7)
    events = []
    types = ["拒绝服务(DoS)", "探测(Probe)", "远程登录(R2L)", "正常"]
    decisions = ["重定向至蜜罐", "无操作"]
    honeypot = ["命中", "未命中"]

    base_min = 59
    for i in range(20):
        events.append(
            {
                "时间": f"2026-01-08 10:{base_min - i:02d}",
                "事件ID": f"EVT-20260108-{1000+i}",
                "类型": np.random.choice(types),
                "置信度": round(float(np.random.uniform(0.70, 1.00)), 3),
                "决策": np.random.choice(decisions, p=[0.25, 0.75]),
                "蜜罐状态": np.random.choice(honeypot, p=[0.15, 0.85]),
            }
        )

    df_events = pd.DataFrame(events)
    # 美化表格显示
    st.dataframe(
        df_events,
        use_container_width=True,
        height=450,
        column_config={
            "置信度": st.column_config.NumberColumn(format="%.3f"),
            "决策": st.column_config.TextColumn(),
            "蜜罐状态": st.column_config.TextColumn()
        }
    )


# ===============================
# 直接执行时调用 render_home()（保持向后兼容）
# ===============================
if __name__ == "__main__":
    render_home()
