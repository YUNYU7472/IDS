import streamlit as st
import pandas as pd
import json
import time
import os
import sys
from datetime import datetime

# ===============================
# 1. 动态导入后端模块
# ===============================
try:
    import analysis_agent
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    try:
        import analysis_agent
    except ImportError:
        st.error("❌ 无法导入 analysis_agent.py，请确保该文件在 IDS_pipeline 根目录下。")

# ===============================
# 2. 真实数据读取函数
# ===============================
def _read_real_logs(filename="honeypot_hits.jsonl", max_rows=100):
    cwd = os.getcwd()
    filepath = os.path.join(cwd, filename)
    
    if not os.path.exists(filepath):
        return []
    
    logs = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in reversed(lines):
            if len(logs) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "ts_ms" not in data:
                    data["ts_ms"] = int(time.time() * 1000)
                
                if "preview" not in data:
                    if "honeypot_observed" in data:
                        data["preview"] = data["honeypot_observed"].get("preview", "N/A")
                        if "client_ip" not in data:
                            data["client_ip"] = data["honeypot_observed"].get("client_ip", "unknown")
                    else:
                        data["preview"] = "N/A"
                
                logs.append(data)
            except json.JSONDecodeError:
                continue
        return logs
    except Exception as e:
        st.error(f"读取日志文件失败: {e}")
        return []

# ===============================
# 3. 样式与导航
# ===============================
def _load_css():
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {display: none;}
        div[data-testid="stVerticalBlock"] > div {background: transparent !important; box-shadow: none !important;}
        * { font-family: "Microsoft YaHei", "PingFang SC", sans-serif !important; }
        .block-container {padding-top: 0.8rem; padding-bottom: 2rem;}
        
        /* 导航栏 */
        .ghids-nav {
          width: 100%; background: #0B5ED7; border-radius: 12px; padding: 12px 20px;
          display:flex; align-items:center; justify-content:space-between;
          box-shadow: 0 8px 24px rgba(11,94,215,0.2); margin-bottom: 20px;
        }
        .ghids-brand { color: #ffffff; font-weight: 700; font-size: 24px; }
        .ghids-nav-right{ display:flex; gap: 16px; }
        .ghids-nav-btn{
          color: rgba(255,255,255,0.95); padding: 10px 16px; border-radius: 10px;
          font-weight: 600; text-decoration: none; transition: all 0.2s ease;
        }
        .ghids-nav-btn:hover{ background: rgba(255,255,255,0.2); }

        /* 标题与卡片 */
        .section-title{ font-size: 20px; font-weight: 700; color: #0F172A; margin: 24px 0 12px 0; display: flex; align-items: center; gap: 8px; }
        .section-title::before { content: ""; width: 4px; height: 22px; background: #0B5ED7; border-radius: 2px; }
        .stat-card { padding: 16px; background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        .stat-val { font-size: 24px; font-weight: 700; color: #0B5ED7; }
        .stat-lbl { font-size: 13px; color: #64748b; }
        </style>
        """, unsafe_allow_html=True
    )

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
        """, unsafe_allow_html=True
    )

def _ensure_state():
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []

# ===============================
# 4. 主渲染逻辑
# ===============================
def render_honeypot():
    st.set_page_config(page_title="蜜罐重定向监控", layout="wide", initial_sidebar_state="collapsed")
    _load_css()
    _ensure_state()
    _nav_bar()

    # --- 顶部控制栏 ---
    c_title, c_refresh = st.columns([8, 1])
    with c_title:
        st.markdown('<div class="section-title" style="margin-top:0;">蜜罐捕获概览</div>', unsafe_allow_html=True)
    with c_refresh:
        if st.button("🔄 刷新数据", use_container_width=True):
            st.rerun()

    logs = _read_real_logs("honeypot_hits.jsonl")

    # --- KPI ---
    total_hits = len(logs)
    beacons = sum(1 for l in logs if l.get("record_type") == "beacon")
    if logs:
        unique_ips = len(set(l.get("client_ip", "unknown") for l in logs))
        last_ts = logs[0].get("ts_ms", 0)
        last_hit = datetime.fromtimestamp(last_ts/1000).strftime("%H:%M:%S")
    else:
        unique_ips = 0
        last_hit = "--"

    k1, k2, k3, k4 = st.columns(4)
    def _kpi(col, lbl, val):
        col.markdown(f"""<div class="stat-card"><div class="stat-lbl">{lbl}</div><div class="stat-val">{val}</div></div>""", unsafe_allow_html=True)
    
    _kpi(k1, "当前显示日志数", total_hits)
    _kpi(k2, "IDS联动信标", beacons)
    _kpi(k3, "攻击源数量", unique_ips)
    _kpi(k4, "最新捕获时间", last_hit)

    # --- 表格 ---
    st.markdown('<div class="section-title">攻击流量日志</div>', unsafe_allow_html=True)
    if logs:
        df = pd.DataFrame(logs)
        if "ts_ms" in df.columns:
            df["Time"] = df["ts_ms"].apply(lambda x: datetime.fromtimestamp(x/1000).strftime("%Y-%m-%d %H:%M:%S"))
        else:
            df["Time"] = "--"
            
        def _clean_preview(text):
            if not isinstance(text, str): return str(text)
            return text.replace("\r\n", " ↵ ").replace("\n", " ↵ ")

        df["Payload_View"] = df["preview"].apply(_clean_preview)

        display_cols = ["Time", "client_ip", "record_type", "Payload_View", "bytes_received"]
        for c in display_cols:
            if c not in df.columns:
                df[c] = "N/A"
        
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            height=280,
            column_config={
                "Payload_View": st.column_config.TextColumn("Payload Preview (Cleaned)", width="large"),
                "record_type": st.column_config.Column("Type", width="small"),
                "bytes_received": st.column_config.NumberColumn("Bytes", format="%d"),
            }
        )

        # 【修改点1】使用 Toggle 替代 Expander，避免出现 keyboard_arrow 图标文字
        if st.toggle("查看原始 JSON 日志"):
            st.json(logs[:3])
    else:
        st.info("暂无蜜罐日志数据。")

    # --- AI 控制台 ---
    st.markdown('<div class="section-title">AI 威胁分析控制台</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.5])

    with c1:
        default_key = os.environ.get("DEEPSEEK_API_KEY", "")
        api_key = st.text_input("DeepSeek API Key", value=default_key, type="password", placeholder="sk-...")
        
        if st.button("🚀 立即分析", use_container_width=True, type="primary"):
            if not logs:
                st.warning("无日志数据，无法分析。")
            elif not api_key:
                st.error("请输入 API Key。")
            else:
                batch_logs = logs[:20] 
                
                minified_logs = []
                for log in batch_logs:
                    raw_payload = log.get("preview", "")
                    if not raw_payload and "honeypot_observed" in log:
                         raw_payload = log["honeypot_observed"].get("preview", "")
                    
                    final_payload = raw_payload[:500] 

                    simplified = {
                        "time": log.get("ts_ms"),
                        "src": f"{log.get('client_ip')}",
                        "type": log.get("record_type", "unknown"),
                        "payload": final_payload 
                    }
                    minified_logs.append(simplified)
                
                log_snippet = json.dumps(minified_logs, indent=2, ensure_ascii=False)
                
                st.session_state["ai_chat_history"].append({
                    "role": "agent",
                    "content": f"正在分析最近 {len(batch_logs)} 条攻击日志...",
                    "code": f"Prompt数据预览：\n{log_snippet}"
                })
                
                with st.spinner("DeepSeek 正在思考中..."):
                    try:
                        report = analysis_agent.call_deepseek_api(
                            logs=batch_logs, 
                            api_key=api_key, 
                            api_url="https://api.deepseek.com/chat/completions",
                            model="deepseek-chat"
                        )
                        st.session_state["ai_chat_history"].append({
                            "role": "ai",
                            "content": report
                        })
                    except Exception as e:
                        st.error(f"分析失败: {e}")
                        st.session_state["ai_chat_history"].append({
                            "role": "ai",
                            "content": f"❌ 分析失败: {str(e)}"
                        })
                st.rerun()

        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state["ai_chat_history"] = []
            st.rerun()

    with c2:
        history = st.session_state["ai_chat_history"]
        if not history:
            st.info("等待指令... 请在左侧输入 API Key 并点击“立即触发分析”。")
        
        for idx, msg in enumerate(history):
            if msg["role"] == "agent":
                with st.chat_message("user", avatar="👨‍💻"):
                    st.write(msg["content"])
                    if "code" in msg:
                        # 【修改点2】使用 Checkbox 替代 Expander，解决离线图标问题
                        # 使用 idx 确保每个 checkbox 的 key 唯一
                        if st.checkbox("显示 Prompt 详情 (JSON)", key=f"show_prompt_{idx}"):
                            st.code(msg["code"], language="json")
            elif msg["role"] == "ai":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])

if __name__ == "__main__":
    render_honeypot()