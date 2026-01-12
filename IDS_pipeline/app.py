import streamlit as st
from pages.dashboard_index import render_home
from pages.dashboard_realtimeids import render_realtime
from pages.dashboard_evidence import render_evidence


# ===============================
# 路由：读取 query param，决定当前页面
# ===============================
def get_current_page() -> str:
    """获取当前页面参数"""
    page = "home"
    try:
        if hasattr(st, "query_params"):
            params = st.query_params
            page_val = params.get("page", "home")
            if isinstance(page_val, list):
                page = page_val[0] if page_val else "home"
            else:
                page = page_val or "home"
        else:
            params = st.experimental_get_query_params()
            page = params.get("page", ["home"])[0]
    except Exception:
        page = "home"
    
    if page not in ("home", "realtime", "evidence", "honeypot"):
        page = "home"
    return page


# ===============================
# 占位页面渲染函数
# ===============================
def render_placeholder(page_name: str, title: str):
    """渲染占位页面"""
    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="collapsed")
    
    # 隐藏 sidebar 的 CSS
    st.markdown(
        """
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    div[data-testid="collapsedControl"] { display: none !important; }
    </style>
        """,
        unsafe_allow_html=True,
    )
    
    # 顶栏 NAV BAR（与首页一致）
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
    
    st.title(f"{title}（开发中）")
    st.info(f"页面 {page_name} 正在开发中，敬请期待...")


# ===============================
# 主入口：根据 page 路由
# ===============================
if __name__ == "__main__":
    _page = get_current_page()
    
    if _page == "home":
        render_home()
    elif _page == "realtime":
        render_realtime()
    elif _page == "evidence":
        render_evidence()
    elif _page == "honeypot":
        render_placeholder("honeypot", "蜜罐重定向")
    else:
        render_home()
