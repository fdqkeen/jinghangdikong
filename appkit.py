# -*- coding: utf-8 -*-
"""
app.py 与 app_pages.py 共用的样式常量与绘图工具。

单独成文件的原因：新增页面（app_pages.py）与主页都要用到同一套字体、
配色与图幅规则，避免两处重复定义导致样式不一致。
"""
import streamlit as st
from plotly.subplots import make_subplots  # noqa: F401  （供页面按需使用）

FONT = "Microsoft YaHei, Noto Sans CJK SC, PingFang SC, SimHei, sans-serif"
BLUE, RED, GREEN, AMBER, GRAY = "#185FA5", "#D85A30", "#1D9E75", "#BA7517", "#888780"
PURPLE, TEAL = "#6A5ACD", "#2A9D8F"


def _st_ver():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


def STRETCH():
    """
    宽度参数兼容层。

    Streamlit ≥1.46 用 width="stretch"；旧版用 use_container_width=True。
    官方已公告 use_container_width 于 2025-12-31 后移除，故按版本自动选择。
    用法：st.dataframe(df, **STRETCH()) / st.plotly_chart(fig, **STRETCH())
    """
    try:
        from packaging import version
        return ({"width": "stretch"}
                if version.parse(st.__version__) >= version.parse("1.46")
                else {"use_container_width": True})
    except Exception:
        return {"width": "stretch"} if _st_ver() >= (1, 46) else {"use_container_width": True}


def IMAGE_W():
    """
    图片宽度参数兼容层——**必须与 STRETCH() 分开**。

    st.image 与 st.dataframe / st.plotly_chart 的参数演进不同步，历史上先后用过：
      ① use_column_width=True       （≤1.34，如 anaconda 自带的 streamlit 1.32）
      ② use_container_width=True    （1.35 – 1.45）
      ③ width="stretch"             （≥1.46，前两个已移除）
    版本号推断容易踩坑（不同发行版打包的版本跨度很大），
    故这里**直接读函数签名**来判断该环境认哪个参数，最稳。
    """
    import inspect
    try:
        params = inspect.signature(st.image).parameters
    except (TypeError, ValueError):
        return {}
    if "use_container_width" in params:
        return {"use_container_width": True}
    if "use_column_width" in params:
        return {"use_column_width": True}
    if "width" in params and _st_ver() >= (1, 46):
        return {"width": "stretch"}
    return {}


def layout_fig(fig, title, h=430, equal_aspect=True):
    """
    统一图幅样式。

    顶部边距按是否存在子图标题动态分配——主标题锚定画布最顶端，
    子图标题占用余下边距，避免主标题与子标题挤在一起。
    equal_aspect=True 仅用于地图类热力图（保持栅格等比）；
    折线/柱状图须传 False，否则 scaleanchor 会把图压扁。
    """
    has_sub = any(getattr(a, "text", None) for a in fig.layout.annotations)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0.0, xanchor="left",
                   y=0.995, yanchor="top", yref="container"),
        font=dict(family=FONT, size=12), height=h,
        margin=dict(l=10, r=10, t=78 if has_sub else 46, b=10),
        title_pad=dict(t=2),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    if equal_aspect:
        fig.update_xaxes(showgrid=False, scaleanchor="y")
    else:
        fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # 让边距随轴标题与刻度标签自动撑开，避免被裁切
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def table(df, **kw):
    """统一表格样式：隐藏索引 + 自适应宽度。"""
    return st.dataframe(df, hide_index=True, **STRETCH(), **kw)


def note(text):
    """统一脚注样式。"""
    st.caption(text)


def card(title, body, color=RED, bg="#FDF1EE"):
    """
    轻量提示卡片。

    不用 st.error / st.warning 的原因：它们是「告警元素」，会被 AppTest 计入
    error/warning 数量，掩盖真正的运行异常。这里改用纯样式 div，
    视觉一致且不干扰回归测试的异常计数。
    """
    st.markdown(
        f'<div style="border-left:4px solid {color};background:{bg};'
        f'padding:.7rem .9rem;border-radius:6px;margin-bottom:.2rem;">'
        f'<div style="font-weight:700;color:{color};margin-bottom:.35rem;">{title}</div>'
        f'<div style="color:#333;line-height:1.6;">{body}</div></div>',
        unsafe_allow_html=True)
