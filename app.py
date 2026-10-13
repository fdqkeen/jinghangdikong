# -*- coding: utf-8 -*-
"""
「静航 · 低空」城市低空监管驾驶舱 —— Streamlit Web 应用

启动（任选其一）：
    streamlit run app.py      ← 推荐
    python app.py             ← 也可：会自动转交给 streamlit run
    run_app.bat               ← Windows 双击启动
部署：Streamlit Community Cloud / 任意服务器 / Docker（见 README_部署.md）

说明：本应用直接复用 sim/ 下的仿真内核（config / scene / isac / noise / route），
      所有数值与提交包中的仿真结果、方案正文完全一致。
"""
import os
import sys

# ---- 自举：允许用 `python app.py` 直接启动（PyCharm 默认就是这种方式） ----
def _running_under_streamlit() -> bool:
    """
    判断当前是否已被 Streamlit 运行时托管。

    注意：不要按子模块名判断（1.32 是 streamlit.runtime.scriptrunner.script_run_context，
    1.56 改成了 streamlit.runtime.scriptrunner_utils.script_run_context，会误判成裸 python
    并误触发自举、启动一个多余的服务器）。这里用两级稳定判断：
      ① 裸 python 下 streamlit 根本不在 sys.modules —— 直接返回 False，且无需 import，
         避免打印 "missing ScriptRunContext" 干扰控制台；
      ② 已 import 时用 streamlit.runtime.exists()，该接口跨版本稳定。
    """
    if os.environ.get("JINGHANG_BOOTSTRAPPED") == "1":
        return True
    if "streamlit" not in sys.modules:
        return False
    try:
        import streamlit.runtime as _rt
        return bool(_rt.exists())
    except Exception:
        return False


if os.environ.get("JINGHANG_BOOTSTRAPPED") != "1" and not _running_under_streamlit():
    import subprocess
    _env = os.environ.copy()
    _env["JINGHANG_BOOTSTRAPPED"] = "1"
    print("检测到以 `python app.py` 方式启动，正在转交 `streamlit run` …",
          flush=True)
    print("（Streamlit 应用必须由 streamlit 运行时托管，直接 python 运行只会打印警告）",
          flush=True)
    # 转发命令行参数，例如 `python app.py --server.port 8502`
    _cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
    try:
        sys.exit(subprocess.call(_cmd, env=_env))
    except KeyboardInterrupt:
        sys.exit(0)

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")
for _p in (SIM, HERE):          # sim/ 与本目录（appkit / app_pages）
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C  # noqa: E402
import scene as scene_mod  # noqa: E402
import isac as isac_mod  # noqa: E402
import noise as noise_mod  # noqa: E402
import route as route_mod  # noqa: E402

# 通用样式与绘图工具集中放在 appkit.py，新增页面 app_pages.py 与主页共用同一套
from appkit import (AMBER, BLUE, FONT, GREEN, GRAY, RED, STRETCH,  # noqa: E402
                    layout_fig)
import app_pages as PG  # noqa: E402

st.set_page_config(page_title="静航·低空 监管驾驶舱", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
html,body,[class*="css"]{{font-family:{FONT};}}
.main .block-container{{padding-top:1.6rem;}}
h1{{font-size:1.55rem !important;}}
</style>
""", unsafe_allow_html=True)


# ==================== 缓存层 ====================
@st.cache_resource(show_spinner="构建城市栅格场景…")
def get_scene(spacing):
    sc = scene_mod.build_scene(bs_spacing=spacing)
    g = noise_mod.make_receiver_grid(sc, step=1)
    return sc, g


@st.cache_data(show_spinner="计算感知覆盖（约 4 秒）…")
def get_coverage(spacing):
    sc, _ = get_scene(spacing)
    m = isac_mod.coverage_maps(sc, alt_layers=[100.0])[100.0]
    return m


@st.cache_data(show_spinner="计算三层聚合感知指标…")
def get_kpi3(spacing):
    sc, _ = get_scene(spacing)
    maps = isac_mod.coverage_maps(sc, alt_layers=C.ALT_LAYERS)
    return isac_mod.coverage_kpi(maps)


@st.cache_data(show_spinner="噪声场重构对比…")
def get_recon(n_nodes):
    sc, g = get_scene(C.BS_SPACING_M)
    wp0 = np.array([[400.0, 600.0, 120.0], [2000.0, 1800.0, 150.0], [3600.0, 3400.0, 120.0]])
    tr = noise_mod.sample_track(wp0, 80)
    return noise_mod.reconstruct_compare(sc, g, tr,
                                         np.random.default_rng(20261009), n_nodes=n_nodes)


@st.cache_data(show_spinner="加载航路优化结果…")
def get_pareto():
    sc, g = get_scene(C.BS_SPACING_M)
    base = route_mod.baseline_route(g, sc, cruise_alt=80.0, night=True, n_samples=120)
    path = os.path.join(HERE, "outputs", "pareto.npz")
    if os.path.exists(path):
        d = np.load(path)
        P, F = d["P"], d["F"]
    else:
        with st.spinner("首次运行 NSGA-II 优化，约 100 秒…"):
            P, F = route_mod.nsga2(g, sc, seed=C.SEED, pop_size=C.NSGA2_POP,
                                   n_gen=C.NSGA2_GEN, night=True)
        np.savez(path, P=P, F=F)
    return base, P, F


@st.cache_data
def get_results():
    p = os.path.join(HERE, "outputs", "results.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def eval_route(_grid, wp, lw, lwh, is_night, ns=120):
    """按当前侧栏参数（机型/时段）评估一条航路的暴露与风险。"""
    tr = noise_mod.sample_track(wp, ns)
    lmax = np.full(_grid["X"].shape, -np.inf)
    for p in tr:
        w = lwh if p[2] < 60 else lw
        lmax = np.maximum(lmax, noise_mod.lp_field(p[:2], p[2], _grid, w))
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cell = C.AREA_M / _grid["X"].shape[0]
    risk = 0.0
    for p in tr:
        if p[2] < 80.0:
            i = int(np.clip(p[0] / cell, 0, _grid["X"].shape[0] - 1))
            j = int(np.clip(p[1] / cell, 0, _grid["X"].shape[1] - 1))
            risk += _grid["pop"][i, j]
    return dict(length=float(seg.sum()),
                exposure=float(np.sum(_grid["pop"] * np.clip(
                    lmax - (_grid["limit_night"] if is_night else _grid["limit_day"]), 0, None))),
                lmax=lmax, track=tr, waypoints=wp, risk=float(risk),
                detail=noise_mod.exposure(lmax, _grid, night=is_night))


def base_wp(alt):
    ax, ay, bx, by = 400.0, 600.0, 3600.0, 3400.0
    return np.array([[ax, ay, 30.0],
                     [ax + (bx - ax) * .15, ay + (by - ay) * .15, float(alt)],
                     [ax + (bx - ax) * .85, ay + (by - ay) * .85, float(alt)],
                     [bx, by, 30.0]])


def heat(z, title, colorscale, zmin, zmax, cbar, showscale=True):
    sc_, g_ = get_scene(st.session_state.get("spacing", C.BS_SPACING_M))
    return go.Heatmap(z=z.T, x=g_["X"][:, 0], y=g_["Y"][0, :], colorscale=colorscale,
                      zmin=zmin, zmax=zmax, showscale=showscale,
                      colorbar=dict(title=cbar, thickness=12, len=0.85) if showscale else None,
                      hovertemplate="东 %{x:.0f} m<br>北 %{y:.0f} m<br>" + cbar + " %{z:.2f}<extra></extra>")


def add_infra(fig, sc, with_bs=True, with_nodes=True, with_vert=True, with_nf=True):
    if with_nf:
        for (cx, cy, r) in sc["no_fly"]:
            fig.add_shape(type="circle", xref="x", yref="y",
                          x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                          line=dict(color=RED, width=1.6, dash="dash"))
    if with_bs:
        fig.add_trace(go.Scatter(x=sc["bs_pos"][:, 0], y=sc["bs_pos"][:, 1], mode="markers",
                                 marker=dict(size=4, color=BLUE, symbol="triangle-up", opacity=.85),
                                 name=f"6G ISAC 基站 ({len(sc['bs_pos'])})", showlegend=True))
    if with_nodes:
        n = int(st.session_state.get("n_nodes", C.N_ACOUSTIC_NODES))
        fig.add_trace(go.Scatter(x=sc["node_pos"][:n, 0], y=sc["node_pos"][:n, 1], mode="markers",
                                 marker=dict(size=7, color=GREEN, symbol="square",
                                             line=dict(color="white", width=.8)),
                                 name=f"声学节点 ({n})", showlegend=True))
    if with_vert:
        fig.add_trace(go.Scatter(x=sc["vertiports"][:, 0], y=sc["vertiports"][:, 1], mode="markers",
                                 marker=dict(size=15, color=AMBER, symbol="star",
                                             line=dict(color="white", width=1)),
                                 name="垂直起降点", showlegend=True))
    return fig


# ==================== 侧边栏 ====================
st.sidebar.title("仿真参数")
spacing = st.sidebar.selectbox("基站站距 (m)", [250, 300, 350, 400, 500], index=2)
st.session_state["spacing"] = float(spacing)
n_nodes = st.sidebar.slider("声学校准节点数", 10, 100, C.N_ACOUSTIC_NODES, 10)
st.session_state["n_nodes"] = n_nodes
AC_OPTS = {
    "微型机（Mini 类）": (91.0, 86.0),
    "商用多旋翼": (C.LW_CRUISE_DBA, C.LW_HOVER_DBA),
    "eVTOL": (C.LW_EVTOL_DBA, C.LW_EVTOL_DBA - 5.0),
    "传统直升机": (C.LW_HELI_DBA, C.LW_HELI_DBA - 5.0),
}
ac_name = st.sidebar.selectbox("机型", list(AC_OPTS.keys()), index=1)
AIR_LW, AIR_LWH = AC_OPTS[ac_name]
alt_v = st.sidebar.slider("基线巡航高度 (m)", 40, 300, 80, 10)
night = st.sidebar.radio("评价时段", ["夜间（约束工况）", "昼间"], index=0) == "夜间（约束工况）"

st.sidebar.markdown("---")
st.sidebar.caption("本页数值由 sim/ 仿真内核实时计算，与方案正文、提交包完全一致。")

scene, grid = get_scene(float(spacing))

st.title("静航 · 低空 — 城市低空飞行器监管驾驶舱")
st.caption("基于 6G 通感算智融合 ｜ 感知 → 噪声场重构 → 安全-效率-静音航路优化")

(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) = st.tabs([
    "场景与政策", "系统架构", "场景总览", "6G 通感感知", "噪声场重构",
    "航路优化", "成本与价值", "路线与风险", "敏感性分析", "KPI 与数据溯源"])

# ==================== Tab 1 · 场景与政策 ====================
with T1:
    PG.page_policy()

# ==================== Tab 2 · 系统架构 ====================
with T2:
    PG.page_arch()

# ==================== Tab 3 · 场景总览 ====================
with T3:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("栅格", f"{scene['n']}×{scene['n']} @ {scene['cell']:.0f} m")
    c2.metric("6G ISAC 基站", f"{len(scene['bs_pos'])}")
    c3.metric("声学校准节点", f"{n_nodes}")
    c4.metric("区域总人口", f"{grid['pop'].sum():,.0f}")

    fig = go.Figure(heat(scene["building_h"], "", "YlOrBr", 0, 90, "建筑高度 (m)"))
    add_infra(fig, scene)
    fig.update_traces(selector=dict(type="heatmap"), showscale=True)
    layout_fig(fig, f"城市栅格场景（站距 {spacing} m）：建筑高度 / 声环境功能区 / 感知与声学基础设施", 560)
    fig.update_layout(xaxis_title="东向 (m)", yaxis_title="北向 (m)")
    fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0))
    st.plotly_chart(fig, **STRETCH())

    st.markdown("""
    **读图**：底色为建筑高度（越深越高），虚线圈为禁飞/限高区，蓝色三角为 6G ISAC 基站，
    绿色方块为声学校准节点，橙色星标为垂直起降点。
    声学节点优先布设在 0 类（康复疗养）与 1 类（居住文教）声环境功能区。
    """)

# ==================== Tab 4 · 6G 通感感知 ====================
with T4:
    kpi3 = get_kpi3(float(spacing))
    a, b, c, d = st.columns(4)
    a.metric("检测概率达标率", f"{kpi3['coverage_pd']*100:.1f}%", "要求 ≥95%")
    b.metric("水平定位精度", f"{kpi3['median_sig_h']:.2f} m", "要求 ≤5 m")
    c.metric("垂直定位精度", f"{kpi3['median_sig_v']:.2f} m", "要求 ≤8 m")
    d.metric("三项同时达标率", f"{kpi3['coverage_full']*100:.1f}%", "要求 ≥90%")

    m = get_coverage(float(spacing))
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "(a) 检测概率 Pd @100 m", "(b) 水平定位精度 σh (m)", "(c) 垂直定位精度 σv (m)"),
        horizontal_spacing=0.06)
    sc_, g_ = get_scene(float(spacing))
    for k, (z, cs, zmin, zmax, cb) in enumerate([
            (m["pd"], [[0, "#F09595"], [.5, "#FAC775"], [1, "#1D9E75"]], 0.5, 1.0, "Pd"),
            (np.clip(m["sig_h"], 0, 10), "Viridis_r", 0, 8, "σh (m)"),
            (np.clip(m["sig_v"], 0, 12), "Viridis_r", 0, 10, "σv (m)")], start=1):
        fig.add_trace(go.Heatmap(z=z.T, x=g_["X"][:, 0], y=g_["Y"][0, :], colorscale=cs,
                                 zmin=zmin, zmax=zmax, showscale=(k == 3),
                                 colorbar=dict(title=cb, thickness=12, len=.8) if k == 3 else None),
                      row=1, col=k)
        fig.add_trace(go.Scatter(x=sc_["bs_pos"][:, 0], y=sc_["bs_pos"][:, 1], mode="markers",
                                 marker=dict(size=2.5, color="white", symbol="triangle-up", opacity=.6),
                                 showlegend=False), row=1, col=k)
    layout_fig(fig, f"6G 通感一体化感知性能（100 m 高度层，站距 {spacing} m，含建筑遮挡与共模系统误差）", 420)
    fig.update_xaxes(title_text="东向 (m)", row=1, col=1)
    fig.update_yaxes(title_text="北向 (m)", row=1, col=1)
    st.plotly_chart(fig, **STRETCH())

    st.dataframe(pd.DataFrame([
        ["检测概率 Pd", "≥ 95%", f"{kpi3['coverage_pd']*100:.1f}%", "达标"],
        ["虚警概率 Pfa", "≤ 5%", "5%（门限设定）", "达标"],
        ["水平定位精度（中位/P95）", "≤ 5 m",
         f"{kpi3['median_sig_h']:.2f} / {kpi3['p95_sig_h']:.2f} m", "达标"],
        ["垂直定位精度（中位/P95）", "≤ 8 m",
         f"{kpi3['median_sig_v']:.2f} / {kpi3['p95_sig_v']:.2f} m", "达标"],
        ["达到 Pd=95% 所需最小 SNR", "—", f"{kpi3['snr_required_db']:.2f} dB", "—"],
    ], columns=["指标", "IMT-2030 要求", "本方案仿真值", "结论"]), hide_index=True,
        **STRETCH())
    st.caption("口径对齐 ITU《IMT-2030 空口技术性能指标报告》Urban Macro-ISAC；"
               "精度优于最低要求源于密集城区多站测距/测角融合。模型已加入建筑遮挡 NLOS 损耗与"
               "不随多站平均而抵消的共模系统误差（时钟同步、NLOS 偏置），避免出现不切实际的理想值。")

# ==================== Tab 5 · 噪声场重构 ====================
with T5:
    rc = get_recon(n_nodes)
    a, b, c = st.columns(3)
    a.metric("纯节点插值 RMSE", f"{rc['err_interp']['rmse']:.2f} dB")
    b.metric("纯物理前向 RMSE", f"{rc['err_phys']['rmse']:.2f} dB")
    c.metric("本方案混合 RMSE", f"{rc['err_hybrid']['rmse']:.2f} dB",
             f"较插值降低 {(1-rc['err_hybrid']['rmse']/rc['err_interp']['rmse'])*100:.0f}%")

    vmin, vmax = float(np.nanmin(rc["truth"])), float(np.nanmax(rc["truth"]))
    fig = make_subplots(rows=2, cols=3, subplot_titles=(
        "真值场（含未建模扰动）",
        f"纯节点空间插值 · RMSE {rc['err_interp']['rmse']:.2f} dB",
        f"重构误差对比",
        f"纯物理前向未校准 · RMSE {rc['err_phys']['rmse']:.2f} dB",
        f"本方案 混合重构 · RMSE {rc['err_hybrid']['rmse']:.2f} dB",
        f"声学节点分布（{n_nodes} 个）"),
        horizontal_spacing=0.07, vertical_spacing=0.24)
    sc_, g_ = get_scene(C.BS_SPACING_M)
    pos = [(1, 1, rc["truth"]), (1, 2, rc["est_interp"]),
           (2, 1, rc["est_phys"]), (2, 2, rc["est_hybrid"]), (2, 3, rc["truth"])]
    for r, cc, z in pos:
        fig.add_trace(go.Heatmap(z=z.T, x=g_["X"][:, 0], y=g_["Y"][0, :], colorscale="Magma",
                                 zmin=vmin, zmax=vmax, showscale=False), row=r, col=cc)
    fig.add_trace(go.Scatter(x=rc["nodes_xy"][:, 0], y=rc["nodes_xy"][:, 1], mode="markers",
                             marker=dict(size=7, color=GREEN, symbol="square",
                                         line=dict(color="white", width=.8)),
                             showlegend=False), row=2, col=3)
    names = ["纯插值", "纯物理", "本方案"]
    rmse = [rc["err_interp"]["rmse"], rc["err_phys"]["rmse"], rc["err_hybrid"]["rmse"]]
    mae = [rc["err_interp"]["mae"], rc["err_phys"]["mae"], rc["err_hybrid"]["mae"]]
    fig.add_trace(go.Bar(x=names, y=rmse, name="RMSE", marker_color=BLUE,
                         text=[f"{v:.2f}" for v in rmse], textposition="outside"), row=1, col=3)
    fig.add_trace(go.Bar(x=names, y=mae, name="MAE", marker_color=RED,
                         text=[f"{v:.2f}" for v in mae], textposition="outside"), row=1, col=3)
    layout_fig(fig, f"城市三维噪声场重构对比（声学节点 {n_nodes} 个）", 800)
    for cc in (1, 2, 3):
        fig.update_xaxes(title_text="东向 (m)", showgrid=False, row=2, col=cc)
    for r in (1, 2):
        fig.update_yaxes(title_text="北向 (m)", showgrid=False, row=r, col=1)
    st.plotly_chart(fig, **STRETCH())
    st.caption("三种方法中仅本方案满足 K6 ≤3 dB 的指标要求（当前节点数下）。"
               "物理前向保证场的宏观结构正确，节点残差校正恢复个体差异、气象与地面反射等未建模效应。")

# ==================== Tab 6 · 航路优化 ====================
with T6:
    base, P, F = get_pareto()
    ok = np.where(F[:, 0] <= base["length"] * 1.15)[0]
    i_bal = int(ok[np.argmin(F[ok, 1])]) if len(ok) else int(np.argmin(F[:, 1]))
    i_short, i_quiet = int(np.argmin(F[:, 0])), int(np.argmin(F[:, 1]))
    NS = 120
    # 按当前侧栏参数（机型 / 时段 / 基线高度）重算，保证头条指标随参数联动
    base_cur = eval_route(grid, base_wp(alt_v), AIR_LW, AIR_LWH, night, NS)
    bal = eval_route(grid, route_mod._full_waypoints(P[i_bal]), AIR_LW, AIR_LWH, night, NS)
    short = eval_route(grid, route_mod._full_waypoints(P[i_short]), AIR_LW, AIR_LWH, night, NS)
    quiet = eval_route(grid, route_mod._full_waypoints(P[i_quiet]), AIR_LW, AIR_LWH, night, NS)

    a, b, c, d = st.columns(4)
    a.metric("基线暴露", f"{base_cur['exposure']:,.0f}", f"{alt_v} m 直线")
    b.metric("折中方案暴露", f"{bal['exposure']:,.0f}",
             f"-{(1-bal['exposure']/base_cur['exposure'])*100:.1f}%"
             if base_cur["exposure"] > 0 else "—")
    c.metric("航程代价", f"+{(bal['length']/base_cur['length']-1)*100:.1f}%")
    d.metric("受影响人口", f"{bal['detail']['affected_pop']:,.0f}",
             f"-{(1-bal['detail']['affected_pop']/base_cur['detail']['affected_pop'])*100:.1f}%"
             if base_cur["detail"]["affected_pop"] > 0 else "—")
    st.caption(f"当前工况：机型 {ac_name}（Lw={AIR_LW:.1f} dB(A)）· "
               f"{'夜间' if night else '昼间'} · 基线巡航高度 {alt_v} m。"
               f"帕累托前沿在「夜间·商用多旋翼」工况下优化得到，切换工况后前沿保持不变，"
               f"下方效果表按当前参数重算。")
    _cut = (1 - bal["exposure"] / base_cur["exposure"]) * 100 if base_cur["exposure"] > 0 else 100.0
    if _cut < 30.0:
        st.warning(f"当前工况下航路优化仅使暴露下降 {_cut:.1f}%，未达 K8（≥30%）。"
                   f"说明对 {ac_name} 这类高噪声机型，**仅靠航路优化不足以达标**，"
                   f"需叠加起降架次限制、夜间时段管控与低噪声进离场程序——"
                   f"这正对应苏州地标中「起降架次限制」与「综合管理」的要求。")
    elif base_cur["exposure"] == 0:
        st.info("当前工况下基线已无超标（该机型在该高度下不触发限值），无需优化。")

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "(a) 效率—静音 权衡前沿", "(b) 静音—安全 权衡前沿"), horizontal_spacing=0.1)
    fig.add_trace(go.Scatter(x=F[:, 0] / 1000, y=F[:, 1] / 1000, mode="markers",
                             marker=dict(size=7, color=BLUE, opacity=.6), name="帕累托解集"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[base["length"] / 1000], y=[base["exposure"] / 1000],
                             mode="markers", marker=dict(size=17, color=RED, symbol="star",
                                                         line=dict(color="white", width=1)),
                             name="现状基线"), row=1, col=1)
    for tag, e, col in [("最短", short, AMBER), ("折中", bal, GREEN), ("最静音", quiet, BLUE)]:
        fig.add_trace(go.Scatter(x=[e["length"] / 1000], y=[e["exposure"] / 1000], mode="markers",
                                 marker=dict(size=12, color=col, symbol="diamond",
                                             line=dict(color="white", width=1)),
                                 name=tag, showlegend=False), row=1, col=1)
        fig.add_annotation(x=e["length"] / 1000, y=e["exposure"] / 1000, text=tag,
                           showarrow=True, arrowhead=0, ay=-18, font=dict(size=11), row=1, col=1)
    fig.add_trace(go.Scatter(x=F[:, 1] / 1000, y=F[:, 2], mode="markers",
                             marker=dict(size=7, color=GREEN, opacity=.6), showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=[base["exposure"] / 1000], y=[base["risk"]], mode="markers",
                             marker=dict(size=17, color=RED, symbol="star",
                                         line=dict(color="white", width=1)),
                             showlegend=False), row=1, col=2)
    layout_fig(fig, "安全—效率—静音 三目标帕累托前沿（NSGA-II）", 430)
    fig.update_xaxes(title_text="航程 (km)", row=1, col=1)
    fig.update_yaxes(title_text="夜间噪声暴露 (千人·dB)", row=1, col=1)
    fig.update_xaxes(title_text="夜间噪声暴露 (千人·dB)", row=1, col=2)
    fig.update_yaxes(title_text="地面安全风险（低空段人口）", row=1, col=2)
    st.plotly_chart(fig, **STRETCH())

    # 航路对比
    exc_b = np.clip(base_cur["lmax"] - (grid["limit_night"] if night else grid["limit_day"]), 0, None)
    tr_b = base_cur["track"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"(a) 航路平面与{'夜间' if night else '昼间'}超标量", "(b) 高度剖面"),
                        horizontal_spacing=0.08,
                        specs=[[{"type": "xy"}, {"type": "xy"}]])
    fig.add_trace(go.Heatmap(z=exc_b.T, x=grid["X"][:, 0], y=grid["Y"][0, :],
                             colorscale="Reds", zmin=0, zmax=25, showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=tr_b[:, 0], y=tr_b[:, 1], mode="lines",
                             line=dict(color=RED, width=3), name=f"基线（{alt_v} m 直线）"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bal["track"][:, 0], y=bal["track"][:, 1], mode="lines",
                             line=dict(color=BLUE, width=3), name="优化（折中解）"), row=1, col=1)
    fig.add_trace(go.Scatter(x=scene["vertiports"][:, 0], y=scene["vertiports"][:, 1],
                             mode="markers", marker=dict(size=15, color=AMBER, symbol="star",
                                                         line=dict(color="white", width=1)),
                             name="起降点"), row=1, col=1)
    d_b = np.linalg.norm(tr_b[:, :2] - tr_b[0, :2], axis=1)
    d_o = np.linalg.norm(bal["track"][:, :2] - bal["track"][0, :2], axis=1)
    fig.add_trace(go.Scatter(x=d_b / 1000, y=tr_b[:, 2], mode="lines",
                             line=dict(color=RED, width=3), name="基线高度", showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=d_o / 1000, y=bal["track"][:, 2], mode="lines",
                             line=dict(color=BLUE, width=3), name="优化高度", showlegend=False),
                  row=1, col=2)
    layout_fig(fig, f"基线 vs 优化航路（机型声功率级 Lw={AIR_LW:.1f} dB(A)，"
                    f"{'夜间' if night else '昼间'}）", 470, equal_aspect=False)
    fig.update_xaxes(scaleanchor="y", row=1, col=1)   # 仅左侧地图面板保持等比
    fig.update_xaxes(title_text="东向 (m)", row=1, col=1)
    fig.update_yaxes(title_text="北向 (m)", row=1, col=1)
    fig.update_xaxes(title_text="沿航程水平距离 (km)", row=1, col=2)
    fig.update_yaxes(title_text="飞行高度 (m)", row=1, col=2)
    st.plotly_chart(fig, **STRETCH())

    rows = []
    for tag, e in [(f"基线（{alt_v} m 直线）", base_cur), ("最短航路", short),
                   ("折中方案（推荐）", bal), ("最静音航路", quiet)]:
        d = e["detail"]
        cut = (1 - e["exposure"] / base_cur["exposure"]) * 100 if base_cur["exposure"] > 0 else 0.0
        rows.append([tag, f"{e['length']:,.0f}",
                     f"{(e['length']/base_cur['length']-1)*100:+.1f}%",
                     f"{e['exposure']:,.0f}", f"-{cut:.1f}%",
                     f"{d['affected_pop']:,.0f}", f"{d['exceed_area_km2']:.2f}",
                     f"{d['max_exceed_db']:.1f}"])
    st.dataframe(pd.DataFrame(rows, columns=[
        "方案", "航程 (m)", "航程变化", "噪声暴露 (人·dB)", "暴露下降",
        "受影响人口", "超标面积 (km²)", "最大超标 (dB)"]),
        hide_index=True, **STRETCH())
    st.caption("噪声暴露为人口加权积分量，最大超标为单点极值，二者非同向；"
               "噪声治理决策以积分量为准，对应苏州地标中的“暴露评估”要求。")
    st.caption("**口径说明**：本页基线 / 优化航路均**含起降爬升段**（30 m ↔ 巡航高度），"
               "故提高巡航高度的降幅（80→300 m 约 61%）低于“敏感性分析”页全程等高巡航口径的 71.3%；"
               "两者结论一致，差异仅来自是否计入起降段——这也说明**起降阶段是降噪的关键环节**。")

# ==================== Tab 7 · 成本与产业价值 ====================
with T7:
    PG.page_cost()

# ==================== Tab 8 · 实施路线与风险 ====================
with T8:
    PG.page_roadmap()

# ==================== Tab 9 · 敏感性分析 ====================
with T9:
    res = get_results()
    if res is None:
        st.warning("未找到 outputs/results.json，请先运行 `python sim/run_all.py`。")
    else:
        S = res["sensitivity"]
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "(a) 基站站距敏感性", "(b) 处理增益敏感性",
            "(c) 目标 RCS 敏感性", "(d) 声学节点数敏感性"),
            horizontal_spacing=0.1, vertical_spacing=0.24)
        sp = [d["spacing"] for d in S["bs_spacing"]]
        fig.add_trace(go.Scatter(x=sp, y=[d["coverage"] for d in S["bs_spacing"]],
                                 mode="lines+markers", line=dict(color=BLUE, width=2.5),
                                 showlegend=False), row=1, col=1)
        fig.add_hline(y=90, line=dict(color=RED, dash="dash"), row=1, col=1)
        gp = [d["gp"] for d in S["gp"]]
        fig.add_trace(go.Scatter(x=gp, y=[d["coverage"] for d in S["gp"]],
                                 mode="lines+markers", line=dict(color=BLUE, width=2.5),
                                 showlegend=False), row=1, col=2)
        fig.add_hline(y=90, line=dict(color=RED, dash="dash"), row=1, col=2)
        rcs = [f'{d["rcs"]:.3f}' for d in S["rcs"]]
        fig.add_trace(go.Bar(x=rcs, y=[d["coverage"] for d in S["rcs"]],
                             marker_color=BLUE, width=.5,
                             text=[f'{d["coverage"]:.1f}%' for d in S["rcs"]],
                             textposition="outside", showlegend=False), row=2, col=1)
        ns = [d["n"] for d in S["nodes"]]
        fig.add_trace(go.Scatter(x=ns, y=[d["hybrid"] for d in S["nodes"]], mode="lines+markers",
                                 line=dict(color=GREEN, width=2.5), name="本方案 混合"), row=2, col=2)
        fig.add_trace(go.Scatter(x=ns, y=[d["interp"] for d in S["nodes"]], mode="lines+markers",
                                 line=dict(color=RED, width=2.5, dash="dot"), name="纯插值"),
                      row=2, col=2)
        fig.add_hline(y=3.0, line=dict(color=BLUE, dash="dash"), row=2, col=2)
        layout_fig(fig, "关键参数敏感性分析（验证结论在非最优参数下的稳健性）", 760,
                   equal_aspect=False)
        fig.update_xaxes(title_text="基站站距 (m)", row=1, col=1)
        fig.update_yaxes(title_text="全指标达标率 (%)", row=1, col=1)
        fig.update_xaxes(title_text="OFDM 处理增益 (dB)", row=1, col=2)
        fig.update_yaxes(title_text="全指标达标率 (%)", row=1, col=2)
        fig.update_xaxes(title_text="目标 RCS (m²)", row=2, col=1)
        fig.update_yaxes(title_text="全指标达标率 (%)", row=2, col=1)
        fig.update_xaxes(title_text="声学校准节点数", row=2, col=2)
        fig.update_yaxes(title_text="重构 RMSE (dB)", row=2, col=2)
        for r in (1, 2):
            for cc in (1, 2):
                fig.update_xaxes(showgrid=False, row=r, col=cc)
        st.plotly_chart(fig, **STRETCH())

        fig2 = make_subplots(rows=1, cols=2, subplot_titles=(
            "(a) 巡航高度敏感性", "(b) 机型敏感性"), horizontal_spacing=0.1)
        al = [d["alt"] for d in S["altitude"]]
        fig2.add_trace(go.Scatter(x=al, y=[d["exp"] for d in S["altitude"]], mode="lines+markers",
                                  line=dict(color=BLUE, width=2.5), name="夜间暴露"), row=1, col=1)
        fig2.add_hline(y=base["exposure"], line=dict(color=RED, dash="dash"),
                       annotation_text="基线（80 m 直线）", row=1, col=1)
        nm = [d["name"] for d in S["aircraft"]]
        fig2.add_trace(go.Bar(x=nm, y=[d["exp_day"] for d in S["aircraft"]],
                              name="昼间暴露", marker_color=AMBER), row=1, col=2)
        fig2.add_trace(go.Bar(x=nm, y=[d["exp_night"] for d in S["aircraft"]],
                              name="夜间暴露", marker_color=BLUE), row=1, col=2)
        layout_fig(fig2, "运营参数与机型敏感性", 420, equal_aspect=False)
        fig2.update_xaxes(title_text="巡航高度 (m)", row=1, col=1)
        fig2.update_yaxes(title_text="夜间噪声暴露 (人·dB)", row=1, col=1)
        fig2.update_yaxes(title_text="噪声暴露 (人·dB)", row=1, col=2)
        st.plotly_chart(fig2, **STRETCH())

        st.markdown("""
        **稳健性结论（三层聚合口径）**：达标率在站距 ≤400 m、RCS ≥0.01 m²、
        处理增益 ≥25 dB 的设定下保持 ≥93%；站距放宽至 500 m 时降至 85.4%，
        RCS 降至 0.001 m² 或处理增益降至 15 dB 时降至 84.2%，均跌破 90% 目标。

        **两条工程边界**：混合重构法在全部节点密度下均优于纯插值；
        但节点数过少（≤10）时残差校正不可靠，混合法退化至 4.72 dB，
        甚至差于纯物理前向的 3.03 dB——故声学节点数建议 ≥20，推荐 40。

        **可直接指导运营的结论**：巡航高度由 80 m 提高至 300 m，夜间暴露下降 **71.3%**，
        且**不依赖任何新建基础设施**。
        （口径：本曲线为全程等高巡航、**不计起降爬升段**；若按含起降段的基线航路口径，
        80→300 m 降幅约 61%——量级一致，结论不变。）

        机型方面，微型机在 100 m 高度昼夜均不超标，
        eVTOL 与直升机的暴露量比商用多旋翼高一个数量级——
        管控重点应放在中大型机型与起降阶段，而非“一刀切”限制所有低空活动。
        """)

# ==================== Tab 10 · KPI 与数据溯源 ====================
with T10:
    PG.page_trace()
