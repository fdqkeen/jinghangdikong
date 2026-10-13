# -*- coding: utf-8 -*-
"""
无人机噪声传播、地面暴露与噪声场重构
- 声源：由 GB 42590-2023 标识值反推声功率级 Lw
- 传播：ISO 9613-2 简化（几何发散 + 空气吸收 + 建筑屏障 Maekawa）
- 重构：物理前向场 + 稀疏声学节点残差校正（对比纯插值 / 纯物理）
"""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.interpolate import RBFInterpolator
from config import (ALPHA_DB_PER_KM, ABAR_MAX_DB, CELL_M, LW_HOVER_DBA, LW_CRUISE_DBA,
                    REF_DIST_M, NOISE_MEAS_SIGMA_DB)

RECV_Z = 1.5            # 接收点（人耳）高度 m
SCREEN_OFFSET_M = 25.0  # 接收点邻域遮挡判据的回溯距离 m
LAMBDA_SOUND_M = 0.686  # 500 Hz 代表波长 (343/500) m


def make_receiver_grid(scene, step=1):
    """构建接收栅格（step>1 时抽稀，用于优化阶段加速）。"""
    sl = (slice(None, None, step), slice(None, None, step))
    bh = scene["building_h"]
    hs_screen = maximum_filter(bh, size=3)[sl]
    g = dict(
        X=scene["X"][sl], Y=scene["Y"][sl],
        Hs=hs_screen,
        pop=scene["pop"][sl],
        limit_day=scene["zone_limit_day"][sl],
        limit_night=scene["zone_limit_night"][sl],
        zone=scene["zone"][sl],
        building_h=bh[sl],
        step=step,
    )
    g["cell_area_km2"] = (CELL_M * step) ** 2 / 1e6
    return g


def lp_field(src_xy, hs, grid, lw, alpha_db_km=ALPHA_DB_PER_KM, shield=False):
    """
    单个声源位置在接收栅格上的 A 计权声压级 (dB)。
    src_xy: (2,) 声源水平坐标; hs: 声源高度 m

    shield=False（默认，主结果）：仅几何发散 + 空气吸收，即 ISO 9613-2 的 Adiv + Aatm。
        这是保守上界（不低估暴露），且随高度单调下降，与"快速爬升降噪"的公认结论一致。
    shield=True：叠加建筑屏蔽二阶修正，用于敏感性分析。
    """
    d_h = np.hypot(grid["X"] - src_xy[0], grid["Y"] - src_xy[1])
    dz = max(float(hs) - RECV_Z, 1.0)
    R = np.sqrt(d_h ** 2 + dz ** 2)

    a_div = 20.0 * np.log10(R) + 11.0
    a_atm = alpha_db_km * R / 1000.0
    a_bar = np.zeros_like(R)

    if shield:
        # 建筑屏蔽二阶修正：仰角判据 + Maekawa 菲涅尔数
        # 仅当射线足够平缓（水平距离超出阈值）时才可能遮挡，避免近头顶误判
        h_scr = grid["Hs"]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = h_scr - RECV_Z - dz * SCREEN_OFFSET_M / np.maximum(d_h, 1e-6)
        delta = np.clip(delta, 0.0, None)
        delta = np.where(d_h >= np.minimum(60.0, 0.5 * d_h.max()), delta, 0.0)
        fresnel = 2.0 * delta / LAMBDA_SOUND_M
        a_bar = np.minimum(ABAR_MAX_DB, 10.0 * np.log10(3.0 + 20.0 * fresnel))
        a_bar = np.where(delta <= 0.0, 0.0, a_bar)

    return lw - a_div - a_atm - a_bar


def sample_track(waypoints, n_samples):
    """把折线航路按弧长等距采样为 (T,3)。waypoints: (K,3)"""
    wp = np.asarray(waypoints, dtype=float)
    seg_len = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    total = seg_len.sum()
    if total <= 0:
        return np.repeat(wp[:1], n_samples, axis=0)
    target = np.linspace(0.0, total, n_samples)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = np.zeros((n_samples, 3))
    for k in range(3):
        out[:, k] = np.interp(target, cum, wp[:, k])
    return out


def event_field(track, grid, lw_hover=LW_HOVER_DBA, lw_cruise=LW_CRUISE_DBA,
                climb_m=40.0):
    """
    一次飞行事件在地面产生的最大 A 声级 L_Amax 场。
    起降段（首尾）用悬停声功率级，巡航段用巡航声功率级。
    """
    lmax = np.full(grid["X"].shape, -np.inf)
    alt = track[:, 2]
    for k, p in enumerate(track):
        lw = lw_hover if (k < 2 or k > len(track) - 3 or p[2] < climb_m) else lw_cruise
        f = lp_field(p[:2], p[2], grid, lw)
        lmax = np.maximum(lmax, f)
    return lmax


def exposure(lmax, grid, night=False):
    """
    地面噪声暴露指标。
    返回 dict: 超标人·dB 总量、受影响人口、超标面积(km^2)、最大超标量
    """
    limit = grid["limit_night"] if night else grid["limit_day"]
    exc = np.clip(lmax - limit, 0.0, None)
    area = grid["cell_area_km2"]
    return dict(
        exposure_person_db=float(np.sum(grid["pop"] * exc)),
        affected_pop=float(np.sum(grid["pop"] * (exc > 0))),
        exceed_area_km2=float(np.sum((exc > 0) * area)),
        max_exceed_db=float(np.max(exc)),
        mean_exceed_db=float(np.mean(exc)),
    )


# ---------------- 噪声场重构对比实验 ----------------

def smooth_random_field(shape, sigma_db, corr_cells, rng):
    """空间相关的未建模扰动场（声源个体差异、气象、地面反射异质性）。"""
    from scipy.ndimage import gaussian_filter
    raw = rng.normal(0.0, 1.0, shape)
    f = gaussian_filter(raw, sigma=corr_cells)
    f = f / (np.std(f) + 1e-9) * sigma_db
    return f


def reconstruct_compare(scene, grid, track, rng,
                        sigma_pert_db=3.0, corr_cells=6.0,
                        n_nodes=None):
    """
    对比三种噪声场重构方式对真值场的还原误差。
    真值 = 物理前向场 + 空间相关扰动（含未建模效应）
    """
    phys = event_field(track, grid, LW_CRUISE_DBA, LW_CRUISE_DBA)   # 标称物理场
    pert = smooth_random_field(grid["X"].shape, sigma_pert_db, corr_cells, rng)
    truth = phys + pert

    node_cells = scene["node_cells"]
    if n_nodes is None:
        from config import N_ACOUSTIC_NODES as _N
        node_cells = node_cells[:_N]
    else:
        node_cells = node_cells[:n_nodes]
    nx = node_cells[:, 0] // grid["step"]
    ny = node_cells[:, 1] // grid["step"]
    nx = np.clip(nx, 0, grid["X"].shape[0] - 1)
    ny = np.clip(ny, 0, grid["X"].shape[1] - 1)

    obs = truth[nx, ny] + rng.normal(0.0, NOISE_MEAS_SIGMA_DB, size=len(nx))
    node_xy = np.column_stack([grid["X"][nx, ny], grid["Y"][nx, ny]])

    # 方案 A：现有方案 —— 纯空间插值（仅用声学节点）
    target_xy = np.column_stack([grid["X"].ravel(), grid["Y"].ravel()])
    rbf = RBFInterpolator(node_xy, obs, kernel="thin_plate_spline", smoothing=1.0)
    est_interp = rbf(target_xy).reshape(grid["X"].shape)

    # 方案 B：纯物理前向（无节点校准）
    est_phys = phys.copy()

    # 方案 C：本方案 —— 物理前向 + 节点残差空间校正
    resid = obs - phys[nx, ny]
    rbf_r = RBFInterpolator(node_xy, resid, kernel="thin_plate_spline", smoothing=1.0)
    est_hybrid = phys + rbf_r(target_xy).reshape(grid["X"].shape)

    def _err(e):
        d = e - truth
        return dict(rmse=float(np.sqrt(np.mean(d ** 2))),
                    mae=float(np.mean(np.abs(d))),
                    p95=float(np.percentile(np.abs(d), 95)),
                    max=float(np.max(np.abs(d))))
    return dict(truth=truth, phys=phys, nodes_xy=node_xy, obs=obs,
                est_interp=est_interp, est_phys=est_phys, est_hybrid=est_hybrid,
                err_interp=_err(est_interp), err_phys=_err(est_phys),
                err_hybrid=_err(est_hybrid))
