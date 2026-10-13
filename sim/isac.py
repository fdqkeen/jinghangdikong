# -*- coding: utf-8 -*-
"""
6G 通感一体化（ISAC）感知模型
- 雷达方程 → 链路 SNR
- Swerling I 单脉冲检测：Pd = Pfa^(1/(1+SNR))
- 多站测距/测角融合 → Fisher 信息矩阵 → CRLB 定位精度
指标口径对齐 ITU《IMT-2030 空口技术性能指标报告》Urban Macro-ISAC：
Pd>=95%, Pfa<=5%, 水平<=5 m, 垂直<=8 m
"""
import numpy as np
from config import (PT_DBM, G_TX_DBI, G_RX_DBI, LAMBDA_M, BW_HZ, NF_DB, LOSS_DB,
                    GP_DB, RCS_M2, PFA, PD_TARGET, C_LIGHT,
                    SIG_R_FLOOR_M, SIG_AZ_FLOOR_DEG, SIG_EL_FLOOR_DEG,
                    THETA_AZ_3DB_DEG, THETA_EL_3DB_DEG, ALT_LAYERS, N_GRID, CELL_M)

KT0_DBM_HZ = -174.0          # 10*log10(k*T0)  dBm/Hz
NOISE_DBM = KT0_DBM_HZ + 10.0 * np.log10(BW_HZ) + NF_DB

# ---- 现实性修正项（使结果落在 IMT-2030 量级而非理想界）----
NLOS_EXCESS_DB = 25.0   # 建筑遮挡造成的额外损耗 (dB, 4.9GHz 城区穿透量级)
LOS_MARGIN_M = 2.0      # 视线上建筑高于射线超过此值判为遮挡
N_LOS_SAMPLES = 10      # 视线采样点数
# 系统误差：时钟同步残差、NLOS 偏置、站址标定误差 —— 共模，不随多站平均而抵消
SYS_SIG_H_M = 3.0
SYS_SIG_V_M = 5.0

# 雷达方程常数项：10log10(lambda^2 * sigma / (4*pi)^3)
_RADAR_CONST = (20.0 * np.log10(LAMBDA_M)
                + 10.0 * np.log10(RCS_M2)
                - 30.0 * np.log10(4.0 * np.pi))


def snr_db(dist_m, rcs_m2=None, gp_db=None):
    """单链路接收 SNR (dB)。dist_m 可为数组；rcs/gp 用于敏感性分析覆盖。"""
    d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
    const = _RADAR_CONST
    if rcs_m2 is not None:
        const = _RADAR_CONST - 10.0 * np.log10(RCS_M2) + 10.0 * np.log10(rcs_m2)
    gp = GP_DB if gp_db is None else gp_db
    pr_dbm = PT_DBM + G_TX_DBI + G_RX_DBI + const - 40.0 * np.log10(d) - LOSS_DB
    return pr_dbm - NOISE_DBM + gp


def pd_from_snr(snr_db_arr, pfa=PFA):
    """Swerling I 目标、单脉冲线性检波：Pd = Pfa^(1/(1+SNR))."""
    snr_lin = np.power(10.0, np.asarray(snr_db_arr, dtype=float) / 10.0)
    return np.power(pfa, 1.0 / (1.0 + snr_lin))


def snr_required(pd_target=PD_TARGET, pfa=PFA):
    """达到目标检测概率所需的最小 SNR (dB)。"""
    # Pd = pfa^(1/(1+S))  =>  1/(1+S) = ln(Pd)/ln(pfa)
    inv = np.log(pd_target) / np.log(pfa)
    s = 1.0 / inv - 1.0
    return 10.0 * np.log10(s)


def _noncoherent_gain(n_link):
    """多站非相干积累增益：10log10(N^0.7)。"""
    n = np.maximum(np.asarray(n_link, dtype=float), 1.0)
    return 7.0 * np.log10(n)


def _meas_sigmas(snr_db_arr):
    """给定 SNR，返回 (测距标准差 m, 方位角标准差 rad, 俯仰角标准差 rad)。"""
    snr_lin = np.power(10.0, np.asarray(snr_db_arr, dtype=float) / 10.0)
    s = np.sqrt(2.0 * snr_lin)
    sig_r = np.sqrt((C_LIGHT / (2.0 * BW_HZ * s)) ** 2 + SIG_R_FLOOR_M ** 2)
    th_az = np.deg2rad(THETA_AZ_3DB_DEG)
    th_el = np.deg2rad(THETA_EL_3DB_DEG)
    sig_az = np.sqrt((th_az / (1.6 * s)) ** 2 + np.deg2rad(SIG_AZ_FLOOR_DEG) ** 2)
    sig_el = np.sqrt((th_el / (1.6 * s)) ** 2 + np.deg2rad(SIG_EL_FLOOR_DEG) ** 2)
    return sig_r, sig_az, sig_el


def los_attenuation(scene, points, bs_pos, chunk=800):
    """
    建筑遮挡判定的附加损耗 (dB)。points: (P,3) 目标点, bs_pos: (M,3)。
    返回 (P,M) 的附加损耗矩阵：LOS=0，NLOS=NLOS_EXCESS_DB。
    """
    bh = scene["building_h"]
    n = scene["n"]
    cell = scene["cell"]
    P = points.shape[0]; M = bs_pos.shape[0]
    att = np.zeros((P, M), dtype=float)
    t = np.linspace(0.0, 1.0, N_LOS_SAMPLES + 2)[1:-1]     # 不取端点
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        pts = points[s:e]                                   # (p,3)
        p1 = bs_pos[None, :, :]                             # (1,M,3)
        p2 = pts[:, None, :]                                # (p,1,3)
        seg = p1[:, :, None, :] + (p2 - p1)[:, :, None, :] * t[None, None, :, None]  # (p,M,K,3)
        ix = np.clip((seg[..., 0] / cell).astype(int), 0, n - 1)
        iy = np.clip((seg[..., 1] / cell).astype(int), 0, n - 1)
        hb = bh[ix, iy]                                     # (p,M,K)
        zr = seg[..., 2]
        blocked = np.any(hb > zr + LOS_MARGIN_M, axis=-1)   # (p,M)
        att[s:e] = np.where(blocked, NLOS_EXCESS_DB, 0.0)
    return att


def apply_realism(sig_h, sig_v):
    """叠加共模系统误差：不随多站融合平均而抵消。"""
    return (np.sqrt(np.asarray(sig_h, float) ** 2 + SYS_SIG_H_M ** 2),
            np.sqrt(np.asarray(sig_v, float) ** 2 + SYS_SIG_V_M ** 2))


def _crlb(target, bs_pos, snr_db_arr, snr_thresh_db):
    """多站测距+测角融合的定位 CRLB。返回 (sigma_h, sigma_v)。"""
    d = bs_pos - target[None, :]                 # (M,3)
    dist = np.linalg.norm(d, axis=1)
    ok = snr_db_arr >= snr_thresh_db
    if not np.any(ok):
        return np.inf, np.inf
    d = d[ok]; dist = dist[ok]; snr = snr_db_arr[ok]
    sig_r, sig_az, sig_el = _meas_sigmas(snr)

    fim = np.zeros((3, 3))
    for k in range(len(dist)):
        dx, dy, dz = d[k]
        R = dist[k]
        rho = np.hypot(dx, dy)
        if rho < 1e-6 or R < 1e-6:
            continue
        # 测距
        g_r = np.array([dx, dy, dz]) / R
        fim += np.outer(g_r, g_r) / sig_r[k] ** 2
        # 方位角
        g_az = np.array([-dy / rho ** 2, dx / rho ** 2, 0.0])
        fim += np.outer(g_az, g_az) / sig_az[k] ** 2
        # 俯仰角
        g_el = np.array([-dz * dx / (R ** 2 * rho),
                         -dz * dy / (R ** 2 * rho),
                         rho / R ** 2])
        fim += np.outer(g_el, g_el) / sig_el[k] ** 2

    try:
        crlb = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        return np.inf, np.inf
    sh = np.sqrt(max(crlb[0, 0] + crlb[1, 1], 0.0))
    sv = np.sqrt(max(crlb[2, 2], 0.0))
    return float(sh), float(sv)


def coverage_maps(scene, alt_layers=ALT_LAYERS, snr_thresh_db=None,
                  rcs_m2=None, gp_db=None):
    """对每个高度层计算：检测概率图、水平/垂直定位精度图。"""
    if snr_thresh_db is None:
        snr_thresh_db = snr_required()
    bs = scene["bs_pos"]
    _snr = lambda d: snr_db(d, rcs_m2=rcs_m2, gp_db=gp_db)
    X, Y = scene["X"], scene["Y"]
    out = {}
    for alt in np.atleast_1d(alt_layers):
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, alt)], axis=1)
        # 逐点计算（6400 点 × 层），向量化距离
        d = pts[:, None, :] - bs[None, :, :]          # (P, M, 3)
        dist = np.linalg.norm(d, axis=2)              # (P, M)
        snr = _snr(dist) - los_attenuation(scene, pts, bs)   # 含建筑遮挡
        snr_max = snr.max(axis=1)
        n_link = np.sum(snr >= snr_thresh_db, axis=1)
        snr_eff = snr_max + _noncoherent_gain(n_link)
        pd = pd_from_snr(snr_eff)

        shape = (scene["n"], scene["n"])
        sh = np.zeros(pts.shape[0]); sv = np.zeros(pts.shape[0])
        for i in range(pts.shape[0]):
            sh[i], sv[i] = _crlb(pts[i], bs, snr[i], snr_thresh_db)
        sh, sv = apply_realism(sh, sv)
        out[float(alt)] = dict(pd=pd.reshape(shape), sig_h=sh.reshape(shape),
                               sig_v=sv.reshape(shape), n_link=n_link.reshape(shape))
    return out


def coverage_kpi(maps, pd_target=PD_TARGET, horiz_target=5.0, vert_target=8.0):
    """汇总覆盖率与定位精度达标率。"""
    pd_all, sh_all, sv_all = [], [], []
    for alt, m in maps.items():
        pd_all.append(m["pd"].ravel()); sh_all.append(m["sig_h"].ravel()); sv_all.append(m["sig_v"].ravel())
    pd_all = np.concatenate(pd_all)
    sh_all = np.concatenate(sh_all)
    sv_all = np.concatenate(sv_all)
    valid = np.isfinite(sh_all) & np.isfinite(sv_all)
    return dict(
        snr_required_db=float(snr_required(pd_target)),
        coverage_pd=float(np.mean(pd_all >= pd_target)),
        coverage_full=float(np.mean((pd_all >= pd_target) & valid &
                                    (sh_all <= horiz_target) & (sv_all <= vert_target))),
        mean_pd=float(np.mean(pd_all)),
        median_sig_h=float(np.nanmedian(np.where(valid, sh_all, np.nan))),
        median_sig_v=float(np.nanmedian(np.where(valid, sv_all, np.nan))),
        p95_sig_h=float(np.nanpercentile(np.where(valid, sh_all, np.nan), 95)),
        p95_sig_v=float(np.nanpercentile(np.where(valid, sv_all, np.nan), 95)),
        horiz_pass=float(np.mean(valid & (sh_all <= horiz_target))),
        vert_pass=float(np.mean(valid & (sv_all <= vert_target))),
    )


def track_perception(scene, track_xyz, snr_thresh_db=None):
    """航迹逐点的检测概率与定位精度。track_xyz: (T,3)"""
    if snr_thresh_db is None:
        snr_thresh_db = snr_required()
    bs = scene["bs_pos"]
    d = track_xyz[:, None, :] - bs[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    snr = snr_db(dist) - los_attenuation(scene, track_xyz, bs)
    snr_max = snr.max(axis=1)
    n_link = np.sum(snr >= snr_thresh_db, axis=1)
    pd = pd_from_snr(snr_max + _noncoherent_gain(n_link))
    sh = np.zeros(len(track_xyz)); sv = np.zeros(len(track_xyz))
    for i in range(len(track_xyz)):
        sh[i], sv[i] = _crlb(track_xyz[i], bs, snr[i], snr_thresh_db)
    sh, sv = apply_realism(sh, sv)
    return pd, sh, sv
