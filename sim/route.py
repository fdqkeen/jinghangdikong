# -*- coding: utf-8 -*-
"""
安全—效率—静音 三目标航路优化（自实现 NSGA-II，无第三方优化库依赖）
决策变量：中间航路点 (x, y, z)
目标：f1 航程（能耗代理） / f2 地面噪声暴露（人·dB） / f3 地面安全风险
约束：高度窗、爬升梯度、禁飞区（惩罚）、空域边界
"""
import numpy as np
from config import (AREA_M, ALT_MIN, ALT_MAX, POP_ROUTE_ORIGIN, POP_ROUTE_DEST,
                    N_WAYPOINTS, NSGA2_POP, NSGA2_GEN, NSGA2_CROSS_P,
                    NSGA2_ETA_C, NSGA2_ETA_M, LW_HOVER_DBA, LW_CRUISE_DBA)
import noise as noise_mod

VERTIPORT_ALT = 30.0          # 起降点平台高度 m
CRUISE_SWITCH_ALT = 60.0      # 低于此高度按悬停声功率级计
MAX_CLIMB_GRADIENT = 0.35     # 爬升梯度上限（爬升率3.5m/s / 巡航10m/s 量级）
LOW_ALT_RISK = 80.0           # 低于此高度计地面安全风险
N_SAMPLES = 24                # 航路采样点数（优化期）

# 决策变量边界
XY_LO, XY_HI = 150.0, AREA_M - 150.0
Z_LO, Z_HI = 40.0, ALT_MAX
# 惩罚系数标定到与目标函数同量级（f1 约 4e3~2e4 m，f2 约 1e4~1e5 人·dB）
NO_FLY_COEF = 5.0e3      # × 平均穿越深度(m)
CLIMB_COEF = 2.0e4       # × 超梯度高差(m)


def _full_waypoints(x):
    """把决策向量拼成完整航路点序列（含起降点）。"""
    mid = x.reshape(-1, 3)
    a = np.array([POP_ROUTE_ORIGIN[0], POP_ROUTE_ORIGIN[1], VERTIPORT_ALT])
    b = np.array([POP_ROUTE_DEST[0], POP_ROUTE_DEST[1], VERTIPORT_ALT])
    return np.vstack([a, mid, b])


def _no_fly_penalty(wp, no_fly, n_samples=N_SAMPLES):
    """航路穿越禁飞区的平均穿越深度 (m)。"""
    if not no_fly:
        return 0.0
    tr = noise_mod.sample_track(wp, n_samples)
    pen = 0.0
    for (cx, cy, r) in no_fly:
        d = np.hypot(tr[:, 0] - cx, tr[:, 1] - cy)
        pen += np.mean(np.clip(r - d, 0.0, None))
    return pen


def _climb_penalty(wp):
    """超出爬升梯度的惩罚。"""
    seg_h = np.linalg.norm(np.diff(wp[:, :2], axis=0), axis=1)
    dz = np.abs(np.diff(wp[:, 2]))
    excess = np.clip(dz - MAX_CLIMB_GRADIENT * np.maximum(seg_h, 1.0), 0.0, None)
    return float(np.sum(excess))


def evaluate(x, grid, scene, night=True, n_samples=N_SAMPLES, full=False):
    """
    返回 (f1 航程, f2 噪声暴露, f3 风险) 及明细。
    full=True 时用全分辨率栅格并输出 Lmax 场。
    """
    wp = _full_waypoints(x)
    tr = noise_mod.sample_track(wp, n_samples)
    pen_nf = _no_fly_penalty(wp, scene["no_fly"], n_samples)
    pen_cl = _climb_penalty(wp)
    penalty = pen_nf * NO_FLY_COEF + pen_cl * CLIMB_COEF

    # f1 航程
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    f1 = float(seg.sum())

    # f2 噪声暴露 + f3 风险
    alt = tr[:, 2]
    lmax = np.full(grid["X"].shape, -np.inf)
    risk = 0.0
    for k, p in enumerate(tr):
        lw = LW_HOVER_DBA if p[2] < CRUISE_SWITCH_ALT else LW_CRUISE_DBA
        f = noise_mod.lp_field(p[:2], p[2], grid, lw)
        lmax = np.maximum(lmax, f)
    # 风险：低空段经过的人口总量（用最近栅格近似）
    cell = (AREA_M / grid["X"].shape[0])
    for k, p in enumerate(tr):
        if p[2] < LOW_ALT_RISK:
            i = int(np.clip(p[0] / cell, 0, grid["X"].shape[0] - 1))
            j = int(np.clip(p[1] / cell, 0, grid["X"].shape[1] - 1))
            risk += grid["pop"][i, j]

    limit = grid["limit_night"] if night else grid["limit_day"]
    exc = np.clip(lmax - limit, 0.0, None)
    f2 = float(np.sum(grid["pop"] * exc))

    out = dict(f1=f1 + penalty, f2=f2 + penalty, f3=risk + penalty * 1e-2,
               length=f1, exposure=f2, risk=risk, penalty=penalty,
               pen_nf=pen_nf, pen_cl=pen_cl)
    if full:
        out["lmax"] = lmax
        out["track"] = tr
        out["waypoints"] = wp
        out["detail"] = noise_mod.exposure(lmax, grid, night=night)
    return out


# ---------------- NSGA-II ----------------

def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def _non_dominated_sort(F):
    n = F.shape[0]
    ranks = np.zeros(n, dtype=int)
    fronts = []
    S = [[] for _ in range(n)]
    ndom = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(F[i], F[j]):
                S[i].append(j); ndom[j] += 1
            elif _dominates(F[j], F[i]):
                S[j].append(i); ndom[i] += 1
    cur = [i for i in range(n) if ndom[i] == 0]
    r = 0
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in S[i]:
                ndom[j] -= 1
                if ndom[j] == 0:
                    nxt.append(j)
        ranks[cur] = r
        cur = nxt
        r += 1
    return ranks, fronts


def _crowding(F, front):
    m = F.shape[1]
    k = len(front)
    cd = np.zeros(k)
    if k <= 2:
        return np.full(k, np.inf)
    for j in range(m):
        vals = F[front, j]
        order = np.argsort(vals)
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        span = vals[order[-1]] - vals[order[0]]
        if span <= 0:
            continue
        for t in range(1, k - 1):
            cd[order[t]] += (vals[order[t + 1]] - vals[order[t - 1]]) / span
    return cd


def _sbx(p1, p2, lo, hi, eta=NSGA2_ETA_C, pc=NSGA2_CROSS_P):
    c1, c2 = p1.copy(), p2.copy()
    if np.random.random() > pc:
        return c1, c2
    for k in range(len(p1)):
        if np.random.random() > 0.5:
            continue
        if abs(p1[k] - p2[k]) < 1e-12:
            continue
        y1, y2 = min(p1[k], p2[k]), max(p1[k], p2[k])
        r = np.random.random()
        beta = 1.0 + 2.0 * (y1 - lo[k]) / (y2 - y1)
        alpha = float(np.clip(2.0 - beta ** -(eta + 1.0), 1.0 + 1e-9, 2.0 - 1e-9))
        if r <= 1.0 / alpha:
            beta_q = (r * alpha) ** (1.0 / (eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - r * alpha)) ** (1.0 / (eta + 1.0))
        c1[k] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
        c2[k] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def _mutate(p, lo, hi, eta=NSGA2_ETA_M):
    q = p.copy()
    for k in range(len(p)):
        if np.random.random() > 1.0 / len(p):
            continue
        r = np.random.random()
        delta = (2.0 * r) ** (1.0 / (eta + 1.0)) - 1.0 if r < 0.5 \
            else 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))
        q[k] += delta * (hi[k] - lo[k])
    return np.clip(q, lo, hi)


def nsga2(grid, scene, seed=0, pop_size=NSGA2_POP, n_gen=NSGA2_GEN, night=True,
          verbose=False):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n_var = 3 * N_WAYPOINTS
    lo = np.tile([XY_LO, XY_LO, Z_LO], N_WAYPOINTS)
    hi = np.tile([XY_HI, XY_HI, Z_HI], N_WAYPOINTS)

    # 初始化：起降点连线附近横向扰动 + 可行爬升剖面（避免初代大量不可行）
    P = np.zeros((pop_size, n_var))
    a, b = POP_ROUTE_ORIGIN, POP_ROUTE_DEST
    for i in range(pop_size):
        alt_cruise = rng.uniform(80.0, Z_HI)
        for k in range(N_WAYPOINTS):
            t = (k + 1) / (N_WAYPOINTS + 1.0)
            base = a + (b - a) * t
            P[i, 3 * k:3 * k + 2] = base + rng.normal(0, 450, 2)
            # 爬升/下降包络：两端低、中间巡航高度
            ramp = min(t / 0.30, (1.0 - t) / 0.30, 1.0)
            P[i, 3 * k + 2] = VERTIPORT_ALT + (alt_cruise - VERTIPORT_ALT) * ramp
            P[i, 3 * k + 2] += rng.normal(0, 8.0)
    P = np.clip(P, lo, hi)

    def _eval_pop(P):
        F = np.zeros((len(P), 3))
        for i, x in enumerate(P):
            e = evaluate(x, grid, scene, night=night)
            F[i] = [e["f1"], e["f2"], e["f3"]]
        return F

    F = _eval_pop(P)
    for gen in range(n_gen):
        # 二元锦标赛
        idx = rng.integers(0, pop_size, size=(pop_size, 2))
        win = np.array([i if _dominates(F[i], F[j]) else (j if _dominates(F[j], F[i])
                        else (i if rng.random() < 0.5 else j)) for i, j in idx])
        Q = []
        for t in range(0, pop_size, 2):
            p1, p2 = P[win[t]], P[win[(t + 1) % pop_size]]
            c1, c2 = _sbx(p1, p2, lo, hi)
            Q.append(_mutate(c1, lo, hi)); Q.append(_mutate(c2, lo, hi))
        Q = np.array(Q)[:pop_size]
        FQ = _eval_pop(Q)

        R = np.vstack([P, Q]); FR = np.vstack([F, FQ])
        ranks, fronts = _non_dominated_sort(FR)
        newP, newF = [], []
        for fr in fronts:
            if len(newP) + len(fr) <= pop_size:
                newP.extend(R[fr]); newF.extend(FR[fr])
            else:
                cd = _crowding(FR, fr)
                order = np.argsort(-cd)
                need = pop_size - len(newP)
                for t in order[:need]:
                    newP.append(R[fr[t]]); newF.append(FR[fr[t]])
                break
        P, F = np.array(newP), np.array(newF)
        if verbose and (gen + 1) % 25 == 0:
            print(f"  gen {gen+1}/{n_gen}  front0={np.sum(ranks==0)}  "
                  f"f2 min={F[:,1].min():.0f}  f1 min={F[:,0].min():.0f}")
    return P, F


def baseline_route(grid, scene, cruise_alt=80.0, night=True, n_samples=120):
    """现状基线：起降点间直线 + 固定巡航高度。"""
    a, b = POP_ROUTE_ORIGIN, POP_ROUTE_DEST
    wp = np.array([[a[0], a[1], VERTIPORT_ALT],
                   [a[0] + (b[0] - a[0]) * 0.15, a[1] + (b[1] - a[1]) * 0.15, cruise_alt],
                   [a[0] + (b[0] - a[0]) * 0.85, a[1] + (b[1] - a[1]) * 0.85, cruise_alt],
                   [b[0], b[1], VERTIPORT_ALT]])
    tr = noise_mod.sample_track(wp, n_samples)
    alt = tr[:, 2]
    lmax = np.full(grid["X"].shape, -np.inf)
    for p in tr:
        lw = LW_HOVER_DBA if p[2] < CRUISE_SWITCH_ALT else LW_CRUISE_DBA
        lmax = np.maximum(lmax, noise_mod.lp_field(p[:2], p[2], grid, lw))
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    limit = grid["limit_night"] if night else grid["limit_day"]
    exc = np.clip(lmax - limit, 0.0, None)
    cell = AREA_M / grid["X"].shape[0]
    risk = 0.0
    for p in tr:
        if p[2] < LOW_ALT_RISK:
            i = int(np.clip(p[0] / cell, 0, grid["X"].shape[0] - 1))
            j = int(np.clip(p[1] / cell, 0, grid["X"].shape[1] - 1))
            risk += grid["pop"][i, j]
    return dict(waypoints=wp, track=tr, lmax=lmax,
                length=float(seg.sum()),
                exposure=float(np.sum(grid["pop"] * exc)),
                risk=risk,
                detail=noise_mod.exposure(lmax, grid, night=night))
