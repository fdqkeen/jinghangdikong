# -*- coding: utf-8 -*-
"""
城市栅格场景生成：建筑高度、声环境功能区划、人口密度、起降点、基站、声学节点
"""
import numpy as np
from config import (AREA_M, CELL_M, N_GRID, CELL_AREA_KM2, BS_SPACING_M, BS_HEIGHT_M,
                    N_ACOUSTIC_NODES, N_ACOUSTIC_NODES_MAX,
                    ZONE_LIMITS, ZONE_POP_DENSITY, SEED)


def _block_buildings(rng, n, block_cells=8):
    """街区式建筑高度图：道路网格 + 街区内建筑。"""
    h = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            on_road = (i % block_cells == 0) or (j % block_cells == 0)
            if on_road:
                h[i, j] = 0.0
            else:
                base = rng.uniform(12.0, 45.0)
                if rng.random() < 0.12:          # 高层
                    base = rng.uniform(60.0, 120.0)
                h[i, j] = base
    # 轻微平滑，消除棋盘感
    from scipy.ndimage import uniform_filter
    h = uniform_filter(h, size=2)
    return h


def _zones(rng, n):
    """声环境功能区划：用随机中心 + 最近邻分配，形成有机分区。"""
    centers = []
    labels = []
    # 1类 居住文教（占比最大）
    for _ in range(6):
        centers.append(rng.uniform(0, n, 2)); labels.append(1)
    for _ in range(4):
        centers.append(rng.uniform(0, n, 2)); labels.append(2)
    for _ in range(2):
        centers.append(rng.uniform(0, n, 2)); labels.append(3)
    for _ in range(1):
        centers.append(rng.uniform(0, n, 2)); labels.append(0)   # 疗养/医院
    centers = np.array(centers); labels = np.array(labels)

    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pts = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(d, axis=1)
    z = labels[assign].reshape(n, n)

    # 交通干线（4a 类）：沿两条主干道
    z[n // 4, :] = 4
    z[:, int(n * 0.7)] = 4
    return z


def build_scene(seed=SEED, bs_spacing=None):
    rng = np.random.default_rng(seed)
    n = N_GRID
    bs_spacing = bs_spacing or BS_SPACING_M

    building_h = _block_buildings(rng, n)
    zone = _zones(rng, n)

    pop_density = np.vectorize(lambda z: ZONE_POP_DENSITY[int(z)])(zone)     # 人/km^2
    pop = pop_density * CELL_AREA_KM2 * rng.uniform(0.7, 1.3, (n, n))       # 人/栅格
    pop = np.clip(pop, 0.0, None)

    # 栅格中心坐标
    xs = (np.arange(n) + 0.5) * CELL_M
    ys = (np.arange(n) + 0.5) * CELL_M
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # 禁飞/限高区（学校、医院、党政机关等），用圆形区域近似
    no_fly = []
    for _ in range(4):
        cx, cy = rng.uniform(0.15, 0.85, 2) * AREA_M
        r = rng.uniform(250.0, 500.0)
        no_fly.append((float(cx), float(cy), float(r)))

    # 起降点（垂直起降场）：选在 2类/4a 类区域
    vertiports = []
    cand = np.argwhere((zone == 2) | (zone == 4))
    pick = cand[rng.choice(len(cand), size=3, replace=False)]
    for (i, j) in pick:
        vertiports.append((float(X[i, j]), float(Y[i, j]), 30.0))
    vertiports = np.array(vertiports)

    # 6G ISAC 基站（城区网格布站）
    n_bs = int(AREA_M / bs_spacing) + 1
    bs_xy = []
    off = (AREA_M - (n_bs - 1) * bs_spacing) / 2.0
    for a in range(n_bs):
        for b in range(n_bs):
            bs_xy.append((off + a * bs_spacing, off + b * bs_spacing))
    bs_xy = np.array(bs_xy)
    bs_pos = np.column_stack([bs_xy, np.full(len(bs_xy), BS_HEIGHT_M)])

    # 声学节点：优先布在敏感区(0/1类)与起降点周边，其余随机
    # 生成 N_ACOUSTIC_NODES_MAX 个构成节点池，主实验取前 N_ACOUSTIC_NODES 个
    sens = np.argwhere((zone == 0) | (zone == 1))
    n_sens = int(N_ACOUSTIC_NODES_MAX * 0.6)
    idx = sens[rng.choice(len(sens), size=min(n_sens, len(sens)), replace=False)]
    node_cells = [tuple(v) for v in idx]
    while len(node_cells) < N_ACOUSTIC_NODES_MAX:
        node_cells.append((int(rng.integers(0, n)), int(rng.integers(0, n))))
    node_cells = np.array(node_cells)
    node_pos = np.column_stack([X[node_cells[:, 0], node_cells[:, 1]],
                                Y[node_cells[:, 0], node_cells[:, 1]],
                                np.full(len(node_cells), 1.5)])  # 1.5 m 传声器高度

    return dict(
        n=n, xs=xs, ys=ys, X=X, Y=Y, cell=CELL_M, area=AREA_M,
        building_h=building_h, zone=zone, pop=pop,
        zone_limit_day=np.vectorize(lambda z: ZONE_LIMITS[int(z)][0])(zone),
        zone_limit_night=np.vectorize(lambda z: ZONE_LIMITS[int(z)][1])(zone),
        no_fly=no_fly, vertiports=vertiports,
        bs_pos=bs_pos, node_pos=node_pos, node_cells=node_cells,
    )


def zone_stats(scene):
    n = scene["n"]
    tot = n * n
    out = []
    for z in sorted(ZONE_LIMITS.keys()):
        c = int(np.sum(scene["zone"] == z))
        out.append((z, c, c / tot))
    return out
