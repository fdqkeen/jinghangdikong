# -*- coding: utf-8 -*-
"""
全局配置：城市低空「监管—降噪」场景仿真
所有参数集中在此，便于评委复现与敏感性分析。
"""
import numpy as np

# ---------------- 场景几何 ----------------
AREA_M = 4000.0          # 区域边长 (m)
CELL_M = 50.0            # 栅格分辨率 (m)
N_GRID = int(AREA_M / CELL_M)          # 80
CELL_AREA_KM2 = (CELL_M ** 2) / 1e6    # 0.0025 km^2

# 评估空域高度层 (m)
ALT_LAYERS = np.array([60.0, 100.0, 150.0])
ALT_MIN, ALT_MAX = 60.0, 300.0

# ---------------- 6G ISAC 基站 ----------------
BS_SPACING_M = 350.0     # 城区网格布站间距 (m)
BS_HEIGHT_M = 30.0       # 基站挂高 (m)

# 链路参数（密集城区 ISAC，Sub-6G）
PT_DBM = 43.0            # 发射功率 20 W
G_TX_DBI = 24.0          # 发射天线增益
G_RX_DBI = 24.0          # 接收天线增益
FC_HZ = 4.9e9            # 载波频率
C_LIGHT = 3.0e8
LAMBDA_M = C_LIGHT / FC_HZ
BW_HZ = 100e6            # 感知带宽
NF_DB = 5.0              # 噪声系数
LOSS_DB = 4.0            # 系统损耗
GP_DB = 25.0             # OFDM 相干处理增益（时频二维积累）

RCS_M2 = 0.01            # 小型消费级无人机 RCS (-20 dBsm)
PFA = 0.05               # 虚警概率（对齐 IMT-2030 要求 <=5%）
PD_TARGET = 0.95         # 目标检测概率（对齐 IMT-2030 要求 >=95%）

# 测量误差下界（系统误差 floor，含多径/同步/阵列校准残余）
SIG_R_FLOOR_M = 1.5            # 测距 (m)
SIG_AZ_FLOOR_DEG = 0.5         # 方位角 (deg)
SIG_EL_FLOOR_DEG = 1.0         # 俯仰角 (deg)
THETA_AZ_3DB_DEG = 6.0         # 方位 3dB 波束宽度
THETA_EL_3DB_DEG = 10.0        # 俯仰 3dB 波束宽度

# ---------------- 声学 ----------------
"""
声源强度标定依据（重要，已纠错）：
GB 42590-2023 §4.14 的标识值归一化到 1 m，其示例表（全消声室、微型/轻型旋翼机）
给出地面观测点悬停 70.1 dB(A)、飞行 1 m/s 76.1 dB(A)。该值用于证明"噪声是强制性
标识指标"，但其量级对应微型机且为消声室条件，不宜作为城市运行场景的声源强度标定。

本方案改用随飞行高度的实测衰减表标定（AIMS Bioengineering 2026, Table 2；
并与独立工程来源交叉验证一致）：
    多旋翼无人机   10 m: 75-80    50 m: 62-67   100 m: 55-60 dB(A)
    eVTOL         10 m: 85-95    50 m: 72-78   100 m: 65-70
    传统直升机     10 m: 90-100   50 m: 75-82   100 m: 68-75
取悬停=区间下界 75、巡航=区间上界 80，反推 Lw 后回代校验：
    Lw=106 -> 50 m: 61.0, 100 m: 55.0   （均落在实测区间内）
    Lw=111 -> 50 m: 66.0, 100 m: 60.0   （均落在实测区间内）
"""
REF_DIST_M = 10.0
LP10_HOVER_DBA = 75.0     # 多旋翼悬停（实测区间 75-80 下界）
LP10_CRUISE_DBA = 80.0    # 多旋翼巡航（实测区间 75-80 上界）
LP10_EVTOL_DBA = 90.0     # eVTOL（85-95 中位）
LP10_HELI_DBA = 92.5      # 传统直升机（90-100 中位）

# 由 Lp(10m) 反推声功率级：自由场点声源 Lp = Lw - 20log10(r) - 11
def lp_to_lw(lp_at_ref: float, ref: float = REF_DIST_M) -> float:
    return lp_at_ref + 20.0 * np.log10(ref) + 11.0

LW_HOVER_DBA = float(lp_to_lw(LP10_HOVER_DBA))    # 106.0
LW_CRUISE_DBA = float(lp_to_lw(LP10_CRUISE_DBA))  # 111.0
LW_EVTOL_DBA = float(lp_to_lw(LP10_EVTOL_DBA))    # 121.0
LW_HELI_DBA = float(lp_to_lw(LP10_HELI_DBA))      # 123.5

ALPHA_DB_PER_KM = 3.0    # 空气吸收（A 计权代表性值，15C/50%RH）
ABAR_MAX_DB = 20.0       # ISO 9613-2 单屏障衰减上限
NOISE_MEAS_SIGMA_DB = 1.5   # 声级计测量标准差 (dB)

# GB 3096-2008 声环境功能区限值 dB(A)：{区类: (昼, 夜)}
ZONE_LIMITS = {0: (50.0, 40.0),
               1: (55.0, 45.0),
               2: (60.0, 50.0),
               3: (65.0, 55.0),
               4: (70.0, 55.0)}
ZONE_NAMES = {0: "0类 康复疗养", 1: "1类 居住文教", 2: "2类 商住混合",
              3: "3类 工业", 4: "4a类 交通干线"}
ZONE_POP_DENSITY = {0: 500.0, 1: 15000.0, 2: 8000.0, 3: 2000.0, 4: 3000.0}  # 人/km^2

N_ACOUSTIC_NODES = 40     # 主实验使用的稀疏声学校准节点数量
N_ACOUSTIC_NODES_MAX = 100  # 场景生成的节点池上限（供敏感性分析取子集）

# ---------------- 航路优化 ----------------
POP_ROUTE_ORIGIN = np.array([400.0, 600.0])    # 起降点 A (m)
POP_ROUTE_DEST = np.array([3600.0, 3400.0])   # 起降点 B (m)
N_WAYPOINTS = 6           # 中间航路点数量
RISK_WEIGHT = 1.0         # 安全风险项权重

NSGA2_POP = 120
NSGA2_GEN = 250
NSGA2_CROSS_P = 0.9
NSGA2_ETA_C = 15.0
NSGA2_ETA_M = 20.0

# ---------------- 随机性 ----------------
SEED = 20260928
OUT_DIR = "outputs"
FIG_DIR = "figs"
