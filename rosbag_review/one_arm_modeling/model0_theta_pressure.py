# theta_pressure.py
import numpy as np
import matplotlib.pyplot as plt

# ---- 設定 ----
L0_list_cm = [24.5]
R_m = 0.1655
r_m = 0.05094
p_min_bar, p_max_bar = 0.0, 7.0
num_points = 600
USE_L_SQUARED = True  # 余弦定理で L^2 を使う（図の式をそのまま使うなら False）
# ========= ユーザー設定（軸幅） =========
P_RANGE      = (0.0, 7.0)        # x軸: 圧力 p [bar]
L_RANGE_CM   = (0, 25.0)      # 左y軸: L [cm]（None なら自動）
THETA_RANGE  = (0, 1.6)      # 右y軸: theta [rad]（None なら自動）
# =====================================

# ---- 係数行列 ----
M = np.array([
    [ 0.4351, 0.0,     0.0183,  -0.0003  ],
    [ 0.5649, 0.0,    -0.0183,   0.0003  ],
    [-0.0141, 0.0,     0.0031,  -0.00006 ],
    [ 0.5487, 0.0,    -0.0136,   0.00007 ],
    [ 0.0,    0.3694,  0.0,      0.0     ],
])

def coeffs_from_L0(L0_cm: float):
    X = np.array([L0_cm, L0_cm**-0.248, L0_cm**2, L0_cm**3])
    a, b, c, d, e = M @ X
    return a, b, c, d, e

def L_of_p_bar(p_bar: np.ndarray, L0_cm: float):
    a, b, c, d, e = coeffs_from_L0(L0_cm)
    term = (p_bar / c) ** d
    return a + b / (1 + term) ** e - 0.009 * L0_cm * np.sqrt(p_bar)

def theta_from_L_cm(L_cm: np.ndarray, R_m: float, r_m: float, use_L_squared: bool = True):
    L_m = L_cm / 100.0 - 0.031
    if use_L_squared:
        x = (R_m**2 + r_m**2 - L_m**2) / (2.0 * R_m * r_m)
    else:
        x = (R_m**2 + r_m**2 - L_m**2) / (2.0 * R_m * r_m)
    x = np.clip(x, -1.0, 1.0)
    return np.pi/2.0 - np.arccos(x)

# ---- 計算 ----
p_bar = np.linspace(p_min_bar, p_max_bar, num_points)
curves = {}
for L0 in L0_list_cm:
    L_cm = L_of_p_bar(p_bar, L0)
    theta_rad = theta_from_L_cm(L_cm, R_m, r_m, use_L_squared=USE_L_SQUARED)
    curves[L0] = {"L_cm": L_cm, "theta_rad": theta_rad}

# ---- プロット（同一ウィンドウに2枚）----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: L(p)
ax = axes[0]
for L0, data in curves.items():
    ax.plot(p_bar, data["L_cm"], label=f"L0={L0} cm")
ax.set_xlabel("Pressure p [bar]")
ax.set_ylabel("L [cm]")
ax.set_title("L(p)")
ax.grid(True)
ax.legend()
ax.set_xlim(*P_RANGE)
if L_RANGE_CM is not None:
    ax.set_ylim(*L_RANGE_CM)

# 右: theta(p)
ax = axes[1]
for L0, data in curves.items():
    ax.plot(p_bar, data["theta_rad"], label=f"L0={L0} cm")
ax.set_xlabel("Pressure p [bar]")
ax.set_ylabel("theta [rad]")
ax.set_title("theta(p)")
ax.grid(True)
ax.legend()
ax.set_xlim(*P_RANGE)
if THETA_RANGE is not None:
    ax.set_ylim(*THETA_RANGE)

plt.tight_layout()
plt.show()

# 係数の確認（任意）
for L0 in L0_list_cm:
    a,b,c,d,e = coeffs_from_L0(L0)
    print(f"L0={L0} cm -> a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}, e={e:.6f}")
