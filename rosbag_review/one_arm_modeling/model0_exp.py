# theta_pressure_with_experiment.py
# --------------------------------
# モデルの L(p), theta(p) を描画し、実験値を重ねて残差とRMSE/MAE/Biasを表示します。

import numpy as np
import matplotlib.pyplot as plt

# ===== ユーザー設定 =====
L0_list_cm = [24.5]     # 必要なら複数個入れてOK
R_m = 0.1655            # 上側プーリ半径 [m]
r_m = 0.05094           # 下側プーリ半径 [m]
L_off_m = 0.051         # 具治治具の固定オフセット [m]（51 mm）
p_min_bar, p_max_bar = 0.0, 7.0
num_points = 600
# 軸範囲
P_RANGE      = (0.0, 7.0)   # x軸: 圧力 p [bar]
L_RANGE_CM   = (0.0, 25.0)  # 左y軸: L [cm]
THETA_RANGE  = (0.0, 1.6)   # 右y軸: theta [rad]
# =======================

# ---- 係数行列（既存のモデル）----
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

def theta_from_L_cm(L_cm: np.ndarray, R_m: float, r_m: float, L_off_m: float):
    L_m = L_cm / 100.0 - L_off_m
    x = (R_m**2 + r_m**2 - (L_m**2)) / (2.0 * R_m * r_m)
    x = np.clip(x, -1.0, 1.0)
    return np.pi/2.0 - np.arccos(x)

# ---- 実験データ（p[bar], theta[rad]）----
p_meas = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7], dtype=float)
theta_meas = np.array([0, 0, 0.0061, 0.1765, 0.277, 0.3558, 0.4065, 0.4433, 0.4822, 0.5048, 0.5268, 0.543, 0.5538, 0.5645, 0.5752], dtype=float)

# ---- モデル曲線生成 ----
p_dense = np.linspace(p_min_bar, p_max_bar, num_points)
curves = {}
for L0 in L0_list_cm:
    L_cm_dense = L_of_p_bar(p_dense, L0)
    th_dense = theta_from_L_cm(L_cm_dense, R_m, r_m, L_off_m)
    curves[L0] = dict(p=p_dense, L=L_cm_dense, th=th_dense)

# ---- 計測点でのモデル値・残差 ----
L0 = L0_list_cm[0]
theta_model_at_meas = theta_from_L_cm(L_of_p_bar(p_meas, L0), R_m, r_m, L_off_m)
residual = theta_model_at_meas - theta_meas
rmse = float(np.sqrt(np.mean(residual**2)))
mae  = float(np.mean(np.abs(residual)))
bias = float(np.mean(residual))

print("=== Coeffs a,b,c,d,e for L0=24.5 cm ===")
print("a,b,c,d,e =", [f"{x:.6f}" for x in coeffs_from_L0(L0)])
print("=== Errors (theta model - theta meas) ===")
print(f"RMSE(rad)={rmse:.6f}, MAE(rad)={mae:.6f}, Bias(rad)={bias:.6f}")
print("p\tmeas(rad)\tmodel(rad)\tres(rad)\tres(deg)")
for p, tm, th, r in zip(p_meas, theta_meas, theta_model_at_meas, residual):
    print(f"{p:.1f}\t{tm:.6f}\t{th:.6f}\t{r:.6f}\t{np.degrees(r):.3f}")

# ---- 図1: L(p) ----
plt.figure(figsize=(7,5))
for L0, data in curves.items():
    plt.plot(data["p"], data["L"], label=f"L0={L0} cm")
plt.xlabel("Pressure p [bar]"); plt.ylabel("L [cm]"); plt.title("Model: L vs p")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*L_RANGE_CM); plt.legend(); plt.tight_layout()

# ---- 図2: theta(p)（モデル線＋実験散布）----
plt.figure(figsize=(7,5))
for L0, data in curves.items():
    plt.plot(data["p"], data["th"], label=f"Model (L0={L0} cm)")
plt.plot(p_meas, theta_meas, "o", label="Experiment")
plt.xlabel("Pressure p [bar]"); plt.ylabel("theta [rad]"); plt.title("theta vs p: Model vs Experiment")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*THETA_RANGE); plt.legend(); plt.tight_layout()

plt.show()
