# model_abcde_regsmooth.py
# ---------------------------------------------------------
# a,b,c,d,e を同定（L0, L_off 固定、旧 sqrt 補正はそのまま）
# 追加: L2正則化（M(L0) 初期値への弱い事前）
#     + 滑らかさ（2階差分）ペナルティ
#     + 単調性 L'(p) <= 0 ペナルティ
#
# 目的: 0–7 bar での高精度を保ちつつ、外挿や他条件での頑健性を向上

import numpy as np
import matplotlib.pyplot as plt

# --- 実験設定 ---
L0_cm = 24.5
R_m = 0.1655
r_m = 0.05094
L_off_m = 0.051   # 固定

p_meas = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7], float)
theta_meas = np.array([0, 0, 0.0061, 0.1765, 0.277, 0.3558, 0.4065, 0.4433, 0.4822, 0.5048, 0.5268, 0.543, 0.5538, 0.5645, 0.5752], float)

# --- 可視化レンジ ---
P_RANGE      = (0.0, 7.0)
L_RANGE_CM   = (0.0, 25.0)
THETA_RANGE  = (0.0, 1.6)
p_min_bar, p_max_bar = 0.0, 7.0
num_points = 600

# --- 初期値生成用 M 行列 ---
M = np.array([
    [ 0.4351, 0.0,     0.0183,  -0.0003  ],
    [ 0.5649, 0.0,    -0.0183,   0.0003  ],
    [-0.0141, 0.0,     0.0031,  -0.00006 ],
    [ 0.5487, 0.0,    -0.0136,   0.00007 ],
    [ 0.0,    0.3694,  0.0,      0.0     ],
])

def coeffs_from_L0(L0_cm: float):
    X = np.array([L0_cm, L0_cm**-0.248, L0_cm**2, L0_cm**3])
    return (M @ X)  # a,b,c,d,e

# --- モデル ---
def L_core(p_bar, a,b,c,d,e):
    term = np.power(np.maximum(p_bar,0.0)/c, d)
    return a + b / np.power(1.0 + term, e)

def L_model(p_bar, a,b,c,d,e, L0_cm):
    # 旧 sqrt 補正はそのまま
    return L_core(p_bar, a,b,c,d,e) - 0.009 * L0_cm * np.sqrt(np.maximum(p_bar,0.0))

def theta_from_L_cm(L_cm, R_m, r_m, L_off_m):
    L_m = L_cm/100.0 - L_off_m
    x = (R_m**2 + r_m**2 - L_m**2) / (2.0 * R_m * r_m)
    x = np.clip(x, -1.0, 1.0)
    return np.pi/2.0 - np.arccos(x)

def metrics(theta_hat, theta_meas):
    res = theta_hat - theta_meas
    rmse = float(np.sqrt(np.mean(res**2)))
    mae  = float(np.mean(np.abs(res)))
    bias = float(np.mean(res))
    return rmse, mae, bias, res

# --- 初期値 & バウンド（e の下限は少し上げる）---
a0,b0,c0,d0,e0 = coeffs_from_L0(L0_cm)
x0 = np.array([a0,b0,c0,d0,e0], float)

lb = np.array([  0.0,  -200.0,  0.05,  0.05,  0.10], float)  # e>=0.10
ub = np.array([ 50.0,   200.0, 20.0,  10.00,  10.00], float)

# --- ベースライン ---
theta_base = theta_from_L_cm(L_model(p_meas, *x0, L0_cm), R_m, r_m, L_off_m)
rmse0, mae0, bias0, _ = metrics(theta_base, theta_meas)
print("=== Baseline from M(L0) ===")
print(f"a0,b0,c0,d0,e0 = {[f'{v:.6f}' for v in x0]}")
print(f"RMSE={rmse0:.6f} rad, MAE={mae0:.6f} rad, Bias={bias0:.6f} rad")

# --- ペナルティ設定 ---
p_dense = np.linspace(p_min_bar, p_max_bar, 260)
MONO_W = 5.0   # 単調性（dL/dp>0）を叩く重み
SMTH_W = 0.5   # 滑らかさ（2階差分）重み
L2_W   = 1e-2  # 係数の L2 正則化重み（初期値からのずれ）

# 係数ごとのスケール感（正則化の単位調整）
scale = np.array([5.0, 5.0, 1.0, 2.0, 0.5], float)

def residual_with_penalty(x):
    a,b,c,d,e = x
    # データ残差（角度）
    L_cm = L_model(p_meas, a,b,c,d,e, L0_cm)
    th   = theta_from_L_cm(L_cm, R_m, r_m, L_off_m)
    res  = th - theta_meas

    # 単調性：dL/dp > 0 を罰する
    L_dense = L_model(p_dense, a,b,c,d,e, L0_cm)
    dL = np.gradient(L_dense, p_dense)
    mono_pen = MONO_W * np.clip(dL, 0.0, None)  # 正の部分のみ

    # 滑らかさ：離散2階差分
    ddL = np.diff(L_dense, n=2)
    smth_pen = SMTH_W * ddL  # 2階差分の L2

    # 係数 L2 正則化（初期値からのズレを抑える）
    l2_pen = np.sqrt(L2_W) * ((x - x0) / scale)

    return np.concatenate([res, mono_pen, smth_pen, l2_pen])

# --- フィット ---
use_scipy = True
try:
    from scipy.optimize import least_squares
except Exception:
    use_scipy = False

if use_scipy:
    sol = least_squares(residual_with_penalty, x0, bounds=(lb, ub),
                        loss='soft_l1', f_scale=0.05, max_nfev=5000)
    x_fit = sol.x
else:
    # フォールバック（簡易座標降下）
    x_fit = x0.copy()
    spans = (ub - lb) * 0.25
    for round_k in range(3):
        for i in range(len(x_fit)):
            lo = max(lb[i], x_fit[i] - spans[i])
            hi = min(ub[i], x_fit[i] + spans[i])
            grid = np.linspace(lo, hi, 25)
            best_val, best_rmse = x_fit[i], 1e9
            for v in grid:
                xt = x_fit.copy(); xt[i]=v
                L_cm = L_model(p_meas, *xt, L0_cm)
                th   = theta_from_L_cm(L_cm, R_m, r_m, L_off_m)
                rm,_,_,_ = metrics(th, theta_meas)
                if rm < best_rmse:
                    best_rmse, best_val = rm, v
            x_fit[i] = best_val
        spans *= 0.4

# --- 結果 ---
a,b,c,d,e = x_fit
theta_fit = theta_from_L_cm(L_model(p_meas, a,b,c,d,e, L0_cm), R_m, r_m, L_off_m)
rmse1, mae1, bias1, _ = metrics(theta_fit, theta_meas)

print("\n=== Fitted a,b,c,d,e (regsmooth) ===")
print(f"a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}, e={e:.6f}")
print(f"RMSE={rmse1:.6f} rad, MAE={mae1:.6f} rad, Bias={bias1:.6f} rad")

# --- 曲線（前後比較） ---
p_plot = np.linspace(p_min_bar, p_max_bar, num_points)
L_base_plot = L_model(p_plot, *x0, L0_cm)
L_fit_plot  = L_model(p_plot, a,b,c,d,e, L0_cm)
theta_base_plot = theta_from_L_cm(L_base_plot, R_m, r_m, L_off_m)
theta_fit_plot  = theta_from_L_cm(L_fit_plot,  R_m, r_m, L_off_m)

plt.figure(figsize=(7,5))
plt.plot(p_plot, L_base_plot, label="L base")
plt.plot(p_plot, L_fit_plot,  label="L fitted (regsmooth)")
plt.xlabel("Pressure p [bar]"); plt.ylabel("L [cm]")
plt.title("L vs p: base vs fitted (regsmooth)")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*L_RANGE_CM); plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(p_plot, theta_base_plot, label="theta base")
plt.plot(p_plot, theta_fit_plot,  label="theta fitted (regsmooth)")
plt.plot(p_meas, theta_meas, "o",  label="Experiment")
plt.xlabel("Pressure p [bar]"); plt.ylabel("theta [rad]")
plt.title("theta vs p: base vs fitted vs experiment")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*THETA_RANGE); plt.legend(); plt.tight_layout()

plt.show()
