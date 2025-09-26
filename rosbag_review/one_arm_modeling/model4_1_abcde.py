# fit_abCde_core.py
# ---------------------------------------------------------
# 目的: a,b,c,d,e を直接同定（L0 と L_off は固定、旧 sqrt 補正も固定）
#       データ: p[bar], theta[rad]
#       モデル: L(p) = a + b / (1 + (p/c)^d)^e - 0.009 * L0 * sqrt(p)
#       幾何:  theta(L) = pi/2 - acos( (R^2 + r^2 - (L/100 - L_off)^2) / (2 R r) )
#
# 機能:
# - SciPy最小二乗 (soft_l1) or フォールバック座標降下
# - L'(p) <= 0 の単調性ペナルティ
# - 初期値は行列 M による既存推定値
# - 前後で RMSE/MAE/Bias と 図(L,theta)を表示

import numpy as np
import matplotlib.pyplot as plt

# ===== 実験設定 =====
L0_cm = 24.5
R_m = 0.1655
r_m = 0.05094
L_off_m = 0.051     # ← 固定（ユーザー指定）

# 実験データ（p[bar], theta[rad]）
p_meas = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7], dtype=float)
theta_meas = np.array([0, 0, 0.0061, 0.1765, 0.277, 0.3558, 0.4065, 0.4433, 0.4822, 0.5048, 0.5268, 0.543, 0.5538, 0.5645, 0.5752], dtype=float)

# 可視化レンジ
P_RANGE      = (0.0, 7.0)
L_RANGE_CM   = (0.0, 25.0)
THETA_RANGE  = (0.0, 1.6)
p_min_bar, p_max_bar = 0.0, 7.0
num_points = 600

# ===== 既存の係数行列（初期値生成に使用） =====
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

# ===== モデル =====
def L_core(p_bar, a,b,c,d,e, L0_cm):
    # a,b,c,d,e は直接同定対象
    term = np.power(np.maximum(p_bar,0.0)/c, d)
    return a + b / np.power(1.0 + term, e)

def L_model(p_bar, a,b,c,d,e, L0_cm):
    # 旧補正を「そのまま」入れる
    return L_core(p_bar, a,b,c,d,e, L0_cm) - 0.009 * L0_cm * np.sqrt(np.maximum(p_bar,0.0))

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

# ===== 初期値とバウンド =====
a0,b0,c0,d0,e0 = coeffs_from_L0(L0_cm)
x0 = np.array([a0,b0,c0,d0,e0], dtype=float)

# バウンド（保守的に広め、c,d,eは正）
lb = np.array([  0.0,  -200.0,  0.05,  0.05,  0.05], dtype=float)
ub = np.array([ 50.0,   200.0, 20.0,  10.00,  10.00], dtype=float)

# ===== ベースライン（初期パラメータ） =====
theta_base = theta_from_L_cm(L_model(p_meas, *x0, L0_cm), R_m, r_m, L_off_m)
rmse0, mae0, bias0, _ = metrics(theta_base, theta_meas)
print("=== Baseline from M(L0) ===")
print(f"a0,b0,c0,d0,e0 = {[f'{v:.6f}' for v in x0]}")
print(f"RMSE={rmse0:.6f} rad, MAE={mae0:.6f} rad, Bias={bias0:.6f} rad")

# ===== 残差 + 単調性ペナルティ =====
p_dense = np.linspace(p_min_bar, p_max_bar, 300)
MONO_PENALTY = 3.0   # L'(p) > 0 に対するペナルティ重み（0で無効）

def residual_with_penalty(x):
    a,b,c,d,e = x
    L_cm = L_model(p_meas, a,b,c,d,e, L0_cm)
    th   = theta_from_L_cm(L_cm, R_m, r_m, L_off_m)
    res  = th - theta_meas
    if MONO_PENALTY>0:
        L_dense = L_model(p_dense, a,b,c,d,e, L0_cm)
        dL = np.gradient(L_dense, p_dense)
        bad = np.clip(dL, 0.0, None)   # 正の部分のみ
        res = np.concatenate([res, MONO_PENALTY * bad])
    return res

# ===== フィット（SciPy → フォールバック）=====
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
    # フォールバック: 座標降下ラインサーチ（3 ラウンド）
    x_fit = x0.copy()
    spans = (ub - lb) * 0.25
    for round_k in range(3):
        for i in range(len(x_fit)):
            # 1次元ラインサーチ
            lo = max(lb[i], x_fit[i] - spans[i])
            hi = min(ub[i], x_fit[i] + spans[i])
            grid = np.linspace(lo, hi, 21)
            best_val = x_fit[i]; best_rmse = 1e9
            for v in grid:
                xt = x_fit.copy(); xt[i] = v
                th = theta_from_L_cm(L_model(p_meas, *xt, L0_cm), R_m, r_m, L_off_m)
                rm,_,_,_ = metrics(th, theta_meas)
                if rm < best_rmse:
                    best_rmse = rm; best_val = v
            x_fit[i] = best_val
        spans *= 0.4  # だんだん狭める

# ===== 結果 =====
a,b,c,d,e = x_fit
theta_fit = theta_from_L_cm(L_model(p_meas, a,b,c,d,e, L0_cm), R_m, r_m, L_off_m)
rmse1, mae1, bias1, res1 = metrics(theta_fit, theta_meas)

print("\n=== Fitted a,b,c,d,e ===")
print(f"a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}, e={e:.6f}")
print(f"RMSE={rmse1:.6f} rad, MAE={mae1:.6f} rad, Bias={bias1:.6f} rad")

# ===== 曲線（前後比較） =====
p_plot = np.linspace(p_min_bar, p_max_bar, num_points)
L_base_plot = L_model(p_plot, *x0, L0_cm)
L_fit_plot  = L_model(p_plot, a,b,c,d,e, L0_cm)
theta_base_plot = theta_from_L_cm(L_base_plot, R_m, r_m, L_off_m)
theta_fit_plot  = theta_from_L_cm(L_fit_plot,  R_m, r_m, L_off_m)

# ===== 図 =====
plt.figure(figsize=(7,5))
plt.plot(p_plot, L_base_plot, label="L base")
plt.plot(p_plot, L_fit_plot,  label="L fitted")
plt.xlabel("Pressure p [bar]"); plt.ylabel("L [cm]")
plt.title("L vs p: base vs fitted")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*L_RANGE_CM); plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(p_plot, theta_base_plot, label="theta base")
plt.plot(p_plot, theta_fit_plot,  label="theta fitted")
plt.plot(p_meas, theta_meas, "o",  label="Experiment")
plt.xlabel("Pressure p [bar]"); plt.ylabel("theta [rad]")
plt.title("theta vs p: base vs fitted vs experiment")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*THETA_RANGE); plt.legend(); plt.tight_layout()

plt.show()
