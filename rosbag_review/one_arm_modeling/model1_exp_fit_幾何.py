# theta_pressure_with_experiment_fit_geom.py
# ------------------------------------------
# 幾何パラメータ（R_m, r_m, L_off_m）の最小二乗同定を含む完全版。
# SciPy が無い場合は粗密グリッド探索に自動フォールバック。

import numpy as np
import matplotlib.pyplot as plt

# ===== ユーザー設定 =====
L0_cm = 24.5
# 初期値（現在の設計値）
R_m_init  = 0.1655
r_m_init  = 0.05094
L_off_init = 0.051  # [m]

# 同定するパラメータの種類: "all" または "offset_only"
FIT_PARAMS = "offset_only"   # or "offset_only"

# 物理的に妥当な探索範囲（必要に応じて微調整）
bounds_all = dict(
    R_m=(0.12, 0.21),
    r_m=(0.04, 0.07),
    L_off=(0.046, 0.056)   # ±5mm 程度
)
bounds_offonly = dict(
    R_m=(R_m_init, R_m_init),
    r_m=(r_m_init, r_m_init),
    L_off=(0.015, 0.075)
)

# 描画・サンプル設定
p_min_bar, p_max_bar = 0.0, 7.0
num_points = 600
P_RANGE      = (0.0, 7.0)
L_RANGE_CM   = (0.0, 25.0)
THETA_RANGE  = (0.0, 1.6)
# =======================

# ---- 実験データ ----
p_meas = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7], dtype=float)
theta_meas = np.array([0, 0, 0.0061, 0.1765, 0.277, 0.3558, 0.4065, 0.4433, 0.4822, 0.5048, 0.5268, 0.543, 0.5538, 0.5645, 0.5752], dtype=float)

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
    return (M @ X)

def L_of_p_bar(p_bar: np.ndarray, L0_cm: float):
    a, b, c, d, e = coeffs_from_L0(L0_cm)
    term = (p_bar / c) ** d
    return a + b / (1 + term) ** e - 0.009 * L0_cm * np.sqrt(p_bar)

def theta_from_L_cm(L_cm: np.ndarray, R_m: float, r_m: float, L_off_m: float):
    L_m = L_cm / 100.0 - L_off_m
    x = (R_m**2 + r_m**2 - (L_m**2)) / (2.0 * R_m * r_m)
    x = np.clip(x, -1.0, 1.0)
    return np.pi/2.0 - np.arccos(x)

def metrics(theta_hat, theta_meas):
    residual = theta_hat - theta_meas
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae  = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    return rmse, mae, bias, residual

# ---- Base（初期パラメータ）性能 ----
L_meas_cm = L_of_p_bar(p_meas, L0_cm)
theta_base = theta_from_L_cm(L_meas_cm, R_m_init, r_m_init, L_off_init)
rmse0, mae0, bias0, res0 = metrics(theta_base, theta_meas)

print("=== Base (initial geometry) ===")
print(f"R={R_m_init:.6f} m, r={r_m_init:.6f} m, L_off={L_off_init:.6f} m")
print(f"RMSE={rmse0:.6f} rad, MAE={mae0:.6f} rad, Bias={bias0:.6f} rad")

# ---- 同定（SciPy が使えれば最小二乗、無ければ粗密グリッド探索）----
use_scipy = True
try:
    from scipy.optimize import least_squares
except Exception:
    use_scipy = False

def eval_params(R, r, Loff):
    th = theta_from_L_cm(L_meas_cm, R, r, Loff)
    return metrics(th, theta_meas)  # (rmse, mae, bias, residual)

if FIT_PARAMS == "all":
    bnds = bounds_all
else:
    bnds = bounds_offonly

if use_scipy:
    # SciPy 最小二乗
    if FIT_PARAMS == "all":
        x0 = np.array([R_m_init, r_m_init, L_off_init])
        lb = np.array([bnds["R_m"][0], bnds["r_m"][0], bnds["L_off"][0]])
        ub = np.array([bnds["R_m"][1], bnds["r_m"][1], bnds["L_off"][1]])
        def resid(x):
            th = theta_from_L_cm(L_meas_cm, x[0], x[1], x[2])
            return th - theta_meas
        sol = least_squares(resid, x0, bounds=(lb, ub))
        R_fit, r_fit, Loff_fit = sol.x
    else:
        # offset のみ
        x0 = np.array([L_off_init])
        lb = np.array([bnds["L_off"][0]])
        ub = np.array([bnds["L_off"][1]])
        def resid_off(x):
            th = theta_from_L_cm(L_meas_cm, R_m_init, r_m_init, x[0])
            return th - theta_meas
        sol = least_squares(resid_off, x0, bounds=(lb, ub))
        R_fit, r_fit, Loff_fit = R_m_init, r_m_init, sol.x[0]

else:
    # 自前の粗密グリッド探索（3段階）。SciPyが無くても動きます。
    def grid_search(bounds, center, steps):
        """ bounds: dict, center: dict, steps: list of (dR, dr, dLoff, ngrid)
            戻り値: 最良(R,r,Loff), そのRMSE
        """
        best = None
        best_rmse = 1e9
        Rmin,Rmax = bounds["R_m"]; rmin,rmax = bounds["r_m"]; lmin,lmax = bounds["L_off"]
        Rc, rc, lc = center["R_m"], center["r_m"], center["L_off"]
        for dR, dr, dL, n in steps:
            R_lo, R_hi = max(Rmin, Rc-dR), min(Rmax, Rc+dR)
            r_lo, r_hi = max(rmin, rc-dr), min(rmax, rc+dr)
            l_lo, l_hi = max(lmin, lc-dL), min(lmax, lc+dL)
            Rs = np.linspace(R_lo, R_hi, n)
            rs = np.linspace(r_lo, r_hi, n) if FIT_PARAMS=="all" else [rc]
            Ls = np.linspace(l_lo, l_hi, n)
            for R in Rs:
                for r in rs:
                    for Loff in Ls:
                        rm, _, _, _ = eval_params(R, r, Loff)
                        if rm < best_rmse:
                            best_rmse = rm
                            best = (R, r, Loff)
            Rc, rc, lc = best  # 中心更新
        return best, best_rmse

    center = dict(R_m=R_m_init, r_m=r_m_init, L_off=L_off_init)
    if FIT_PARAMS == "all":
        steps = [
            (0.03, 0.015, 0.004, 13),  # 粗
            (0.012,0.006, 0.002, 13),  # 中
            (0.004,0.003, 0.001, 13),  # 細
        ]
    else:
        steps = [
            (0.0,  0.0,   0.004, 41),  # offset のみ
            (0.0,  0.0,   0.002, 41),
            (0.0,  0.0,   0.001, 41),
        ]
    (R_fit, r_fit, Loff_fit), _ = grid_search(bnds, center, steps)

# ---- フィット後の性能 ----
theta_fit = theta_from_L_cm(L_meas_cm, R_fit, r_fit, Loff_fit)
rmse1, mae1, bias1, res1 = metrics(theta_fit, theta_meas)

print("\n=== Fitted geometry ===")
print(f"R={R_fit:.6f} m, r={r_fit:.6f} m, L_off={Loff_fit:.6f} m")
print(f"RMSE={rmse1:.6f} rad, MAE={mae1:.6f} rad, Bias={bias1:.6f} rad")

# ---- 連続圧力での曲線（前後比較）----
p_dense = np.linspace(p_min_bar, p_max_bar, num_points)
L_dense = L_of_p_bar(p_dense, L0_cm)
theta_dense_base = theta_from_L_cm(L_dense, R_m_init, r_m_init, L_off_init)
theta_dense_fit  = theta_from_L_cm(L_dense, R_fit, r_fit, Loff_fit)

# ---- 図1: L(p)（モデル）----
plt.figure(figsize=(7,5))
plt.plot(p_dense, L_dense, label=f"L0={L0_cm} cm")
plt.xlabel("Pressure p [bar]"); plt.ylabel("L [cm]"); plt.title("Model: L vs p")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*L_RANGE_CM); plt.legend(); plt.tight_layout()

# ---- 図2: theta(p)（実験＋前後のモデル）----
plt.figure(figsize=(7,5))
plt.plot(p_dense, theta_dense_base, label="Model (base)")
plt.plot(p_dense, theta_dense_fit,  label="Model (fitted)")
plt.plot(p_meas, theta_meas, "o",  label="Experiment")
plt.xlabel("Pressure p [bar]"); plt.ylabel("theta [rad]"); plt.title("theta vs p: base vs fitted vs experiment")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*THETA_RANGE); plt.legend(); plt.tight_layout()

plt.show()
