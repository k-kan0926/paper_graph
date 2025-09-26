# model_exp_fit_corr.py
# ---------------------------------------------------------
# θ(p)の実験値に合わせて、L(p)側の補正項を同定するスクリプト。
# - 既存の a + b/(1+(p/c)^d)^e に対して、補正項 ΔL(p; θ) を
#   「置き換え」または「追加」する形でフィットできます。
# - 補正モデルは 'power', 'exp_sat', 'logistic', 'sat_power' から選択。
# - SciPyが無い環境でもグリッド探索に自動フォールバック。

import numpy as np
import matplotlib.pyplot as plt

# ===== 実験条件・幾何（必要に応じて変更） =====
L0_cm = 24.5
R_m = 0.1655
r_m = 0.05094
L_off_m = 0.039338   # ← あなたが幾何フィットで得た値（変更可）

# ===== 補正モデルの選択 =====
# 'power'    : ΔL = -K * L0 * p^α
# 'exp_sat'  : ΔL = -K * L0 * (1 - exp(-λ p))
# 'logistic' : ΔL = -S * L0 / (1 + exp(-(p - p0)/w))   （p0: 立ち上がり中心, w: 幅）
# 'sat_power': ΔL = -K * L0 * (p^α) / (1 + (p/ps)^α)   （低圧は ~p^α, 高圧は飽和）←推奨
MODEL_NAME = "sat_power"

# 既存の -0.009 L0 sqrt(p) を置き換えるか？（True=置換, False=追加）
REPLACE_BASE_CORR = True

# ===== 解析レンジ =====
p_min_bar, p_max_bar = 0.0, 7.0
num_points = 600
P_RANGE      = (0.0, 7.0)
L_RANGE_CM   = (0.0, 25.0)
THETA_RANGE  = (0.0, 1.6)

# ===== 実験データ（p[bar], theta[rad]） =====
p_meas = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7], dtype=float)
theta_meas = np.array([0, 0, 0.0061, 0.1765, 0.277, 0.3558, 0.4065, 0.4433, 0.4822, 0.5048, 0.5268, 0.543, 0.5538, 0.5645, 0.5752], dtype=float)

# ===== 既存モデルの係数行列 =====
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

# ---- 既存のL(p)（元々の補正付き/なしを分離）----
def L_base_core(p_bar, L0_cm):
    a, b, c, d, e = coeffs_from_L0(L0_cm)
    term = (p_bar / c) ** d
    return a + b / (1 + term) ** e

def L_base_with_oldcorr(p_bar, L0_cm):
    return L_base_core(p_bar, L0_cm) - 0.009 * L0_cm * np.sqrt(np.maximum(p_bar, 0.0))

# ---- 幾何変換：L[cm] -> theta[rad] ----
def theta_from_L_cm(L_cm, R_m, r_m, L_off_m):
    L_m = L_cm / 100.0 - L_off_m
    x = (R_m**2 + r_m**2 - (L_m**2)) / (2.0 * R_m * r_m)
    x = np.clip(x, -1.0, 1.0)
    return np.pi/2.0 - np.arccos(x)

# ---- 新しい補正項 ΔL(p; params) ----
def corr_power(p, L0_cm, params):
    K, alpha = params
    return -K * L0_cm * np.power(np.maximum(p, 0.0), alpha)

def corr_exp_sat(p, L0_cm, params):
    K, lam = params
    return -K * L0_cm * (1.0 - np.exp(-lam * np.maximum(p, 0.0)))

def corr_logistic(p, L0_cm, params):
    S, p0, w = params
    return -S * L0_cm / (1.0 + np.exp(-(np.maximum(p,0.0) - p0)/w))

def corr_sat_power(p, L0_cm, params):
    # ΔL = -K * L0 * (p^α) / (1 + (p/ps)^α)
    K, alpha, ps = params
    p = np.maximum(p, 0.0)
    return -K * L0_cm * (np.power(p, alpha) / (1.0 + np.power(p / ps, alpha)))

# モデル選択
def corr_fn(MODEL_NAME):
    if MODEL_NAME == "power":     return corr_power, 2
    if MODEL_NAME == "exp_sat":   return corr_exp_sat, 2
    if MODEL_NAME == "logistic":  return corr_logistic, 3
    if MODEL_NAME == "sat_power": return corr_sat_power, 3
    raise ValueError("Unknown MODEL_NAME")

# ---- L(p) 全体（補正の置換/追加を切替）----
def L_model(p_bar, L0_cm, params, replace=True):
    base = L_base_core(p_bar, L0_cm) if replace else L_base_with_oldcorr(p_bar, L0_cm)
    corr, _ = corr_fn(MODEL_NAME)
    return base + corr(p_bar, L0_cm, params)

# ---- 指標 ----
def metrics(theta_hat, theta_meas):
    res = theta_hat - theta_meas
    rmse = float(np.sqrt(np.mean(res**2)))
    mae  = float(np.mean(np.abs(res)))
    bias = float(np.mean(res))
    return rmse, mae, bias, res

# ---- ベースライン（旧補正のまま）----
L_meas_base = L_base_with_oldcorr(p_meas, L0_cm)
theta_base  = theta_from_L_cm(L_meas_base, R_m, r_m, L_off_m)
rmse0, mae0, bias0, res0 = metrics(theta_base, theta_meas)

print("=== Baseline (old sqrt correction) ===")
print(f"RMSE={rmse0:.6f} rad, MAE={mae0:.6f} rad, Bias={bias0:.6f} rad")

# ---- フィット設定（初期値・バウンド）----
corr, nparam = corr_fn(MODEL_NAME)
if MODEL_NAME == "power":
    x0   = np.array([0.009, 0.5])         # K, alpha
    lb   = np.array([0.0,   0.2])
    ub   = np.array([0.05,  1.5])
elif MODEL_NAME == "exp_sat":
    x0   = np.array([0.009, 0.8])         # K, lambda
    lb   = np.array([0.0,   0.05])
    ub   = np.array([0.05,  5.0])
elif MODEL_NAME == "logistic":
    x0   = np.array([0.50,  2.0, 0.8])    # S, p0, w
    lb   = np.array([0.0,   0.0, 0.1])
    ub   = np.array([1.50,  6.0, 3.0])
elif MODEL_NAME == "sat_power":
    x0   = np.array([0.02,  0.8, 3.0])    # K, alpha, ps
    lb   = np.array([0.0,   0.3, 0.5])
    ub   = np.array([0.10,  1.5, 8.0])

# ---- 最小二乗（SciPy→フォールバック）----
use_scipy = True
try:
    from scipy.optimize import least_squares
except Exception:
    use_scipy = False

def residual_params(x):
    L_cm = L_model(p_meas, L0_cm, x, replace=REPLACE_BASE_CORR)
    th   = theta_from_L_cm(L_cm, R_m, r_m, L_off_m)
    return th - theta_meas

if use_scipy:
    sol = least_squares(residual_params, x0, bounds=(lb, ub), loss='soft_l1', f_scale=0.05)
    x_fit = sol.x
else:
    # 粗→中→細の3段階グリッド探索
    def grid_best(xc, widths, grids):
        best_x, best_rmse = None, 1e9
        for w, n in zip(widths, grids):
            lo = np.maximum(lb, xc - w/2)
            hi = np.minimum(ub, xc + w/2)
            axes = [np.linspace(lo[i], hi[i], n) for i in range(len(xc))]
            mesh = np.meshgrid(*axes, indexing="ij")
            for idx in np.ndindex(*[n]*len(xc)):
                x = np.array([mesh[i][idx] for i in range(len(xc))])
                th = theta_from_L_cm(L_model(p_meas, L0_cm, x, replace=REPLACE_BASE_CORR), R_m, r_m, L_off_m)
                rmse, *_ = metrics(th, theta_meas)
                if rmse < best_rmse:
                    best_rmse, best_x = rmse, x.copy()
            xc = best_x
        return best_x
    x_fit = grid_best(x0, widths=[(ub-lb)*0.8, (ub-lb)*0.3, (ub-lb)*0.1], grids=[11,11,13])

# ---- フィット後の評価 ----
L_meas_fit = L_model(p_meas, L0_cm, x_fit, replace=REPLACE_BASE_CORR)
theta_fit  = theta_from_L_cm(L_meas_fit, R_m, r_m, L_off_m)
rmse1, mae1, bias1, res1 = metrics(theta_fit, theta_meas)

print(f"\n=== Fitted correction: {MODEL_NAME} (replace={REPLACE_BASE_CORR}) ===")
print("params =", ", ".join([f"{v:.6f}" for v in x_fit]))
print(f"RMSE={rmse1:.6f} rad, MAE={mae1:.6f} rad, Bias={bias1:.6f} rad")

# ---- 連続圧力での曲線 ----
p_dense = np.linspace(p_min_bar, p_max_bar, num_points)
L_dense_base = L_base_with_oldcorr(p_dense, L0_cm)
theta_dense_base = theta_from_L_cm(L_dense_base, R_m, r_m, L_off_m)

L_dense_new = L_model(p_dense, L0_cm, x_fit, replace=REPLACE_BASE_CORR)
theta_dense_new = theta_from_L_cm(L_dense_new, R_m, r_m, L_off_m)

# ---- 図: L(p) と θ(p) ----
plt.figure(figsize=(7,5))
plt.plot(p_dense, L_dense_base, label="L base (old sqrt)")
plt.plot(p_dense, L_dense_new,  label=f"L new ({MODEL_NAME})")
plt.xlabel("Pressure p [bar]"); plt.ylabel("L [cm]")
plt.title("L vs p: base vs new")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*L_RANGE_CM); plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(p_dense, theta_dense_base, label="theta base (old sqrt)")
plt.plot(p_dense, theta_dense_new,  label=f"theta new ({MODEL_NAME})")
plt.plot(p_meas, theta_meas, "o",  label="Experiment")
plt.xlabel("Pressure p [bar]"); plt.ylabel("theta [rad]")
plt.title("theta vs p: base vs new vs experiment")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*THETA_RANGE); plt.legend(); plt.tight_layout()

plt.show()
