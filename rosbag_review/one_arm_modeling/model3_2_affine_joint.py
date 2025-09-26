# model_exp_fit_corr_joint_all.py
# ---------------------------------------------------------
# 目的: θ(p) 実験値に合わせて、複数の補正モデル（power / exp_sat / logistic / sat_power）
#       を「同時に（並列に）」フィットして、結果を一括可視化する。
# 特徴:
#  - 既存スクリプトのアフィン補正(S,A[cm])・旧√p補正の置き換え/併用・単調性ペナルティを踏襲
#  - SciPy least_squares（soft_l1）で各モデルを個別に最適化（なければ簡易フォールバック）
#  - ベースラインと各モデルの RMSE/MAE/Bias を表示し、L(p) と θ(p) を重ね描画
#
# 使い方:
#  - 主要な設定は「===== どれを同定するか / 補正・可視化設定 =====」付近を調整
#  - 旧スクリプトの MODEL_NAME 依存を廃し、ALL_MODELS をループで処理

import numpy as np
import matplotlib.pyplot as plt

# ===== 実験条件・幾何（必要に応じて変更） =====
L0_cm = 24.5
R_m = 0.1655
r_m = 0.05094
L_off_init = 0.039338   # 直前の幾何フィットで得た値

# ===== どれを同定するか / 補正・可視化設定 =====
INCLUDE_AFFINE = True     # コア L に scale S と shift A[cm] を付ける
INCLUDE_CORR   = True     # 新しい補正項 ΔL を使う（各モデルごとに同定）
FIT_L_OFF      = False    # L_off も同定する（全モデルで同一設定を適用）

# 同時に評価する補正モデルの一覧
ALL_MODELS = ["power", "exp_sat", "logistic", "sat_power"]  # ←必要に応じて増減

# 既存の -0.009 L0 sqrt(p) を置き換えるか？（True=除去, False=残す）
REPLACE_BASE_CORR = True

# KやS_corrの符号を許容（旧補正を残す場合に相殺を許すなら True）
ALLOW_SIGNED_K = True

# ===== 解析レンジ・可視化設定 =====
p_min_bar, p_max_bar = 0.0, 7.0
num_points   = 600
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

# ---- 既存 L(p) ----
def L_base_core(p_bar, L0_cm):
    a, b, c, d, e = coeffs_from_L0(L0_cm)
    term = (p_bar / c) ** d
    return a + b / (1 + term) ** e

def L_oldcorr(p_bar, L0_cm):
    return -0.009 * L0_cm * np.sqrt(np.maximum(p_bar, 0.0))

# ---- 幾何: L[cm] -> theta[rad] ----
def theta_from_L_cm(L_cm, R_m, r_m, L_off_m):
    L_m = L_cm / 100.0 - L_off_m
    x = (R_m**2 + r_m**2 - (L_m**2)) / (2.0 * R_m * r_m)
    x = np.clip(x, -1.0, 1.0)
    return np.pi/2.0 - np.arccos(x)

# ---- 各補正 ΔL(p; params) ----
def corr_power(p, L0_cm, x):    # x=[K, alpha]
    K, alpha = x
    return K * L0_cm * np.power(np.maximum(p,0.0), alpha)

def corr_exp_sat(p, L0_cm, x):  # x=[K, lam]
    K, lam = x
    return K * L0_cm * (1.0 - np.exp(-lam * np.maximum(p,0.0)))

def corr_logistic(p, L0_cm, x): # x=[S, p0, w]
    S, p0, w = x
    return S * L0_cm / (1.0 + np.exp(-(np.maximum(p,0.0) - p0)/w))

def corr_sat_power(p, L0_cm, x): # x=[K, alpha, ps]
    K, alpha, ps = x
    p = np.maximum(p, 0.0)
    return K * L0_cm * (np.power(p, alpha) / (1.0 + np.power(p/ps, alpha)))

def corr_fn(name):
    if name=="power":     return corr_power,    2
    if name=="exp_sat":   return corr_exp_sat,  2
    if name=="logistic":  return corr_logistic, 3
    if name=="sat_power": return corr_sat_power,3
    raise ValueError(f"Unknown model '{name}'")

# ---- パラメタ仕様（モデル毎に生成）----
def build_param_spec(model_name):
    names = []
    x0, lb, ub = [], [], []

    # 1) affine (S, A[cm])
    if INCLUDE_AFFINE:
        names += ["S_core","A_cm"]
        x0 += [1.0, 0.0]
        lb += [0.85, -2.0]
        ub += [1.15,  2.0]

    # 2) correction params
    if INCLUDE_CORR:
        if model_name == "power":
            names += ["K","alpha"]
            x0 += [0.00, 0.8]
            kmin, kmax = (-0.08, 0.08) if ALLOW_SIGNED_K else (0.0, 0.08)
            lb += [kmin, 0.3]
            ub += [kmax, 1.8]
        elif model_name == "exp_sat":
            names += ["K","lam"]
            x0 += [0.00, 0.8]
            kmin, kmax = (-0.08, 0.08) if ALLOW_SIGNED_K else (0.0, 0.08)
            lb += [kmin, 0.05]
            ub += [kmax, 5.0]
        elif model_name == "logistic":
            names += ["S_corr","p0","w"]
            x0 += [0.00, 2.0, 0.8]
            smin, smax = (-1.5, 1.5) if ALLOW_SIGNED_K else (0.0, 1.5)
            lb += [smin, 0.0, 0.1]
            ub += [smax, 6.0, 3.0]
        elif model_name == "sat_power":
            names += ["K","alpha","ps"]
            x0 += [0.00, 0.9, 3.0]
            kmin, kmax = (-0.08, 0.08) if ALLOW_SIGNED_K else (0.0, 0.08)
            lb += [kmin, 0.3, 0.5]
            ub += [kmax, 1.8, 8.0]
        else:
            raise ValueError

    # 3) L_off
    if FIT_L_OFF:
        names += ["L_off_m"]
        x0 += [L_off_init]
        lb += [L_off_init - 0.010]  # ±10 mm
        ub += [L_off_init + 0.010]

    return np.array(names), np.array(x0, float), np.array(lb, float), np.array(ub, float)

def unpack_params(names, x, model_name):
    d = dict(zip(names, x))
    S_core = d.get("S_core", 1.0)
    A_cm   = d.get("A_cm",   0.0)
    L_off  = d.get("L_off_m", L_off_init)
    # 補正パラメタ
    if model_name=="power":
        pvec = [d.get("K",0.0), d.get("alpha",0.8)]
    elif model_name=="exp_sat":
        pvec = [d.get("K",0.0), d.get("lam",0.8)]
    elif model_name=="logistic":
        pvec = [d.get("S_corr",0.0), d.get("p0",2.0), d.get("w",0.8)]
    elif model_name=="sat_power":
        pvec = [d.get("K",0.0), d.get("alpha",0.9), d.get("ps",3.0)]
    else:
        pvec = []
    return S_core, A_cm, np.array(pvec, float), L_off

def L_model_from_params(p, L0_cm, names, x, model_name, replace=True):
    S_core, A_cm, corr_params, _ = unpack_params(names, x, model_name)
    core = L_base_core(p, L0_cm)
    base = S_core * core + A_cm
    if not replace:
        base = base + L_oldcorr(p, L0_cm)
    if INCLUDE_CORR:
        corr, _ = corr_fn(model_name)
        return base + corr(p, L0_cm, corr_params)
    else:
        return base

# ---- 指標 ----
def metrics(theta_hat, theta_meas):
    res = theta_hat - theta_meas
    rmse = float(np.sqrt(np.mean(res**2)))
    mae  = float(np.mean(np.abs(res)))
    bias = float(np.mean(res))
    return rmse, mae, bias, res

# ---- ベースライン（旧√p補正の扱いは REPLACE_BASE_CORR に従う）----
def baseline_metrics(L_off_m):
    base_L = (L_base_core(p_meas, L0_cm) if REPLACE_BASE_CORR
              else L_base_core(p_meas, L0_cm) + L_oldcorr(p_meas, L0_cm))
    th = theta_from_L_cm(base_L, R_m, r_m, L_off_m)
    return metrics(th, theta_meas)

rmse0, mae0, bias0, _ = baseline_metrics(L_off_init)
print("=== Baseline ===")
print(f"replace_oldcorr={REPLACE_BASE_CORR}, L_off={L_off_init:.6f} m")
print(f"RMSE={rmse0:.6f} rad, MAE={mae0:.6f} rad, Bias={bias0:.6f} rad")

# ---- フィット設定（SciPy → フォールバック）----
use_scipy = True
try:
    from scipy.optimize import least_squares
except Exception:
    use_scipy = False

# 単調性ペナルティ設定
MONO_PENALTY = 5.0     # 大きいほど L'(p)>0 を抑制
p_dense = np.linspace(p_min_bar, p_max_bar, 300)

def make_residual_with_penalty(names, model_name):
    def residual(x):
        # データ残差
        L_cm = L_model_from_params(p_meas, L0_cm, names, x, model_name, replace=REPLACE_BASE_CORR)
        _, _, _, L_off_m = unpack_params(names, x, model_name)
        th   = theta_from_L_cm(L_cm, R_m, r_m, L_off_m)
        res  = th - theta_meas
        # 単調性ペナルティ（L'(p) <= 0 を好む）
        L_dense = L_model_from_params(p_dense, L0_cm, names, x, model_name, replace=REPLACE_BASE_CORR)
        dL = np.gradient(L_dense, p_dense)
        bad = np.clip(dL, 0.0, None)   # 正の部分のみ
        if MONO_PENALTY>0:
            res = np.concatenate([res, MONO_PENALTY * bad])
        return res
    return residual

# ---- 各モデルを個別にフィット ----
fit_results = {}  # model_name -> dict(params=x_fit, names=names, metrics=(rmse,mae,bias), L_off, etc.)
for model_name in ALL_MODELS:
    names, x0, lb, ub = build_param_spec(model_name)
    residual = make_residual_with_penalty(names, model_name)

    if use_scipy:
        sol = least_squares(residual, x0, bounds=(lb, ub), loss='soft_l1', f_scale=0.05, max_nfev=5000)
        x_fit = sol.x
    else:
        # 簡易フォールバック
        x_fit = x0.copy()
        for it in range(3):
            for i in range(len(x0)):
                span = (ub[i]-lb[i])*0.3/(it+1)
                grid = np.linspace(np.clip(x_fit[i]-span, lb[i], ub[i]),
                                   np.clip(x_fit[i]+span, lb[i], ub[i]), 15)
                best = x_fit[i]; best_rmse = 1e9
                for val in grid:
                    x_try = x_fit.copy(); x_try[i]=val
                    L_cm  = L_model_from_params(p_meas, L0_cm, names, x_try, model_name, replace=REPLACE_BASE_CORR)
                    _, _, _, L_off_m = unpack_params(names, x_try, model_name)
                    th = theta_from_L_cm(L_cm, R_m, r_m, L_off_m)
                    rm,_,_,_ = metrics(th, theta_meas)
                    if rm < best_rmse:
                        best_rmse = rm; best = val
                x_fit[i] = best

    # 評価
    L_cm_fit = L_model_from_params(p_meas, L0_cm, names, x_fit, model_name, replace=REPLACE_BASE_CORR)
    S_core, A_cm, corr_params, L_off_fit = unpack_params(names, x_fit, model_name)
    theta_fit = theta_from_L_cm(L_cm_fit, R_m, r_m, L_off_fit)
    rmse1, mae1, bias1, res1 = metrics(theta_fit, theta_meas)

    fit_results[model_name] = {
        "names": names, "x_fit": x_fit, "rmse": rmse1, "mae": mae1, "bias": bias1,
        "L_off": L_off_fit, "S_core": S_core, "A_cm": A_cm, "corr_params": corr_params
    }

# ---- まとめ表示 ----
print("\n=== Fitted results (all models) ===")
for m in ALL_MODELS:
    info = fit_results[m]
    print(f"[{m}]  RMSE={info['rmse']:.6f}  MAE={info['mae']:.6f}  Bias={info['bias']:.6f}  L_off={info['L_off']:.6f}")
    for k, v in zip(info["names"], info["x_fit"]):
        print(f"   {k:>8s} = {v:.6f}")

# ---- 連続圧力での前後比較（図）----
p_plot = np.linspace(p_min_bar, p_max_bar, num_points)

L_base_plot = (L_base_core(p_plot, L0_cm) if REPLACE_BASE_CORR
               else L_base_core(p_plot, L0_cm) + L_oldcorr(p_plot, L0_cm))
theta_base_plot = theta_from_L_cm(L_base_plot, R_m, r_m, L_off_init)

# L(p) 図
plt.figure(figsize=(8,5))
plt.plot(p_plot, L_base_plot, label="L base", linewidth=2)
for m in ALL_MODELS:
    info = fit_results[m]
    L_fit_plot = L_model_from_params(p_plot, L0_cm, info["names"], info["x_fit"], m, replace=REPLACE_BASE_CORR)
    plt.plot(p_plot, L_fit_plot, label=f"L fitted ({m})")
plt.xlabel("Pressure p [bar]"); plt.ylabel("L [cm]")
plt.title("L vs p: base vs fitted (all models)")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*L_RANGE_CM); plt.legend(); plt.tight_layout()

# θ(p) 図
plt.figure(figsize=(8,5))
plt.plot(p_plot, theta_base_plot, label="theta base", linewidth=2)
for m in ALL_MODELS:
    info = fit_results[m]
    L_fit_plot = L_model_from_params(p_plot, L0_cm, info["names"], info["x_fit"], m, replace=REPLACE_BASE_CORR)
    theta_fit_plot = theta_from_L_cm(L_fit_plot, R_m, r_m, info["L_off"])
    plt.plot(p_plot, theta_fit_plot, label=f"theta fitted ({m})")
plt.plot(p_meas, theta_meas, "o", label="Experiment")
plt.xlabel("Pressure p [bar]"); plt.ylabel("theta [rad]")
plt.title("theta vs p: base vs fitted vs experiment (all models)")
plt.grid(True); plt.xlim(*P_RANGE); plt.ylim(*THETA_RANGE); plt.legend(); plt.tight_layout()

plt.show()
