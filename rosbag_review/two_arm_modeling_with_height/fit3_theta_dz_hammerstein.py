#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit minimal Hammerstein dynamics on top of a mono (tanh) static model.

Model:
  dθ/dt = α*(θ_stat(Σ,Δ) - θ) + kΣ*Σ̇ + kΔ*Δ̇
where θ_stat(Σ,Δ) = A(Σ) + B(Σ)*tanh(C(Σ)*Δ).

Inputs:
  --csv         : time series with columns: p_sum[MPa], p_diff[MPa], theta[deg], (optional) time[s]
  --mono-model  : *_mono_model.npz (from fit_theta_dz_monotone_tanh.py)
  --out-prefix  : output prefix for *_hamm_model.npz
  
Usage example:
python fit3_theta_dz_hammerstein.py --csv out/diff_run1_h_data.csv --mono-model out/model_k2/model_k2_mono_model.npz --out-prefix out/model_k3/model_k3

"""
import argparse, os, json
import numpy as np
import pandas as pd  # ← エイリアス pd は pandas 専用にする（p_diff で使わない）
from scipy.optimize import lsq_linear

def poly_eval(ps, coef):
    ps = np.asarray(ps, float)
    out = np.zeros_like(ps, float); p = np.ones_like(ps, float)
    for c in coef:
        out += c*p; p *= ps
    return out

def theta_static(ps, pdiff, a, b, c):
    A = poly_eval(ps, a)
    B = np.exp(poly_eval(ps, b))
    C = np.exp(poly_eval(ps, c))
    return A + B*np.tanh(C*pdiff)

def smooth_and_deriv(x, dt, win=11, poly=3):
    x = np.asarray(x, float)
    N = len(x)
    win = min(win, (N//2)*2-1)  # odd & <= N
    if win < 5:
        dx = np.gradient(x, dt)
        return x, dx
    try:
        from scipy.signal import savgol_filter
        xs = savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
        dx = savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=dt, mode="interp")
        return xs, dx
    except Exception:
        k = win
        w = np.ones(k)/k
        xs = np.convolve(x, w, mode="same")
        dx = np.gradient(xs, dt)
        return xs, dx

def ridge_fit(X, y, l2=1e-3):
    XT = X.T
    A = XT @ X
    A.flat[::A.shape[0]+1] += l2
    b = XT @ y
    return np.linalg.solve(A, b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mono-model", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--dt", type=float, default=None, help="if omitted, infer from time[s] or fallback 0.01")
    ap.add_argument("--sg-win", type=int, default=11)
    ap.add_argument("--sg-poly", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # load mono model
    M = np.load(args.mono_model, allow_pickle=True)
    a = M["a_coef"]; b = M["b_coef"]; c = M["c_coef"]
    z_coef = M["z_coef"]
    pmax = float(M["pmax"])
    sigma_ref = float(M["sigma_ref"])
    ps_rng = tuple(M["ps_rng"]); pd_rng = tuple(M["pd_rng"])

    # load data
    df = pd.read_csv(args.csv, comment="#")  # ← ここで pandas の pd を使えるようになった
    for col in ["p_sum[MPa]","p_diff[MPa]","theta[deg]"]:
        if col not in df.columns: raise RuntimeError(f"missing column {col}")

    ps     = df["p_sum[MPa]"].to_numpy(float)
    pdiff  = df["p_diff[MPa]"].to_numpy(float)   # ← 変数名 pdiff に統一
    theta  = df["theta[deg]"].to_numpy(float)

    if args.dt is not None:
        dt = float(args.dt)
    elif "time[s]" in df.columns:
        t = df["time[s]"].to_numpy(float)
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = 0.01

    # smooth & derivatives
    ps_s,   dps   = smooth_and_deriv(ps, dt, args.sg_win, args.sg_poly)
    pdiff_s,dpdiff= smooth_and_deriv(pdiff, dt, args.sg_win, args.sg_poly)
    th_s,   dth   = smooth_and_deriv(theta, dt, args.sg_win, args.sg_poly)

    # features for linear regression: y = alpha*phi1 + kS*dps + kD*dpdiff
    th_stat = theta_static(ps_s, pdiff_s, a,b,c)
    phi1 = th_stat - th_s  # "gap"
    X = np.column_stack([phi1, dps, dpdiff])
    y = dth
    lam = args.l2
    Xd = np.vstack([X, np.sqrt(lam) * np.eye(3)])
    yd = np.hstack([y, np.zeros(3)])
    lb = np.array([1e-6, -np.inf, 0.0])
    ub = np.array([np.inf,  np.inf,  np.inf])

    sol = lsq_linear(Xd, yd, bounds=(lb, ub), method="trf", lsmr_tol='auto')
    alpha, kS, kD = sol.x.tolist()

    #coef = ridge_fit(X, y, args.l2)
    #alpha, kS, kD = coef.tolist()
    #if alpha <= 1e-6:
    #    alpha = 1e-6  # enforce positivity

    # rollout sim with measured inputs
    th_pred = th_s.copy()
    for k in range(len(th_pred)-1):
        th_stat_k = theta_static(ps_s[k], pdiff_s[k], a,b,c)
        rhs = alpha*(th_stat_k - th_pred[k]) + kS*dps[k] + kD*dpdiff[k]
        th_pred[k+1] = th_pred[k] + dt*rhs

    def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
    rmse_roll = rmse(th_pred, th_s)

    # one-step (teacher forced)
    th_hat_gap = alpha*(th_stat - th_s) + kS*dps + kD*dpdiff
    th_1step = th_s + dt*th_hat_gap
    rmse_1step = rmse(th_1step, th_s)

    print("=== Hammerstein fit (on mono kernel) ===")
    print(f" alpha={alpha:.4f} [1/s]  =>  T=1/alpha={1.0/alpha:.3f} s")
    print(f" kSigma={kS:.4f} [deg·s/MPa],  kDelta={kD:.4f} [deg·s/MPa]")
    print(f" RMSE_rollout={rmse_roll:.3f} deg,  RMSE_1step={rmse_1step:.3f} deg")

    meta = dict(alpha=alpha, T=1.0/alpha, kSigma=kS, kDelta=kD,
                rmse_rollout=rmse_roll, rmse_1step=rmse_1step, dt=dt)

    out = args.out_prefix + "_hamm_model.npz"
    np.savez(out,
             a_coef=a, b_coef=b, c_coef=c, z_coef=z_coef,
             deg=int(M["deg"]),
             ps_rng=np.array(ps_rng), pd_rng=np.array(pd_rng),
             sigma_ref=float(sigma_ref), pmax=float(pmax),
             alpha=float(alpha), kSigma=float(kS), kDelta=float(kD),
             dt=float(dt),
             dyn_report=json.dumps(meta))
    print(f"[OK] saved: {out}")

if __name__ == "__main__":
    main()
