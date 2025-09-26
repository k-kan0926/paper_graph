#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hammerstein fit on top of mono (tanh) static model, with:
- input delay search (d in a grid),
- bounded least-squares (alpha >= 0, kDelta >= kdelta_min),
- optional column standardization for features,
- model selection by rollout RMSE.

Dynamics (continuous time -> Euler):
  dθ/dt = α*(θ_stat(Σ,Δ) - θ) + kΣ*Σ̇ + kΔ*Δ̇
In discrete rollout we use inputs applied with delay d:
  at stage k, plant uses (Σ,Δ) at index k-d (history if k<d).

Usage:
  python fit3c_theta_dz_hammerstein_delay.py --csv out/diff_run1_h_data.csv --mono-model out/model_k2/model_k2_mono_model.npz --out-prefix out/model_k3c/model_k3c --delay-grid 0,1,2,3 --l2 5e-3 --sg-win 17 --kdelta-min -2.0 --standardize-x
"""
import argparse, os, json
import numpy as np
import pandas as pd

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

def smooth_and_deriv(x, dt, win=17, poly=3):
    x = np.asarray(x, float)
    N = len(x)
    win = min(win, (N//2)*2-1)
    if win < 5:
        return x, np.gradient(x, dt)
    try:
        from scipy.signal import savgol_filter
        xs = savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
        dx = savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=dt, mode="interp")
        return xs, dx
    except Exception:
        k = win; w = np.ones(k)/k
        xs = np.convolve(x, w, mode="same")
        dx = np.gradient(xs, dt)
        return xs, dx

def build_features(ps_s, pdiff_s, th_s, dps, dpdiff, d, dth, a,b,c, dt):
    """
    Align y = dθ/dt at time k with inputs at time (k-d).
    Need (k-d-1)>=0 for rate terms -> k starts from d+1.
    Returns X (N,3), y (N,), start_idx used.
    """
    N = len(ps_s)
    start = d + 1
    idx = np.arange(start, N)
    ks  = idx           # for θ, dθ/dt
    ku  = idx - d       # for θ_stat(Σ,Δ), rates
    kup = idx - d - 1   # for previous (rates)

    phi1 = theta_static(ps_s[ku], pdiff_s[ku], a,b,c) - th_s[ks]
    dS   = (ps_s[ku]    - ps_s[kup]) / dt
    dD   = (pdiff_s[ku] - pdiff_s[kup]) / dt
    y    = dth[ks]
    X = np.column_stack([phi1, dS, dD])  # units -> y is [deg/s]
    return X, y, start

def rollout_rmse(ps_s, pdiff_s, th_s, d, a,b,c, alpha,kS,kD, dt):
    N = len(ps_s)
    start = d + 1
    th_pred = th_s.copy()
    for k in range(start, N-1):
        ku = k - d; kup = k - d - 1
        ths = theta_static(ps_s[ku], pdiff_s[ku], a,b,c)
        dS  = (ps_s[ku]    - ps_s[kup]) / dt
        dD  = (pdiff_s[ku] - pdiff_s[kup]) / dt
        rhs = alpha*(ths - th_pred[k]) + kS*dS + kD*dD
        th_pred[k+1] = th_pred[k] + dt*rhs
    err = th_pred[start:] - th_s[start:]
    return float(np.sqrt(np.mean(err**2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mono-model", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--sg-win", type=int, default=17)
    ap.add_argument("--sg-poly", type=int, default=3)
    ap.add_argument("--l2", type=float, default=5e-3)
    ap.add_argument("--delay-grid", type=str, default="0", help="comma-separated e.g. 0,1,2,3")
    ap.add_argument("--kdelta-min", type=float, default=-2.0, help="lower bound for kDelta (deg·s/MPa)")
    ap.add_argument("--standardize-x", action="store_true", help="standardize columns of X")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    M = np.load(args.mono_model, allow_pickle=True)
    a = M["a_coef"]; b = M["b_coef"]; c = M["c_coef"]
    z_coef = M["z_coef"] if "z_coef" in M.files else np.array([])
    pmax = float(M["pmax"]); sigma_ref = float(M["sigma_ref"])
    ps_rng = tuple(M["ps_rng"]); pd_rng = tuple(M["pd_rng"])

    df = pd.read_csv(args.csv, comment="#")
    for col in ["p_sum[MPa]","p_diff[MPa]","theta[deg]"]:
        if col not in df.columns:
            raise RuntimeError(f"missing column {col}")

    ps     = df["p_sum[MPa]"].to_numpy(float)
    pdiff  = df["p_diff[MPa]"].to_numpy(float)
    theta  = df["theta[deg]"].to_numpy(float)

    if args.dt is not None:
        dt = float(args.dt)
    elif "time[s]" in df.columns:
        t = df["time[s]"].to_numpy(float)
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = 0.01

    ps_s, dps = smooth_and_deriv(ps, dt, args.sg_win, args.sg_poly)
    pd_s, dpd = smooth_and_deriv(pdiff, dt, args.sg_win, args.sg_poly)
    th_s, dth = smooth_and_deriv(theta, dt, args.sg_win, args.sg_poly)

    try:
        from scipy.optimize import lsq_linear
    except Exception:
        raise RuntimeError("scipy が必要です（lsq_linear を使用）。")

    delays = [int(x.strip()) for x in args.delay_grid.split(",") if x.strip()!=""]
    results = []
    best = None

    for d in delays:
        if d < 0: continue
        X_raw, y, start = build_features(ps_s, pd_s, th_s, dps, dpd, d, dth, a,b,c, dt)

        # standardize if requested
        if args.standardize_x:
            scales = np.std(X_raw, axis=0, ddof=1)
            scales = np.where(scales<1e-8, 1.0, scales)
            Xs = X_raw / scales
        else:
            scales = np.ones(3, float)
            Xs = X_raw

        # Ridge via dummy rows (Tikhonov)
        lam = args.l2
        X_aug = np.vstack([Xs, np.sqrt(lam)*np.eye(3)])
        y_aug = np.hstack([y, np.zeros(3)])

        # Bounds: alpha>=0, kSigma free, kDelta >= kdelta_min
        lb = [0.0, -np.inf, args.kdelta_min]
        ub = [np.inf, np.inf, np.inf]
        sol = lsq_linear(X_aug, y_aug, bounds=(lb,ub), method="trf", lsmr_tol='auto', verbose=0)
        w_scaled = sol.x
        alpha, kS, kD = (w_scaled / scales).tolist()

        rmse_roll = rollout_rmse(ps_s, pd_s, th_s, d, a,b,c, alpha,kS,kD, dt)
        results.append((d, alpha, kS, kD, rmse_roll))
        if (best is None) or (rmse_roll < best[-1]):
            best = (d, alpha, kS, kD, rmse_roll)

    print("=== Hammerstein fit (delay grid search) ===")
    for d, A,Ks,Kd,R in results:
        print(f" d={d}: alpha={A:.4f} [1/s] (T={1.0/max(A,1e-9):.3f}s), kΣ={Ks:.4f}, kΔ={Kd:.4f}, RMSE_rollout={R:.3f} deg")
    d, alpha, kS, kD, rmse = best
    print(f" -> selected d={d}  (RMSE_rollout={rmse:.3f} deg)")

    meta = dict(alpha=alpha, T=1.0/alpha if alpha>0 else float('inf'),
                kSigma=kS, kDelta=kD, delay=d, rmse_rollout=rmse,
                dt=dt, sg_win=args.sg_win, l2=args.l2, kdelta_min=args.kdelta_min,
                standardize_x=bool(args.standardize_x))

    out = args.out_prefix + "_hamm_model.npz"
    np.savez(out,
             a_coef=a, b_coef=b, c_coef=c, z_coef=z_coef,
             deg=int(M["deg"]),
             ps_rng=np.array(ps_rng), pd_rng=np.array(pd_rng),
             sigma_ref=float(sigma_ref), pmax=float(pmax),
             alpha=float(alpha), kSigma=float(kS), kDelta=float(kD),
             delay=int(d), dt=float(dt),
             dyn_report=json.dumps(meta))
    print(f"[OK] saved: {out}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
