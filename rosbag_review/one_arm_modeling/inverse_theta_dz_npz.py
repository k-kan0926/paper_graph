#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVから静的マップを学習して <out-prefix>_model.npz を保存します。
  theta = f(p_sum, p_diff)   [deg]
  dz    = g(p_sum, p_diff)   [m]  （CSVに dz[m] 列がある場合だけ）

使い方例:
  python inverse_theta_dz_npz.py \
    --csv out/diff_run1_h_data.csv \
    --out-prefix out/model_k1/model_k1 \
    --degree 3 --l2 1e-4 --pmax 0.7
"""
import argparse, os
import numpy as np
import pandas as pd

def design_mat(ps, pdiff, degree=3):
    ps = np.asarray(ps); pdiff = np.asarray(pdiff)
    X = [np.ones_like(ps), pdiff, ps, pdiff*ps, pdiff**2, ps**2]
    if degree >= 3:
        X += [pdiff**3, ps**3, (pdiff**2)*ps, pdiff*(ps**2)]
    return np.column_stack(X)

def ridge_fit(X, y, l2=1e-4):
    XT = X.T
    A = XT @ X
    A.flat[::A.shape[0]+1] += l2
    b = XT @ y
    return np.linalg.solve(A, b)

def metrics(y, yhat):
    e = yhat - y
    return float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e))), float(np.mean(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--pmax", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    df = pd.read_csv(args.csv, comment="#")
    for col in ["p_sum[MPa]","p_diff[MPa]","theta[deg]"]:
        if col not in df.columns:
            raise RuntimeError(f"CSVに列 {col} がありません。")

    has_dz = ("dz[m]" in df.columns)

    mask = np.isfinite(df["p_sum[MPa]"]) & np.isfinite(df["p_diff[MPa]"]) & np.isfinite(df["theta[deg]"])
    if has_dz:
        mask &= np.isfinite(df["dz[m]"])
    sub = df.loc[mask].copy()

    ps    = sub["p_sum[MPa]"].to_numpy()
    pdiff = sub["p_diff[MPa]"].to_numpy()
    thdeg = sub["theta[deg]"].to_numpy()
    dz    = sub["dz[m]"].to_numpy() if has_dz else None

    ps_rng = (float(np.nanpercentile(ps,1)), float(np.nanpercentile(ps,99)))
    pd_rng = (float(np.nanpercentile(pdiff,1)), float(np.nanpercentile(pdiff,99)))
    sigma_ref = float(np.nanmedian(ps))

    X = design_mat(ps, pdiff, degree=args.degree)
    coef_th = ridge_fit(X, thdeg, l2=args.l2)
    th_hat = X @ coef_th
    rmse, mae, bias = metrics(thdeg, th_hat)
    print("=== theta fit ===")
    print(f" degree={args.degree}, l2={args.l2}")
    print(f" RMSE={rmse:.4f} deg, MAE={mae:.4f} deg, Bias={bias:.4f} deg")

    if has_dz:
        coef_dz = ridge_fit(X, dz, l2=args.l2)
        dz_hat = X @ coef_dz
        rmse_dz, mae_dz, bias_dz = metrics(dz, dz_hat)
        print("=== dz fit ===")
        print(f" RMSE={rmse_dz:.6f} m, MAE={mae_dz:.6f} m, Bias={bias_dz:.6f} m")
    else:
        coef_dz = np.array([])
        print("=== dz 列が無いので g は学習しません ===")

    out_path = args.out_prefix + "_model.npz"
    np.savez(out_path,
             coef_th=coef_th,
             coef_dz=coef_dz,
             degree=args.degree,
             ps_rng=np.array(ps_rng),
             pd_rng=np.array(pd_rng),
             sigma_ref=sigma_ref,
             pmax=float(args.pmax))
    print(f"[OK] saved: {out_path}")

if __name__ == "__main__":
    main()
