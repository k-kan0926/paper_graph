#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monotone-Saturating static fit for antagonistic PAMs
  theta(Σ,Δ) = A(Σ) + B(Σ) * tanh(C(Σ)*Δ)   [deg]
  z(Σ,Δ)     ≈ α0 + α1 Σ + α2 Δ + α3 ΣΔ     [m]   (optional)

CSV 必須列:
  p_sum[MPa], p_diff[MPa], theta[deg]
任意列:
  dz[m]

出力:
  <out-prefix>_mono_model.npz に以下を保存:
    a_coef, b_coef, c_coef  （A,B,C の Σ 多項式係数）
    z_coef                   （z の線形係数。無い場合は空配列）
    deg                      （A,B,C の多項式次数）
    pmax, sigma_ref
    ps_rng, pd_rng           （学習レンジの指標）
    fit_report               （RMSE 等）

使い方例:
  python fit2_theta_dz_monotone_tanh.py --csv out/diff_run1_h_data.csv --out-prefix out/model_k2/model_k2 --deg 2 --lamA 1e-4 --lamB 1e-3 --lamC 1e-3 --z-l2 1e-3 --pmax 0.7
"""
import argparse, os, json
import numpy as np
import pandas as pd
from math import isfinite

try:
    from scipy.optimize import least_squares
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def poly_eval(ps, coef):
    # coef: [c0, c1, ..., c_deg]  -> sum_k coef[k]*ps^k
    ps = np.asarray(ps, float)
    out = np.zeros_like(ps, dtype=float)
    p  = np.ones_like(ps, dtype=float)
    for c in coef:
        out += c * p
        p   *= ps
    return out

def theta_model(ps, pd, a_coef, b_coef, c_coef):
    A = poly_eval(ps, a_coef)
    B = np.exp(poly_eval(ps, b_coef))  # > 0
    C = np.exp(poly_eval(ps, c_coef))  # > 0
    return A + B * np.tanh(C * pd)

def design_z(ps, pd):
    # 線形基底: [1, Σ, Δ, ΣΔ]
    return np.column_stack([np.ones_like(ps), ps, pd, ps*pd])

def ridge_fit(X, y, l2=1e-3):
    XT = X.T
    A = XT @ X
    A.flat[::A.shape[0]+1] += l2
    b = XT @ y
    return np.linalg.solve(A, b)

def finite_ok(*arrs):
    for a in arrs:
        if not np.all(np.isfinite(a)): return False
    return True

def init_params(ps, pd, th, deg):
    # A 初期: θ を Σ のみでリッジ回帰（Δを無視）
    X = np.column_stack([ps**k for k in range(deg+1)])
    a0 = ridge_fit(X, th, 1e-6)
    # 残差スケールから B,C 初期
    res = th - (X @ a0)
    B0 = max(np.median(np.abs(res)) * 1.2, 1.0)    # deg スケール
    C0 = 1.0 / max(np.median(np.abs(pd))+1e-3, 0.1) # [1/MPa]
    # b_coef, c_coef は対数空間で定数項のみ
    b0 = np.zeros(deg+1); b0[0] = np.log(B0)
    c0 = np.zeros(deg+1); c0[0] = np.log(C0)
    return a0, b0, c0

def pack(a,b,c): return np.concatenate([a,b,c])
def unpack(p, deg):
    n = deg+1
    return p[:n], p[n:2*n], p[2*n:3*n]

def residual_theta(p, ps, pd, th, deg, lamA, lamB, lamC):
    a,b,c = unpack(p, deg)
    pred = theta_model(ps, pd, a,b,c)
    res  = pred - th
    # 正則化（各係数に同次元 L2）
    regA = np.sqrt(lamA)*a
    regB = np.sqrt(lamB)*b
    regC = np.sqrt(lamC)*c
    return np.hstack([res, regA, regB, regC])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--deg", type=int, default=2)
    ap.add_argument("--lamA", type=float, default=1e-4)
    ap.add_argument("--lamB", type=float, default=1e-3)
    ap.add_argument("--lamC", type=float, default=1e-3)
    ap.add_argument("--z-l2", type=float, default=1e-3)
    ap.add_argument("--pmax", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    df = pd.read_csv(args.csv, comment="#")

    for col in ["p_sum[MPa]","p_diff[MPa]","theta[deg]"]:
        if col not in df.columns:
            raise RuntimeError(f"列 {col} がありません。")
    has_z = ("dz[m]" in df.columns)

    ps = df["p_sum[MPa]"].to_numpy(float)
    pdiff = df["p_diff[MPa]"].to_numpy(float)
    th = df["theta[deg]"].to_numpy(float)
    z  = df["dz[m]"].to_numpy(float) if has_z else None

    # 有効データ抽出
    mask = np.isfinite(ps) & np.isfinite(pdiff) & np.isfinite(th)
    if has_z: mask &= np.isfinite(z)
    ps, pdiff, th = ps[mask], pdiff[mask], th[mask]
    if has_z: z = z[mask]

    ps_rng = (float(np.nanpercentile(ps,1)), float(np.nanpercentile(ps,99)))
    pd_rng = (float(np.nanpercentile(pdiff,1)), float(np.nanpercentile(pdiff,99)))
    sigma_ref = float(np.nanmedian(ps))

    # 初期値
    a0,b0,c0 = init_params(ps, pdiff, th, args.deg)
    p0 = pack(a0,b0,c0)

    if not HAVE_SCIPY:
        raise RuntimeError("scipy が必要です（least_squares を使用）。")

    # θ フィット（ロバスト損失）
    opt = least_squares(
        residual_theta, p0,
        args=(ps, pdiff, th, args.deg, args.lamA, args.lamB, args.lamC),
        loss="soft_l1", f_scale=5.0, max_nfev=5000
    )
    a_hat, b_hat, c_hat = unpack(opt.x, args.deg)
    th_hat = theta_model(ps, pdiff, a_hat,b_hat,c_hat)

    # z フィット（任意）
    if has_z:
        Xz = design_z(ps, pdiff)
        z_coef = ridge_fit(Xz, z, args.z_l2)
        z_hat  = Xz @ z_coef
        rmse_z = float(np.sqrt(np.mean((z_hat - z)**2)))
        mae_z  = float(np.mean(np.abs(z_hat - z)))
    else:
        z_coef = np.array([])
        rmse_z = mae_z = None

    # メトリクス表示
    def mse(a,b): return np.mean((a-b)**2)
    rmse_th = float(np.sqrt(mse(th_hat, th)))
    mae_th  = float(np.mean(np.abs(th_hat - th)))
    # tanh 引数の飽和度（|x|>=1 近傍）
    A = poly_eval(ps, a_hat)
    B = np.exp(poly_eval(ps, b_hat))
    x = (th - A) / np.maximum(B, 1e-8)
    sat_ratio = float(np.mean(np.abs(x) >= 0.98))

    print("=== Monotone-Saturating Fit (theta) ===")
    print(f" deg={args.deg}, lamA={args.lamA}, lamB={args.lamB}, lamC={args.lamC}")
    print(f" RMSE_theta={rmse_th:.3f} deg, MAE_theta={mae_th:.3f} deg, sat(|x|>=0.98)={sat_ratio*100:.1f}%")
    if has_z:
        print("=== z fit (linear in Σ,Δ) ===")
        print(f" RMSE_z={rmse_z:.5f} m, MAE_z={mae_z:.5f} m")
    else:
        print("=== z 列が無いので g は学習しません ===")

    # 保存
    out_path = args.out_prefix + "_mono_model.npz"
    meta = dict(
        rmse_theta=rmse_th, mae_theta=mae_th,
        rmse_z=rmse_z, mae_z=mae_z,
        sat_ratio=sat_ratio, fun=opt.cost, nfev=opt.nfev, status=int(opt.status)
    )
    np.savez(out_path,
             a_coef=a_hat, b_coef=b_hat, c_coef=c_hat,
             z_coef=z_coef,
             deg=int(args.deg),
             ps_rng=np.array(ps_rng), pd_rng=np.array(pd_rng),
             sigma_ref=float(sigma_ref),
             pmax=float(args.pmax),
             fit_report=json.dumps(meta))
    print(f"[OK] saved: {out_path}")

if __name__ == "__main__":
    main()
