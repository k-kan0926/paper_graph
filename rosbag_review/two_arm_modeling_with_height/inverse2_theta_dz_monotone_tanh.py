#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse mapping for the monotone-saturating model

入力:
  --model   : *_mono_model.npz （fit スクリプトの出力）
  --theta-deg : 目標角度 θ* [deg]
任意:
  --pmax      : 未指定ならモデル内の値
  --wz        : z≈0 ペナルティ重み（dz列が無い学習なら効果なし）
  --wsigma    : Σ を sigma_ref 付近に保つ重み
  --grid      : Σ の初期グリッド点数（粗探索）
  --ref-sigma : Σ の参照値を上書きしたい場合
出力:
  Sigma, Delta, p1, p2 を表示（箱拘束 [0,pmax] 適用）
使い方例:
  python inverse2_theta_dz_monotone_tanh.py --model out/model_k2/model_k2_mono_model.npz --theta-deg 20 --wz 0.3 --wsigma 0.02
"""
import argparse, os, numpy as np

def poly_eval(ps, coef):
    ps = np.asarray(ps, float)
    out = np.zeros_like(ps, dtype=float)
    p  = np.ones_like(ps, dtype=float)
    for c in coef:
        out += c * p
        p   *= ps
    return out

def A_B_C(ps, a_coef, b_coef, c_coef):
    A = poly_eval(ps, a_coef)
    B = np.exp(poly_eval(ps, b_coef))
    C = np.exp(poly_eval(ps, c_coef))
    return A,B,C

def predict_theta(ps, pd, a_coef, b_coef, c_coef):
    A,B,C = A_B_C(ps, a_coef, b_coef, c_coef)
    return A + B * np.tanh(C*pd)

def predict_z(ps, pd, z_coef):
    if z_coef is None or z_coef.size==0:
        return np.zeros_like(np.asarray(ps), dtype=float)
    return (z_coef[0]
            + z_coef[1]*ps
            + z_coef[2]*pd
            + z_coef[3]*ps*pd)

def clamp_phys(ps, pd, pmax):
    p1 = np.clip(0.5*(ps+pd), 0.0, pmax)
    p2 = np.clip(0.5*(ps-pd), 0.0, pmax)
    return (p1+p2), (p1-p2), p1, p2

def invert_closed_delta(ps, theta_star, a_coef, b_coef, c_coef):
    A,B,C = A_B_C(ps, a_coef, b_coef, c_coef)
    B = np.maximum(B, 1e-8)
    C = np.maximum(C, 1e-8)
    x = (theta_star - A)/B
    x = np.clip(x, -0.999, 0.999)  # 安定化
    pd = np.arctanh(x)/C
    # 飽和近傍の罰（|x|→1 は危険）
    sat_pen = np.maximum(0.0, np.abs(x) - 0.95)
    return float(pd), float(sat_pen)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--theta-deg", type=float, required=True)
    ap.add_argument("--pmax", type=float, default=None)
    ap.add_argument("--wz", type=float, default=0.2)
    ap.add_argument("--wsigma", type=float, default=0.02)
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--ref-sigma", type=float, default=None)
    args = ap.parse_args()

    data = np.load(args.model, allow_pickle=True)
    a = data["a_coef"]; b = data["b_coef"]; c = data["c_coef"]
    z_coef = data["z_coef"]
    deg = int(data["deg"])
    ps_rng = tuple(data["ps_rng"])
    pd_rng = tuple(data["pd_rng"])
    sigma_ref = float(data["sigma_ref"])
    pmax = float(args.pmax) if args.pmax is not None else float(data["pmax"])
    if args.ref_sigma is not None:
        sigma_ref = float(args.ref_sigma)

    # Σ の探索範囲（学習レンジ ∩ 物理レンジ）
    ps_lo = max(0.0, ps_rng[0])
    ps_hi = min(2.0*pmax, ps_rng[1])
    if ps_hi <= ps_lo + 1e-6:
        ps_lo, ps_hi = 0.0, 2.0*pmax

    # 1) 粗探索（Σ グリッド）
    best = None
    for ps in np.linspace(ps_lo, ps_hi, max(args.grid, 11)):
        pd, sat_pen = invert_closed_delta(ps, args.theta_deg, a,b,c)
        # 箱拘束違反に強罰
        p1 = 0.5*(ps+pd); p2 = 0.5*(ps-pd)
        box = 0.0
        if p1<0: box += -p1
        if p2<0: box += -p2
        if p1>pmax: box += (p1-pmax)
        if p2>pmax: box += (p2-pmax)

        z_est = predict_z(ps, pd, z_coef)
        th_err = predict_theta(ps, pd, a,b,c) - args.theta_deg
        cost = (th_err**2) + args.wz*(z_est**2) + args.wsigma*((ps - sigma_ref)**2)
        cost += 1e6*box + 10.0*float(sat_pen)  # 安全化
        if (best is None) or (cost < best[0]):
            best = (cost, ps, pd)

    _, ps, pd = best

    # 2) 近傍微調整（Σ を小域で1D 最適化）
    for _ in range(8):
        span = 0.05*(ps_hi - ps_lo)
        cand = []
        for s in [ps-span, ps, ps+span]:
            s = float(np.clip(s, ps_lo, ps_hi))
            d, sat_pen = invert_closed_delta(s, args.theta_deg, a,b,c)
            p1 = 0.5*(s+d); p2 = 0.5*(s-d)
            box = 0.0
            if p1<0: box += -p1
            if p2<0: box += -p2
            if p1>pmax: box += (p1-pmax)
            if p2>pmax: box += (p2-pmax)
            z_est = predict_z(s, d, z_coef)
            th_err = predict_theta(s, d, a,b,c) - args.theta_deg
            cost = (th_err**2) + args.wz*(z_est**2) + args.wsigma*((s - sigma_ref)**2)
            cost += 1e6*box + 10.0*float(sat_pen)
            cand.append((cost, s, d))
        cand.sort(key=lambda t: t[0])
        _, ps, pd = cand[0]

    # 最終投影（箱拘束）
    ps, pd, p1, p2 = clamp_phys(ps, pd, pmax)

    # 参考出力
    th_est = predict_theta(ps, pd, a,b,c)
    if z_coef.size>0:
        z_est = predict_z(ps, pd, z_coef)
    else:
        z_est = 0.0

    print("=== inverse (monotone-tanh) ===")
    print(f" theta* : {args.theta_deg:.3f} deg")
    print(f" Sigma* : {ps:.4f} MPa")
    print(f" Delta* : {pd:.4f} MPa")
    print(f" -> p1  : {p1:.4f} MPa, p2: {p2:.4f} MPa  (box [0,{pmax}])")
    print(f" check  : th_est={th_est:.3f} deg, z_est={z_est:.5f} m")

if __name__ == "__main__":
    main()
