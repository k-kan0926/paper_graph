#!/usr/bin/env python3
#python inverse1_theta_dz_constraint.py --model out/model_k1/model_k1_model.npz --theta-deg 20 --wz 0.3 --wsigma 0.02 
# -*- coding: utf-8 -*-
import argparse, os, numpy as np

def design_mat(ps, pd, degree=3):
    ps = np.asarray(ps); pd = np.asarray(pd)
    X = [np.ones_like(ps), pd, ps, pd*ps, pd**2, ps**2]
    if degree >= 3:
        X += [pd**3, ps**3, (pd**2)*ps, pd*(ps**2)]
    return np.column_stack(X)

def predict_theta(ps, pd, coef_th, degree):
    return design_mat(ps, pd, degree) @ coef_th

def predict_dz(ps, pd, coef_dz, degree):
    if coef_dz is None: return np.zeros_like(np.asarray(ps), dtype=float)
    return design_mat(ps, pd, degree) @ coef_dz

def clamp_phys(ps, pd, pmax):
    # 物理箱拘束（p1,p2 in [0,pmax]）へ投影しなおして整合
    p1 = np.clip(0.5*(ps+pd), 0.0, pmax)
    p2 = np.clip(0.5*(ps-pd), 0.0, pmax)
    return (p1+p2), (p1-p2), p1, p2

def have_scipy():
    try:
        import scipy  # noqa
        return True
    except Exception:
        return False

def solve_least_squares(theta_ref_deg, coef_th, coef_dz, degree, ps_rng, pd_rng, pmax,
                        w_theta=1.0, w_z=0.2, w_sigma=0.02, sigma_ref=None):
    from scipy.optimize import least_squares
    # 変数: x = [ps, pd]
    lo = [max(0.0, ps_rng[0]), max(-pmax, pd_rng[0])]
    hi = [min(2*pmax, ps_rng[1]), min(+pmax, pd_rng[1])]
    x0 = np.array([np.clip(sigma_ref if sigma_ref is not None else 0.5*(lo[0]+hi[0]), lo[0], hi[0]),
                   0.0], dtype=float)
    def resid(x):
        ps, pd = x
        th = predict_theta(ps, pd, coef_th, degree)
        dz = predict_dz(ps, pd, coef_dz, degree)
        r = []
        r.append(np.sqrt(w_theta)*(th - theta_ref_deg))
        r.append(np.sqrt(w_z)*(dz - 0.0))
        if sigma_ref is not None:
            r.append(np.sqrt(w_sigma)*(ps - sigma_ref))
        # 物理箱拘束（やや緩くペナルティ）
        p1 = 0.5*(ps+pd); p2 = 0.5*(ps-pd)
        box = 0.0
        for v in (p1, p2):
            if v < 0.0: box += (-v)
            if v > pmax: box += (v - pmax)
        if box>0:
            r.append(1e3*box)
        return np.array(r, dtype=float)
    out = least_squares(resid, x0, bounds=(lo,hi), method="trf", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
    ps, pd = out.x
    # 最終的に物理箱へ投影
    ps, pd, p1, p2 = clamp_phys(ps, pd, pmax)
    return ps, pd, p1, p2, out.cost

def root_in_pd_for_theta(ps, theta_ref_deg, coef_th, degree, pd_lo, pd_hi, iters=50):
    # f(ps,pd)=theta* を pd で解く（二分法）。外れる場合は最寄り端に張り付く
    lo, hi = pd_lo, pd_hi
    f_lo = predict_theta(ps, lo, coef_th, degree) - theta_ref_deg
    f_hi = predict_theta(ps, hi, coef_th, degree) - theta_ref_deg
    if np.isnan(f_lo) or np.isnan(f_hi):
        return None, False
    # bracket 無し → 端で近い方を採用
    if f_lo==0: return lo, True
    if f_hi==0: return hi, True
    if f_lo * f_hi > 0:
        # 端のどちらが近いか
        err_lo = abs(f_lo); err_hi = abs(f_hi)
        return (lo if err_lo < err_hi else hi), False
    # bracket あり → 二分法
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        f_mid = predict_theta(ps, mid, coef_th, degree) - theta_ref_deg
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5*(lo+hi), True

def solve_lexi_theta_first(theta_ref_deg, coef_th, coef_dz, degree, ps_rng, pd_rng, pmax,
                           w_z=0.2, w_sigma=0.02, sigma_ref=None):
    # ps の探索範囲（学習範囲 ∩ 物理範囲）
    ps_lo = max(0.0, ps_rng[0]); ps_hi = min(2*pmax, ps_rng[1])
    pd_lo = max(-pmax, pd_rng[0]); pd_hi = min(+pmax, pd_rng[1])

    # 1) 粗探索
    N = 121
    best = None
    for ps in np.linspace(ps_lo, ps_hi, N):
        pd, ok = root_in_pd_for_theta(ps, theta_ref_deg, coef_th, degree, pd_lo, pd_hi)
        th_err = predict_theta(ps, pd, coef_th, degree) - theta_ref_deg
        dz = predict_dz(ps, pd, coef_dz, degree)
        cost = (1.0*th_err**2) + (w_z*dz**2) + (w_sigma*((ps - (sigma_ref if sigma_ref is not None else 0.5*(ps_lo+ps_hi)))**2))
        # 物理箱拘束違反に大罰
        p1 = 0.5*(ps+pd); p2 = 0.5*(ps-pd)
        if (p1<0) or (p2<0) or (p1>pmax) or (p2>pmax):
            cost += 1e6
        if (best is None) or (cost < best[0]):
            best = (cost, ps, pd)

    _, ps, pd = best

    # 2) 近傍微調整（ゴールデンセクション風に ps を1D最適化、各点で pd を再解）
    for _ in range(6):
        span = 0.05*(ps_hi - ps_lo)
        cand = []
        for s in [ps-span, ps, ps+span]:
            s = float(np.clip(s, ps_lo, ps_hi))
            d, ok = root_in_pd_for_theta(s, theta_ref_deg, coef_th, degree, pd_lo, pd_hi)
            th_err = predict_theta(s, d, coef_th, degree) - theta_ref_deg
            dz = predict_dz(s, d, coef_dz, degree)
            cost = (1.0*th_err**2) + (w_z*dz**2) + (w_sigma*((s - (sigma_ref if sigma_ref is not None else 0.5*(ps_lo+ps_hi)))**2))
            p1 = 0.5*(s+d); p2 = 0.5*(s-d)
            if (p1<0) or (p2<0) or (p1>pmax) or (p2>pmax): cost += 1e6
            cand.append((cost, s, d))
        cand.sort(key=lambda t: t[0])
        _, ps, pd = cand[0]

    ps, pd, p1, p2 = clamp_phys(ps, pd, pmax)
    # 参考コスト（θはほぼ一致しているはず）
    th_err = predict_theta(ps, pd, coef_th, degree) - theta_ref_deg
    dz = predict_dz(ps, pd, coef_dz, degree)
    cost = (1.0*th_err**2) + (w_z*dz**2) + (w_sigma*((ps - (sigma_ref if sigma_ref is not None else 0.5*(ps_lo+ps_hi)))**2))
    return ps, pd, p1, p2, cost

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fit_inverse_static_dynamic.py で保存した *_model.npz")
    ap.add_argument("--theta-deg", type=float, required=True, help="目標角度 θ* [deg]")
    ap.add_argument("--pmax", type=float, default=None, help="MPa（未指定ならモデル内の値）")
    # 重み（θ優先：w_theta は 1.0 に固定、w_z で Δz≈0 の強さ、w_sigma で Σ を中庸に）
    ap.add_argument("--wz", type=float, default=0.2)
    ap.add_argument("--wsigma", type=float, default=0.02)
    args = ap.parse_args()

    data = np.load(args.model, allow_pickle=True)
    coef_th = data["coef_th"]
    coef_dz = data["coef_dz"]; coef_dz = None if coef_dz.size==0 else coef_dz
    degree  = int(data["degree"])
    ps_rng  = tuple(data["ps_rng"])
    pd_rng  = tuple(data["pd_rng"])
    sigma_ref = float(data["sigma_ref"])
    pmax = float(args.pmax) if args.pmax is not None else float(data["pmax"])

    if have_scipy():
        ps,pd,p1,p2,cost = solve_least_squares(args.theta_deg, coef_th, coef_dz, degree,
                                               ps_rng, pd_rng, pmax,
                                               w_theta=1.0, w_z=args.wz, w_sigma=args.wsigma,
                                               sigma_ref=sigma_ref)
        method = "least_squares"
    else:
        ps,pd,p1,p2,cost = solve_lexi_theta_first(args.theta_deg, coef_th, coef_dz, degree,
                                                  ps_rng, pd_rng, pmax,
                                                  w_z=args.wz, w_sigma=args.wsigma,
                                                  sigma_ref=sigma_ref)
        method = "lexicographic"

    print("=== inverse (Δz≈0 constraint) ===")
    print(f" method   : {method}")
    print(f" theta*   : {args.theta_deg:.3f} deg")
    print(f" Sigma*   : {ps:.4f} MPa")
    print(f" Delta*   : {pd:.4f} MPa")
    print(f" -> p1    : {p1:.4f} MPa,  p2: {p2:.4f} MPa  (box [0,{pmax}])")
    if coef_dz is not None:
        dz_est = predict_dz(ps, pd, coef_dz, degree)
        print(f" dz_est   : {dz_est[0]:.5f} m  (should be ~0)")
    print(f" cost     : {cost:.4e}")

if __name__ == "__main__":
    main()
