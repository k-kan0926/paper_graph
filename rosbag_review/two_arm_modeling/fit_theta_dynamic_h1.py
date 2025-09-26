#!/usr/bin/env python3
#python fit_theta_dynamic_h1.py --csv out/diff_run1_data.csv --deg 3 --use-scipy --out-prefix out/theta_dyn/theta_dyn
import argparse, os, json, numpy as np, pandas as pd

def build_poly2d(ps, pd, deg=3):
    feats=[]; idx=[]
    for i in range(deg+1):
        for j in range(deg+1-i):
            feats.append((ps**i)*(pd**j)); idx.append((i,j))
    return np.column_stack(feats), idx

def simulate(theta0, ps, pd, dt, coef, deg, tau0, tau1):
    # f(ps,pd)
    X,_ = build_poly2d(ps, pd, deg)
    f = X @ coef
    tau = np.maximum(1e-3, tau0 + tau1*ps)  # 安全下限
    th_hat = np.empty_like(ps)
    th = theta0
    for k in range(len(ps)):
        th = th + (dt[k]/tau[k]) * (f[k] - th)
        th_hat[k] = th
    return th_hat

def ls_static(ps, pd, th, deg, lam):
    X,_ = build_poly2d(ps, pd, deg)
    XT = X.T
    return np.linalg.solve(XT@X + lam*np.eye(X.shape[1]), XT@th)

def fit_dynamic(ps, pd, t, th, deg=3, lam=1e-3, max_iter=30):
    # 交互最適化: f を固定→tau推定 → tau固定→f推定 を繰返し（SciPy無しフォールバック）
    dt = np.diff(t, prepend=t[0])
    # 初期: 静的フィット
    coef = ls_static(ps, pd, th, deg, lam)
    tau0, tau1 = 0.15, 0.0  # 初期値（秒）: 実機に合わせて調整
    for it in range(max_iter):
        # 1) tau を線形回帰で更新（近似：誤差微分を無視した準ニュートン風）
        th_hat = simulate(th[0], ps, pd, dt, coef, deg, tau0, tau1)
        e = th - th_hat
        rmse = float(np.sqrt(np.mean(e**2)))
        # 速度項 v = (f - θ)
        X,_ = build_poly2d(ps, pd, deg)
        f = X @ coef
        v = f - np.concatenate(([th[0]], th[:-1]))
        # 近似: th_{k+1} - th_k ≈ (dt/τ)(f - th_k)
        y = np.diff(th, prepend=th[0])
        # τ ≈ dt * v / y （不安定な点は除外）
        mask = (np.abs(y)>1e-4) & (np.abs(v)>1e-6) & (dt>1e-4)
        if np.any(mask):
            tau_est = (dt[mask] * v[mask]) / y[mask]
            # τ(ps) = tau0 + tau1*ps を最小二乗
            A = np.column_stack([np.ones_like(ps[mask]), ps[mask]])
            sol, *_ = np.linalg.lstsq(A, tau_est, rcond=None)
            tau0 = float(max(1e-3, sol[0])); tau1 = float(sol[1])
        # 2) f を更新（θの一歩先予測に合わせる）
        #    θ_{k+1} = θ_k + (dt/τ)(f_k - θ_k) ⇒ f_k ≈ θ_k + (τ/dt)(θ_{k+1}-θ_k)
        tau = np.maximum(1e-3, tau0 + tau1*ps)
        rhs = th + (tau/np.maximum(1e-4, dt)) * (np.diff(th, prepend=th[0]))
        X,_ = build_poly2d(ps, pd, deg)
        coef = np.linalg.lstsq(X, rhs, rcond=None)[0]
        # 収束チェック（任意）
        th_hat2 = simulate(th[0], ps, pd, dt, coef, deg, tau0, tau1)
        rmse2 = float(np.sqrt(np.mean((th - th_hat2)**2)))
        if abs(rmse - rmse2) < 1e-5:
            break
    return coef, tau0, tau1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    ap.add_argument("--out-prefix", default="out/theta_dyn")
    ap.add_argument("--use-scipy", action="store_true", help="SciPy least_squares を使う（あれば）")
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    df = pd.read_csv(args.csv, comment="#")
    t  = df["t[s]"].to_numpy()
    ps = df["p_sum[MPa]"].to_numpy()
    pdiff = df["p_diff[MPa]"].to_numpy()
    th = df["theta[rad]"].to_numpy()

    if args.use_scipy:
        try:
            from scipy.optimize import least_squares
            def pack(coef,t0,t1): return np.concatenate([coef, [t0,t1]])
            def unpack(x, n): return x[:n], x[n], x[n+1]
            X,_ = build_poly2d(ps, pdiff, args.deg)
            coef0 = np.linalg.lstsq(X, th, rcond=None)[0]
            x0 = pack(coef0, 0.15, 0.0)
            dt = np.diff(t, prepend=t[0])
            def residual(x):
                coef, tau0, tau1 = unpack(x, len(coef0))
                tau0 = max(1e-3, tau0)
                th_hat = simulate(th[0], ps, pdiff, dt, coef, args.deg, tau0, tau1)
                return th_hat - th
            res = least_squares(residual, x0, method="trf")
            coef, tau0, tau1 = unpack(res.x, len(coef0))
        except Exception:
            coef, tau0, tau1 = fit_dynamic(ps, pdiff, t, th, args.deg, args.lam)
    else:
        coef, tau0, tau1 = fit_dynamic(ps, pdiff, t, th, args.deg, args.lam)

    # 評価
    dt = np.diff(t, prepend=t[0])
    th_hat = simulate(th[0], ps, pdiff, dt, coef, args.deg, tau0, tau1)
    err = th - th_hat
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))

    print(f"[DynFit] deg={args.deg}  RMSE={rmse:.4f} rad  MAE={mae:.4f} rad  tau(ps)=max(1e-3,{tau0:.3f}+{tau1:.3f}*ps)")

    # 保存
    out = {"deg":args.deg, "coef":coef.tolist(), "tau0":tau0, "tau1":tau1,
           "rmse":rmse, "mae":mae}
    with open(args.out_prefix+"_model.json","w") as f: json.dump(out,f,indent=2)
    # オンライン用関数
    code = f'''# auto-generated dynamic model
import numpy as np
coef = np.array({coef.tolist()}, dtype=float)
deg = {args.deg}
tau0 = {tau0:.6f}
tau1 = {tau1:.6f}

def f_static(ps, pd):
    ps = np.asarray(ps); pd = np.asarray(pd)
    feats=[]
    for i in range(deg+1):
        for j in range(deg+1-i):
            feats.append((ps**i)*(pd**j))
    X = np.column_stack(feats) if (np.ndim(ps)>0 or np.ndim(pd)>0) else np.array(feats, float).reshape(1,-1)
    return (X @ coef)

def step_one(theta_prev, ps_k, pd_k, dt_k):
    tau = max(1e-3, tau0 + tau1*float(ps_k))
    f = float(f_static(ps_k, pd_k))
    return theta_prev + (dt_k/tau)*(f - theta_prev)

def simulate(theta0, ps, pd, dt):
    ps=np.asarray(ps); pd=np.asarray(pd); dt=np.asarray(dt)
    th=np.empty_like(ps, dtype=float); x=float(theta0)
    for k in range(ps.size):
        x = step_one(x, ps[k], pd[k], dt[k])
        th[k]=x
    return th
'''
    with open(args.out_prefix+"_predictor.py","w") as f:
        f.write(code)

    print("[OK] wrote", args.out_prefix+"_model.json, _predictor.py")

if __name__=="__main__":
    main()
