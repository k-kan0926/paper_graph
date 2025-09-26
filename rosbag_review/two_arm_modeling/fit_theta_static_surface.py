#!/usr/bin/env python3
#python fit_theta_static_surface.py --csv out/diff_run1_data.csv --deg 3 --lambda 1e-3 --out-prefix out/theta_static/theta_static
import argparse, os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def build_poly2d(ps, pd, deg=3):
    """和(ps)・差(pd)の多項式基底（全交差項）"""
    feats = []; names=[]
    for i in range(deg+1):
        for j in range(deg+1-i):
            feats.append((ps**i)*(pd**j)); names.append(f"ps^{i}*pd^{j}")
    X = np.column_stack(feats)
    return X, names

def ridge_fit(X, y, lam=1e-3):
    # (X^T X + lam I)^{-1} X^T y
    XT = X.T
    A = XT @ X + lam*np.eye(X.shape[1])
    b = XT @ y
    coef = np.linalg.solve(A, b)
    return coef

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    ap.add_argument("--out-prefix", default="out/theta_static")
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    df = pd.read_csv(args.csv, comment="#")
    ps = df["p_sum[MPa]"].to_numpy()
    pdiff = df["p_diff[MPa]"].to_numpy()
    th = df["theta[rad]"].to_numpy()
    # オプション：静置点だけ使うなどの前処理はここに追加可能

    X, names = build_poly2d(ps, pdiff, deg=args.deg)
    coef = ridge_fit(X, th, lam=args.lam)
    th_hat = X @ coef
    resid = th - th_hat
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    r2   = 1.0 - float(np.sum(resid**2)/max(1e-12, np.sum((th-np.mean(th))**2)))

    print(f"[StaticFit] deg={args.deg} lam={args.lam}  RMSE={rmse:.4f} rad  MAE={mae:.4f} rad  R2={r2:.4f}")

    # 保存
    np.savez(args.out_prefix+"_coef.npz", coef=coef, names=np.array(names), deg=args.deg)
    with open(args.out_prefix+"_report.json","w") as f:
        json.dump({"deg":args.deg,"lambda":args.lam,"rmse":rmse,"mae":mae,"r2":r2}, f, indent=2)

    # オンライン利用用の小モジュールも書き出す
    code = f'''# auto-generated
import numpy as np
coef = np.array({coef.tolist()}, dtype=float)
deg = {args.deg}
def theta_hat(ps, pd):
    # ps, pd: numpy配列 or スカラ(MPa) -> theta(rad)
    ps = np.asarray(ps); pd = np.asarray(pd)
    feats=[]
    for i in range(deg+1):
        for j in range(deg+1-i):
            feats.append((ps**i)*(pd**j))
    X = np.column_stack(feats) if np.ndim(ps)>0 or np.ndim(pd)>0 else np.array(feats).reshape(1,-1)
    return X @ np.asarray(coef)
'''
    with open(args.out_prefix+"_predictor.py","w") as f:
        f.write(code)
    print("[OK] wrote", args.out_prefix+"_coef.npz, _report.json, _predictor.py")

if __name__=="__main__":
    main()
