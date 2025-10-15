#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differential Evolution (DE) only — time/profiling oriented inverse mapping (small MPC)
for antagonistic McKibben with a trained NARX model.

- Path-tracking objective (terminal + ramp/smooth path)
- Static z model from meta["fz_ps_quadratic"] (optional CSV override)
- Diamond feasibility via strong penalty inside objective
- Optional rate limits as nonlinear inequality penalty
- Reports wall time, eval count, evals/sec

Outputs:
  out_dir/report.json
  out_dir/trajectories.png   (best solution rollout)
  
python inverse6_3_narx_map_mpc_forDE.py \
  --meta out_narx/out_narx3_z/narx_meta.json \
  --model out_narx/out_narx3_z/narx_model.pt \
  --context_csv out/dynamic_prbs_data.csv \
  --theta_target_deg 20 \
  --horizon 10 \
  --path smooth \
  --pmax 0.7 \
  --w_theta_term 8.0 --w_theta_path 3.0 --w_ps 0.02 --w_rate 0.05 \
  --rate_ps 2.0 --rate_pd 2.0 \
  --maxiter 150 --popsize 15 --tol 1e-6 --seed 42 --polish \
  --workers 1 \
  --out_dir model_sim/out_inv_de

"""

import os, json, math, argparse, time, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# ---------------- NARX (same arch as training) ----------------
class MLP_NARX(nn.Module):
    def __init__(self, in_dim, hidden=[128,128], out_dim=1, dropout=0.0):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.ReLU()]
            d=h
        layers += [nn.Linear(d,out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

def load_meta(meta_path):
    with open(meta_path,"r") as f:
        meta = json.load(f)
    feat_cols = meta["feature_names_single_slice"]
    lags      = int(meta["lags"])
    delay     = int(meta["delay"])
    mu        = np.array(meta["mu"], dtype=np.float32)
    std       = np.array(meta["std"], dtype=np.float32)
    hidden    = int(meta.get("hidden",128))
    dropout   = float(meta.get("dropout",0.0))
    theta_minmax = meta.get("theta_train_minmax", None)
    fz_meta = meta.get("fz_ps_quadratic", None)
    return meta, feat_cols, lags, delay, mu, std, hidden, dropout, theta_minmax, fz_meta

def load_model(model_path, in_dim, hidden, dropout, device):
    m = MLP_NARX(in_dim, hidden=[hidden,hidden], out_dim=1, dropout=dropout).to(device)
    # Future-proof torch.load
    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(model_path, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m

# -------------- Context window / features --------------
REQ = ["t[s]","p_sum[MPa]","p_diff[MPa]","theta[rad]","dz[m]"]

def estimate_dt(df):
    t = df["t[s]"].to_numpy()
    if len(t)>=3:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt<=0: dt=0.01
    else:
        dt=0.01
    return dt

def load_context_window(context_csv, lags, delay, feat_cols):
    df = pd.read_csv(context_csv).sort_values("t[s]").reset_index(drop=True)
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise ValueError(f"context_csv missing columns: {miss}")
    dt = estimate_dt(df)
    # derive dp_sum, dp_diff
    for col, dcol in [("p_sum[MPa]","dp_sum[MPa/s]"),("p_diff[MPa]","dp_diff[MPa/s]")]:
        x = df[col].to_numpy()
        dx = np.zeros_like(x)
        if len(x)>1:
            dx[1:] = (x[1:]-x[:-1])/dt
            dx[0]  = dx[1]
        df[dcol]=dx
    need = lags+delay
    if len(df)<need:
        raise ValueError(f"context too short: need >= {need} rows")
    dfw = df.tail(need).reset_index(drop=True)
    return dfw, dt

def build_xt_from_window(df_win, feat_cols, lags, delay, mu, std, device,
                         ps_now=None, pd_now=None, ps_prev=None, pd_prev=None, dt=0.01,
                         theta_used_override=None):
    base  = len(df_win)-1
    b_del = base - delay
    fv=[]
    name2idx = {n:i for i,n in enumerate(feat_cols)}
    for k in range(lags):
        idx = b_del - k
        row = df_win.iloc[idx][feat_cols].to_numpy().astype(np.float32)
        # override theta history (closed-loop rollout)
        if theta_used_override is not None and "theta[rad]" in name2idx:
            rel = idx
            row[name2idx["theta[rad]"]] = float(theta_used_override[rel])
        # latest slice: inject candidate ps/pd (+ their derivatives)
        if k==0 and ps_now is not None and pd_now is not None:
            if "p_sum[MPa]" in name2idx: row[name2idx["p_sum[MPa]"]] = ps_now
            if "p_diff[MPa]" in name2idx: row[name2idx["p_diff[MPa]"]] = pd_now
            psp = float(df_win.iloc[idx-1]["p_sum[MPa]"]) if ps_prev is None else ps_prev
            pdp = float(df_win.iloc[idx-1]["p_diff[MPa]"]) if pd_prev is None else pd_prev
            dps = (ps_now - psp)/max(dt,1e-6)
            dpd = (pd_now - pdp)/max(dt,1e-6)
            if "dp_sum[MPa/s]" in name2idx: row[name2idx["dp_sum[MPa/s]"]] = dps
            if "dp_diff[MPa/s]" in name2idx: row[name2idx["dp_diff[MPa/s]"]] = dpd
        fv.append(row)
    x = np.concatenate(fv, axis=0)[None,:]
    x_std = (x - mu)/std
    return torch.from_numpy(x_std).float().to(device)

# -------------- z model --------------
def z_model_from_meta(fz_meta):
    if fz_meta is None:
        return None
    a0 = float(fz_meta.get("a0",0.0))
    a1 = float(fz_meta.get("a1",0.0))
    a2 = float(fz_meta.get("a2",0.0))
    def f(ps, pd):
        return a0 + a1*ps + a2*(ps**2)
    return f

def fit_static_z_model(static_csv=None):
    if static_csv is None:
        return None
    df = pd.read_csv(static_csv)
    need = ["p_sum[MPa]","z[m]"]
    if any(c not in df.columns for c in need):
        warnings.warn("static z model: columns missing; fallback to meta/fallback.")
        return None
    X = np.c_[ np.ones(len(df)), df["p_sum[MPa]"], df["p_sum[MPa]"]**2 ]
    y = df["z[m]"].to_numpy()
    lam=1e-6
    A = X.T@X + lam*np.eye(X.shape[1])
    b = X.T@y
    coef = np.linalg.solve(A,b)
    def f(ps,pd):
        return float(coef[0] + coef[1]*ps + coef[2]*ps*ps)
    return f

# -------------- Feasible region helper --------------
def in_feasible(ps,pd,pmax):
    return (0<=ps<=2*pmax) and (abs(pd)<=min(ps, 2*pmax-ps))

# -------------- Rollout & objective --------------
def rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                  seq_ps, seq_pd):
    H = len(seq_ps)
    theta_used = df_win["theta[rad]"].to_numpy().astype(np.float32).copy()
    preds = []
    base = len(df_win)-1; b_del = base - delay
    ps_prev = float(df_win.iloc[b_del-1]["p_sum[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_sum[MPa]"])
    pd_prev = float(df_win.iloc[b_del-1]["p_diff[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_diff[MPa]"])
    for k in range(H):
        ps_k, pd_k = float(seq_ps[k]), float(seq_pd[k])
        xt = build_xt_from_window(df_win, feat_cols, lags, delay, mu, std, device,
                                  ps_now=ps_k, pd_now=pd_k,
                                  ps_prev=ps_prev, pd_prev=pd_prev, dt=dt,
                                  theta_used_override=theta_used)
        with torch.no_grad():
            th_next = float(model(xt).cpu().numpy().reshape(-1)[0])
        preds.append(th_next)
        theta_used = np.roll(theta_used, -1); theta_used[-1] = th_next
        ps_prev, pd_prev = ps_k, pd_k
    return np.array(preds, dtype=float)

def make_theta_path(theta0, theta_star, H, kind="smooth"):
    alphas = np.linspace(1/H, 1.0, H)
    if kind == "smooth":
        # cosine-ease (S-curve)
        s = 0.5*(1 - np.cos(np.pi*alphas))
    else:
        s = alphas  # linear ramp
    return theta0 + s*(theta_star - theta0)

# -------------- Plotting --------------
def plot_trajectories(out_png, theta_roll, theta_ref, theta_target, ps_seq, pd_seq):
    H = len(ps_seq)
    t = np.arange(1,H+1)
    fig, ax = plt.subplots(2,1, figsize=(9,6), sharex=True)
    ax[0].plot(t, theta_roll, marker='o', label="theta_hat")
    ax[0].plot(t, theta_ref,  ls='--', label="theta_ref (path)")
    ax[0].axhline(theta_target, ls=':', label="theta*")
    ax[0].set_ylabel("theta [rad]"); ax[0].legend()
    ax[1].plot(t, ps_seq, marker='o', label="p_sum")
    ax[1].plot(t, pd_seq, marker='o', label="p_diff")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("MPa"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

# -------------- Main (DE only) --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context_csv", required=True)
    ap.add_argument("--theta_target_deg", type=float, default=None)
    ap.add_argument("--theta_target_rad", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--path", type=str, default="smooth", choices=["ramp","smooth"])
    ap.add_argument("--pmax", type=float, default=0.7)
    # weights
    ap.add_argument("--w_theta_term", type=float, default=10.0)
    ap.add_argument("--w_theta_path", type=float, default=2.0)
    ap.add_argument("--w_ps", type=float, default=0.02)
    ap.add_argument("--w_rate", type=float, default=0.03)
    # optional hard-ish rate limits (penalty)
    ap.add_argument("--rate_ps", type=float, default=None)  # MPa/s
    ap.add_argument("--rate_pd", type=float, default=None)  # MPa/s
    ap.add_argument("--z_static_csv", type=str, default=None)  # optional override
    ap.add_argument("--out_dir", type=str, default="out_inv_de_only")
    ap.add_argument("--cpu", action="store_true")
    # DE options
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help=">1 for parallel evaluation (requires fork/spawn safety).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    theta_target = (args.theta_target_rad if args.theta_target_rad is not None
                    else math.radians(args.theta_target_deg))

    meta, feat_cols, lags, delay, mu, std, hidden, dropout, theta_minmax, fz_meta = load_meta(args.meta)
    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    model  = load_model(args.model, in_dim=len(feat_cols)*lags, hidden=hidden, dropout=dropout, device=device)

    df_win, dt = load_context_window(args.context_csv, lags, delay, feat_cols)
    base = len(df_win)-1; b_del = base - delay
    ps_prev0 = float(df_win.iloc[b_del-1]["p_sum[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_sum[MPa]"])
    pd_prev0 = float(df_win.iloc[b_del-1]["p_diff[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_diff[MPa]"])

    # z model: prefer CSV override, else meta quadratic, else None
    z_model = fit_static_z_model(args.z_static_csv)
    if z_model is None:
        z_model = z_model_from_meta(fz_meta)

    H = args.horizon; pmax = args.pmax

    # reference path
    theta0 = float(df_win["theta[rad]"].iloc[-1])
    theta_ref = make_theta_path(theta0, theta_target, H, args.path)

    # build objective with penalties; count evaluations
    eval_counter = {"n": 0}
    def cost_flat(x):
        eval_counter["n"] += 1
        x = np.asarray(x, dtype=float)
        ps_seq = x[0::2]; pd_seq = x[1::2]

        # feasibility penalty (diamond)
        pen = 0.0
        for k in range(H):
            ps, pd = ps_seq[k], pd_seq[k]
            if ps < 0: pen += (0-ps)**2
            if ps > 2*pmax: pen += (ps-2*pmax)**2
            if abs(pd) > ps: pen += (abs(pd)-ps)**2
            if abs(pd) > 2*pmax-ps: pen += (abs(pd)-(2*pmax-ps))**2

        # optional rate-limit penalty
        if args.rate_ps is not None or args.rate_pd is not None:
            for k in range(H):
                ref_ps = ps_prev0 if k==0 else ps_seq[k-1]
                ref_pd = pd_prev0 if k==0 else pd_seq[k-1]
                if args.rate_ps is not None:
                    pen += 1e6*max(0.0, abs(ps_seq[k]-ref_ps) - args.rate_ps*dt)**2
                if args.rate_pd is not None:
                    pen += 1e6*max(0.0, abs(pd_seq[k]-ref_pd) - args.rate_pd*dt)**2

        # rollout
        theta_roll = rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                                   ps_seq, pd_seq)

        # costs
        j_term = (theta_roll[-1] - theta_target)**2
        j_path = np.sum((theta_roll - theta_ref)**2)

        j_z = 0.0
        if z_model is not None:
            for k in range(H):
                j_z += z_model(ps_seq[k], pd_seq[k])**2

        j_rate = 0.0
        for k in range(H):
            ref_ps = ps_prev0 if k==0 else ps_seq[k-1]
            ref_pd = pd_prev0 if k==0 else pd_seq[k-1]
            j_rate += (ps_seq[k]-ref_ps)**2 + (pd_seq[k]-ref_pd)**2

        return (args.w_theta_term*j_term
                + args.w_theta_path*j_path
                + args.w_ps*j_z
                + args.w_rate*j_rate
                + 1e6*pen)

    # bounds for DE
    bnds = [(0.0, 2*pmax), (-2*pmax, 2*pmax)]*H

    # run DE (time it)
    t0 = time.time()
    r = differential_evolution(cost_flat,
                               bounds=bnds,
                               strategy='best1bin',
                               maxiter=args.maxiter,
                               popsize=args.popsize,
                               tol=args.tol,
                               seed=args.seed,
                               polish=args.polish,
                               workers=args.workers)
    t1 = time.time()
    wall_sec = t1 - t0
    neval = eval_counter["n"]
    evps = neval / max(wall_sec, 1e-9)

    # Post-process best
    x = np.array(r.x, dtype=float)
    ps_seq = x[0::2]; pd_seq = x[1::2]
    theta_roll = rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                               ps_seq, pd_seq)
    # components (for transparency)
    j_term = float((theta_roll[-1]-theta_target)**2)
    j_path = float(np.sum((theta_roll - theta_ref)**2))
    j_z = 0.0
    if z_model is not None:
        for k in range(H):
            j_z += float(z_model(ps_seq[k], pd_seq[k])**2)
    j_rate = 0.0
    for k in range(H):
        ref_ps = float(ps_prev0 if k==0 else ps_seq[k-1])
        ref_pd = float(pd_prev0 if k==0 else pd_seq[k-1])
        j_rate += float((ps_seq[k]-ref_ps)**2 + (pd_seq[k]-ref_pd)**2)

    # save report
    report = {
        "de": {
            "x": r.x.tolist(),
            "fun": float(r.fun),
            "nit": r.nit,
            "nfev": int(r.nfev),
            "success": bool(r.success),
            "message": str(r.message),
            "theta_roll": theta_roll.tolist(),
            "theta_ref": theta_ref.tolist(),
            "theta_terminal": float(theta_roll[-1]),
            "theta_err_terminal": float(theta_roll[-1]-theta_target),
            "ps_seq": ps_seq.tolist(),
            "pd_seq": pd_seq.tolist(),
            "J_term": j_term, "J_path": j_path,
            "J_z": float(j_z), "J_rate": float(j_rate),
        },
        "timing": {
            "wall_seconds": wall_sec,
            "eval_count": neval,
            "evals_per_second": evps,
            "dt_context_est": float(dt),
        },
        "config": {
            "horizon": H, "path": args.path, "pmax": pmax,
            "weights": {
                "w_theta_term": args.w_theta_term,
                "w_theta_path": args.w_theta_path,
                "w_ps": args.w_ps,
                "w_rate": args.w_rate,
            },
            "rate_limits": {
                "ps_MPa_per_s": args.rate_ps,
                "pd_MPa_per_s": args.rate_pd,
            },
            "de": {
                "maxiter": args.maxiter, "popsize": args.popsize,
                "tol": args.tol, "seed": args.seed,
                "polish": args.polish, "workers": args.workers,
            }
        }
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"report.json"),"w") as f:
        json.dump(report, f, indent=2)

    # plot trajectories
    plot_trajectories(os.path.join(args.out_dir,"trajectories.png"),
                      theta_roll, theta_ref, theta_target, ps_seq, pd_seq)

    # stdout summary
    print("== Differential Evolution (DE) only ==")
    print(f"time: {wall_sec:.3f} s | evals: {neval} | {evps:.1f} eval/s")
    print(f"best J: {float(r.fun):.6f} | success: {bool(r.success)} | iters: {r.nit}")
    print(f"theta_terminal: {float(theta_roll[-1]):.6f} rad "
          f"(err {float(theta_roll[-1]-theta_target):+.6f})")
    print(f"Saved: {os.path.join(args.out_dir,'report.json')}")
    print(f"Saved: {os.path.join(args.out_dir,'trajectories.png')}")

if __name__ == "__main__":
    main()
