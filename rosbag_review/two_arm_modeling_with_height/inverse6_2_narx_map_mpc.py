#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script uses z_model_from_meta


Path-tracking finite-horizon inverse mapping (small MPC) for antagonistic McKibben + NARX.

Add a path-following term so theta approaches theta* smoothly (ramp or S-curve),
while keeping previous features: multi-solver, feasibility, rate limits, figures, JSON.

Outputs:
  out_dir/report.json
  out_dir/landscape.png        (1st-step proxy landscape)
  out_dir/trajectories.png     (theta & pressures vs step for best solution)

Example:
  python inverse6_2_narx_map_mpc.py \
    --meta out_narx/out_narx3_z/narx_meta.json \
    --model out_narx/out_narx3_z/narx_model.pt \
    --context_csv out/dynamic_prbs_data.csv \
    --theta_target_deg 20 \
    --horizon 10 \
    --path smooth \
    --pmax 0.7 \
    --w_theta_term 10.0 --w_theta_path 1.0 --w_ps 0.02 --w_rate 0.02 \
    --rate_ps 2.0 --rate_pd 2.0 \
    --out_dir model_sim/out_inv2
"""

import os, json, math, argparse, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from scipy.optimize import minimize, Bounds, LinearConstraint, differential_evolution
import matplotlib.pyplot as plt

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

# -------------- Feasible region --------------
def bounds_lin_for_ps_pd(pmax):
    # diamond constraints:
    # +pd - ps <= 0
    # -pd - ps <= 0
    # +pd + ps <= 2pmax
    # -pd + ps <= 2pmax
    A = np.array([[ 1.0, -1.0],
                  [-1.0, -1.0],
                  [ 1.0,  1.0],
                  [-1.0,  1.0]], dtype=float)
    ub = np.array([0.0, 0.0, 2*pmax, 2*pmax], dtype=float)
    lb = -np.inf*np.ones_like(ub)
    bounds = Bounds([0.0, -2*pmax],[2*pmax, 2*pmax])
    lc = LinearConstraint(A, lb, ub)
    return bounds, lc

def in_feasible(ps,pd,pmax):
    return (0<=ps<=2*pmax) and (abs(pd)<=min(ps, 2*pmax-ps))

# -------------- Rollout & objectives --------------
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
    """Reference path theta_ref[1..H] from current theta0 to theta_star."""
    alphas = np.linspace(1/H, 1.0, H)
    if kind == "smooth":
        # cosine-ease (S-curve)
        s = 0.5*(1 - np.cos(np.pi*alphas))
    else:
        # linear ramp
        s = alphas
    return theta0 + s*(theta_star - theta0)

def build_objective(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                    theta_target, H, w_term, w_path, w_ps, w_rate,
                    z_model, pmax, ps_prev0, pd_prev0, theta_path_kind):
    theta0 = float(df_win["theta[rad]"].iloc[-1])  # current theta
    theta_ref = make_theta_path(theta0, theta_target, H, theta_path_kind)

    def cost_flat(x):
        x = np.asarray(x, dtype=float)
        ps_seq = x[0::2]; pd_seq = x[1::2]
        # strong feasibility penalty (for unconstrained solvers):
        pen = 0.0
        for k in range(H):
            ps, pd = ps_seq[k], pd_seq[k]
            if ps < 0: pen += (0-ps)**2
            if ps > 2*pmax: pen += (ps-2*pmax)**2
            if abs(pd) > ps: pen += (abs(pd)-ps)**2
            if abs(pd) > 2*pmax-ps: pen += (abs(pd)-(2*pmax-ps))**2
        # rollout
        theta_roll = rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                                   ps_seq, pd_seq)
        # terminal + path tracking
        j_term = (theta_roll[-1] - theta_target)**2
        j_path = np.sum((theta_roll - theta_ref)**2)
        # z proxy (if provided)
        j_z = 0.0
        if z_model is not None:
            for k in range(H):
                j_z += z_model(ps_seq[k], pd_seq[k])**2
        # rate smoothness
        j_rate = 0.0
        for k in range(H):
            ref_ps = ps_prev0 if k==0 else ps_seq[k-1]
            ref_pd = pd_prev0 if k==0 else pd_seq[k-1]
            j_rate += (ps_seq[k]-ref_ps)**2 + (pd_seq[k]-ref_pd)**2
        return w_term*j_term + w_path*j_path + w_ps*j_z + w_rate*j_rate + 1e6*pen
    return cost_flat, theta_ref

# -------------- Plotting --------------
def plot_landscape_first_step(out_png, cost_first_step, pmax, theta_target):
    ps_lin = np.linspace(0, 2*pmax, 161)
    pts_ps=[]; pts_pd=[]; pts_J=[]
    for ps in ps_lin:
        lim = min(ps, 2*pmax-ps)
        pds = np.linspace(-lim, lim, 161)
        for pd in pds:
            pts_ps.append(ps); pts_pd.append(pd); pts_J.append(cost_first_step(ps,pd))
    ps = np.array(pts_ps); pd = np.array(pts_pd); J = np.array(pts_J)
    fig, ax = plt.subplots(figsize=(8,6))
    cf = ax.tricontourf(ps, pd, J, levels=30)
    cbar = plt.colorbar(cf, ax=ax); cbar.set_label("Objective (first-step proxy)")
    ps_line = np.linspace(0, 2*pmax, 200)
    ax.plot(ps_line,  ps_line, 'k--', lw=1)
    ax.plot(ps_line, -ps_line, 'k--', lw=1)
    ax.plot(ps_line,  2*pmax-ps_line, 'k--', lw=1)
    ax.plot(ps_line, -2*pmax+ps_line, 'k--', lw=1)
    ax.set_xlim(0, 2*pmax); ax.set_ylim(-2*pmax, 2*pmax)
    ax.set_xlabel("p_sum [MPa]"); ax.set_ylabel("p_diff [MPa]")
    ax.set_title(f"First-step landscape (theta*={theta_target:.3f} rad)")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

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

# -------------- Main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context_csv", required=True)
    ap.add_argument("--theta_target_deg", type=float, default=None)
    ap.add_argument("--theta_target_rad", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--path", type=str, default="smooth", choices=["ramp","smooth"])
    ap.add_argument("--pmax", type=float, default=0.7)
    ap.add_argument("--w_theta_term", type=float, default=10.0)
    ap.add_argument("--w_theta_path", type=float, default=1.0)
    ap.add_argument("--w_ps", type=float, default=0.02)
    ap.add_argument("--w_rate", type=float, default=0.01)
    ap.add_argument("--rate_ps", type=float, default=None)  # MPa/s
    ap.add_argument("--rate_pd", type=float, default=None)  # MPa/s
    ap.add_argument("--z_static_csv", type=str, default=None)  # optional override
    ap.add_argument("--out_dir", type=str, default="out_inv_mpc_path")
    ap.add_argument("--cpu", action="store_true")
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

    # Build objective with path-tracking
    cost_flat, theta_ref = build_objective(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                                           theta_target, H, args.w_theta_term, args.w_theta_path,
                                           args.w_ps, args.w_rate, z_model, pmax,
                                           ps_prev0, pd_prev0, args.path)

    # constraints: block-diagonal linear diamond + bounds
    A = np.array([[ 1.0, -1.0], [-1.0, -1.0], [ 1.0,  1.0], [-1.0,  1.0]], dtype=float)
    ub = np.array([0.0, 0.0, 2*pmax, 2*pmax], dtype=float)
    lb = -np.inf*np.ones_like(ub)
    Ablk = np.zeros((4*H, 2*H))
    for k in range(H):
        Ablk[4*k:4*k+4, 2*k:2*k+2] = A
    lc_all = LinearConstraint(Ablk, np.tile(lb, H), np.tile(ub, H))
    bounds = Bounds([0.0, -2*pmax]*H, [2*pmax, 2*pmax]*H)

    # optional rate constraints (nonlinear)
    nonlin_cons=[]
    def rate_fun_factory(idx, prev_val, rate, is_ps):
        if rate is None or rate<=0: return None
        def g(x):
            val = x[2*idx] if is_ps else x[2*idx+1]
            ref = prev_val if idx==0 else (x[2*(idx-1)] if is_ps else x[2*(idx-1)+1])
            return rate*dt - abs(val - ref)
        return g
    for k in range(H):
        g1 = rate_fun_factory(k, ps_prev0, args.rate_ps, True)
        g2 = rate_fun_factory(k, pd_prev0, args.rate_pd, False)
        if g1: nonlin_cons.append({"type":"ineq","fun":g1})
        if g2: nonlin_cons.append({"type":"ineq","fun":g2})

    # seeds
    ps0 = float(df_win.iloc[b_del]["p_sum[MPa]"]); pd0 = float(df_win.iloc[b_del]["p_diff[MPa]"])
    x0 = np.array([v for _ in range(H) for v in (ps0, pd0)], dtype=float)

    # First-step landscape (proxy)
    def cost_first_step(ps, pd):
        x = x0.copy(); x[0]=ps; x[1]=pd
        return cost_flat(x)
    plot_landscape_first_step(os.path.join(args.out_dir,"landscape.png"),
                              cost_first_step, pmax, theta_target)

    results = {}
    # SLSQP
    try:
        r = minimize(cost_flat, x0=x0, method="SLSQP",
                     bounds=bounds, constraints=[lc_all]+nonlin_cons,
                     options={"maxiter":500, "ftol":1e-9, "disp":False})
        results["slsqp"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                            "success": bool(r.success), "message": r.message}
    except Exception as e:
        results["slsqp"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}
    # trust-constr
    try:
        r = minimize(cost_flat, x0=x0, method="trust-constr",
                     bounds=bounds, constraints=[lc_all],
                     hess=lambda x: np.zeros((2*H,2*H)),
                     options={"maxiter":500, "gtol":1e-6, "xtol":1e-6, "verbose":0})
        results["trust_constr"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                                   "success": bool(r.success), "message": r.message}
    except Exception as e:
        results["trust_constr"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}
    # Powell
    try:
        r = minimize(cost_flat, x0=x0, method="Powell",
                     bounds=bounds, options={"maxiter":700, "xtol":1e-6, "ftol":1e-9, "disp":False})
        results["powell"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                             "success": bool(r.success), "message": r.message}
    except Exception as e:
        results["powell"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}
    # Differential Evolution
    try:
        def feas_penalty(x):  # cost_flat already has strong feasibility penalty
            return cost_flat(x)
        bnds = [(0.0,2*pmax), (-2*pmax,2*pmax)]*H
        r = differential_evolution(feas_penalty, bounds=bnds, maxiter=150, tol=1e-6, polish=True)
        results["de"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                         "success": bool(r.success), "message": "differential_evolution"}
    except Exception as e:
        results["de"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}

    # Post-process & save
    enriched = {}
    best_name = None; best_val = np.inf
    for name, r in results.items():
        if r.get("x") is None: enriched[name]=r; continue
        x = np.array(r["x"], dtype=float)
        ps_seq = x[0::2]; pd_seq = x[1::2]
        theta_roll = rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                                   ps_seq, pd_seq)
        # components for transparency
        j_term = (theta_roll[-1]-theta_target)**2
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
        r2 = dict(r); r2.update({
            "theta_roll": theta_roll.tolist(),
            "theta_ref": theta_ref.tolist(),
            "theta_terminal": float(theta_roll[-1]),
            "theta_err_terminal": float(theta_roll[-1]-theta_target),
            "ps_seq": ps_seq.tolist(),
            "pd_seq": pd_seq.tolist(),
            "J_term": float(j_term), "J_path": float(j_path),
            "J_z": float(j_z), "J_rate": float(j_rate)
        })
        enriched[name]=r2
        if r["fun"] is not None and r["fun"]<best_val:
            best_val = r["fun"]; best_name = name
    enriched["best"] = {"name": best_name, "fun": best_val}

    with open(os.path.join(args.out_dir,"report.json"),"w") as f:
        json.dump(enriched, f, indent=2)

    # plot best
    if best_name and enriched[best_name].get("x"):
        x = np.array(enriched[best_name]["x"], dtype=float)
        ps_seq = x[0::2]; pd_seq = x[1::2]
        theta_roll = np.array(enriched[best_name]["theta_roll"], dtype=float)
        plot_trajectories(os.path.join(args.out_dir,"trajectories.png"),
                          theta_roll, np.array(enriched[best_name]["theta_ref"],dtype=float),
                          theta_target, ps_seq, pd_seq)

    print(f"Saved: {os.path.join(args.out_dir,'report.json')}")
    print(f"Saved: {os.path.join(args.out_dir,'landscape.png')}")
    if best_name:
        print(f"Best: {best_name} (J={best_val:.3e})")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
