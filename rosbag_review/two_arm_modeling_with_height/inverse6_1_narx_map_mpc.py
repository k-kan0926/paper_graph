#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-horizon inverse mapping (small MPC) for antagonistic McKibben with a trained NARX.

Given theta*, find a sequence {(ps_k, pd_k)}_{k=1..H} that drives theta to the target
while keeping z small (proxy: small ps, or an optional static z model), and respecting
feasible region and rate limits.

Outputs:
  out_dir/report.json   - per-solver solution details (pressures, theta rollouts, costs)
  out_dir/landscape.png - 1st-step landscape with feasible region and solver marks
  out_dir/trajectories.png - theta & pressures vs step for best solution

Usage (example):
python inverse6_1_narx_map_mpc.py \
--meta out_narx/narx_meta.json \
--model out_narx/narx_model.pt \
--context_csv out/dynamic_prbs_data.csv \
--theta_target_deg 20 \
--horizon 8 \
--pmax 0.7 \
--w_theta 10.0 --w_ps 0.02 --w_rate 0.01 \
--rate_ps 3.0 --rate_pd 3.0 \
--z_static_csv out/static1_data.csv \
--out_dir out_inv_mpc
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
    return meta, feat_cols, lags, delay, mu, std, hidden, dropout

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
    """
    Build standardized NARX input vector using df_win (size = lags+delay).
    - Replace most recent (k=0) ps/pd and their derivatives by ps_now/pd_now if provided.
    - Optionally override theta at each slice using theta_used_override[list] during rollout.
    """
    base  = len(df_win)-1  # current row
    b_del = base - delay
    fv=[]
    name2idx = {n:i for i,n in enumerate(feat_cols)}
    for k in range(lags):
        idx = b_del - k
        row = df_win.iloc[idx][feat_cols].to_numpy().astype(np.float32)
        # theta override (for closed-loop rollout)
        if theta_used_override is not None:
            # theta at absolute index idx within df_win history
            rel = idx  # 0..(lags+delay-1) range
            if "theta[rad]" in name2idx and 0<=rel<len(theta_used_override):
                row[name2idx["theta[rad]"]] = float(theta_used_override[rel])
        # latest slice replace p_sum / p_diff
        if k==0 and ps_now is not None and pd_now is not None:
            if "p_sum[MPa]" in name2idx: row[name2idx["p_sum[MPa]"]] = ps_now
            if "p_diff[MPa]" in name2idx: row[name2idx["p_diff[MPa]"]] = pd_now
            # derivatives by prev
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

# -------------- Optional static z model --------------
def fit_static_z_model(static_csv=None):
    if static_csv is None:
        return lambda ps,pd: float(ps)
    df = pd.read_csv(static_csv)
    need = ["p_sum[MPa]","p_diff[MPa]","z[m]"]
    if any(c not in df.columns for c in need):
        warnings.warn("static z model: columns missing; fallback to z~ps")
        return lambda ps,pd: float(ps)
    X = np.c_[ np.ones(len(df)), df["p_sum[MPa]"], df["p_sum[MPa]"]**2, np.abs(df["p_diff[MPa]"]) ]
    y = df["z[m]"].to_numpy()
    lam=1e-6
    A = X.T@X + lam*np.eye(X.shape[1])
    b = X.T@y
    coef = np.linalg.solve(A,b)
    return lambda ps,pd: float(coef[0] + coef[1]*ps + coef[2]*ps*ps + coef[3]*abs(pd))

# -------------- Feasible region per-step --------------
def bounds_lin_for_ps_pd(pmax):
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

# -------------- Horizon rollout & objective --------------
def rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                  seq_ps, seq_pd, theta_init_hist):
    """
    Closed-loop multi-step rollout:
      At each step k, build features using latest theta_used history (theta_used shifts in),
      inject candidate (ps_k,pd_k) at the most recent slice, predict theta_{+1}, append it.
    """
    H = len(seq_ps)
    # theta_used history buffer aligned with df_win indices [0..lags+delay-1]:
    # we copy df_win theta for the initial buffer
    theta_used = df_win["theta[rad]"].to_numpy().astype(np.float32).copy()
    preds = []
    # previous pressures for derivative at k=1 (from delayed-1)
    base = len(df_win)-1
    b_del = base - delay
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
        # shift theta_used: drop oldest, append th_next at the end
        theta_used = np.roll(theta_used, -1)
        theta_used[-1] = th_next
        # update "prev" for next derivative (current ps/pd become prev)
        ps_prev, pd_prev = ps_k, pd_k
    return np.array(preds, dtype=float)

def build_objective(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                    theta_target, H, w_theta, w_ps, w_rate, z_model, pmax,
                    ps_prev0, pd_prev0):
    def cost_flat(x):
        x = np.asarray(x, dtype=float)
        ps_seq = x[0::2]; pd_seq = x[1::2]
        # feasibility projection penalty (soft) to help unconstrained solvers
        pen = 0.0
        for k in range(H):
            ps, pd = ps_seq[k], pd_seq[k]
            # outside diamond -> quadratic penalty
            if ps < 0: pen += (0-ps)**2
            if ps > 2*pmax: pen += (ps-2*pmax)**2
            if abs(pd) > ps: pen += (abs(pd)-ps)**2
            if abs(pd) > 2*pmax-ps: pen += (abs(pd)-(2*pmax-ps))**2
        # rollout
        theta_roll = rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                                   ps_seq, pd_seq, df_win["theta[rad]"].to_numpy())
        j_theta = (theta_roll[-1] - theta_target)**2
        # z proxy & rate
        j_z = 0.0; j_rate = 0.0
        for k in range(H):
            ps, pd = ps_seq[k], pd_seq[k]
            j_z += z_model(ps,pd)**2
            if k==0:
                j_rate += (ps-ps_prev0)**2 + (pd-pd_prev0)**2
            else:
                j_rate += (ps-ps_seq[k-1])**2 + (pd-pd_seq[k-1])**2
        return w_theta*j_theta + w_ps*j_z + w_rate*j_rate + 1e6*pen
    return cost_flat

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

def plot_trajectories(out_png, theta_roll, theta_target, ps_seq, pd_seq):
    H = len(ps_seq)
    t = np.arange(1,H+1)
    fig, ax = plt.subplots(2,1, figsize=(9,6), sharex=True)
    ax[0].plot(t, theta_roll, marker='o', label="theta_hat")
    ax[0].axhline(theta_target, ls='--', label="theta*")
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
    ap.add_argument("--pmax", type=float, default=0.7)
    ap.add_argument("--w_theta", type=float, default=10.0)
    ap.add_argument("--w_ps", type=float, default=0.02)
    ap.add_argument("--w_rate", type=float, default=0.01)
    ap.add_argument("--rate_ps", type=float, default=None)  # MPa/s
    ap.add_argument("--rate_pd", type=float, default=None)  # MPa/s
    ap.add_argument("--z_static_csv", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="out_inv_mpc")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    theta_target = (args.theta_target_rad if args.theta_target_rad is not None
                    else math.radians(args.theta_target_deg))

    meta, feat_cols, lags, delay, mu, std, hidden, dropout = load_meta(args.meta)
    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    model  = load_model(args.model, in_dim=len(feat_cols)*lags, hidden=hidden, dropout=dropout, device=device)

    df_win, dt = load_context_window(args.context_csv, lags, delay, feat_cols)

    # previous pressures at delayed-1
    base = len(df_win)-1; b_del = base - delay
    ps_prev0 = float(df_win.iloc[b_del-1]["p_sum[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_sum[MPa]"])
    pd_prev0 = float(df_win.iloc[b_del-1]["p_diff[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_diff[MPa]"])

    z_model = fit_static_z_model(args.z_static_csv)

    H = args.horizon; pmax = args.pmax
    # Decision vector x = [ps1,pd1, ps2,pd2, ..., psH,pdH]
    def cost_flat(x):
        return build_objective(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                               theta_target, H, args.w_theta, args.w_ps, args.w_rate,
                               z_model, pmax, ps_prev0, pd_prev0)(x)

    # constraints: per-step diamond + bounds + (optional) rate limits between steps
    bounds_list = []
    As=[]; lbs=[]; ubs=[]
    for k in range(H):
        b, lc = bounds_lin_for_ps_pd(pmax)
        bounds_list.append(b)
        As.append(lc.A); lbs.append(lc.lb); ubs.append(lc.ub)
    # SciPy supports a single LinearConstraint over full x; block-diagonalize A
    Ablk = np.zeros((4*H, 2*H))
    for k in range(H):
        Ablk[4*k:4*k+4, 2*k:2*k+2] = As[k]
    lb_all = np.concatenate(lbs); ub_all = np.concatenate(ubs)
    lc_all = LinearConstraint(Ablk, lb_all, ub_all)
    # simple box bounds
    lb_box = []; ub_box=[]
    for k in range(H):
        lb_box += [0.0, -2*pmax]; ub_box += [2*pmax, 2*pmax]
    bounds = Bounds(lb_box, ub_box)

    # optional rate constraints as nonlinear inequalities
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

    # seeds: hold previous + small nudges
    x0 = []
    ps0 = float(df_win.iloc[b_del]["p_sum[MPa]"]); pd0 = float(df_win.iloc[b_del]["p_diff[MPa]"])
    for k in range(H):
        x0 += [ps0, pd0]
    x0 = np.array(x0, dtype=float)

    # --- First-step landscape (for visualization only) ---
    def cost_first_step(ps, pd):
        # proxy: fix step1=(ps,pd), hold others at x0, compute cost
        x = x0.copy(); x[0]=ps; x[1]=pd
        return cost_flat(x)
    plot_landscape_first_step(os.path.join(args.out_dir,"landscape.png"), cost_first_step, pmax, theta_target)

    # --- Run solvers ---
    results = {}
    # SLSQP
    try:
        r = minimize(cost_flat, x0=x0, method="SLSQP",
                     bounds=bounds, constraints=[lc_all]+nonlin_cons,
                     options={"maxiter":400, "ftol":1e-9, "disp":False})
        results["slsqp"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                            "success": bool(r.success), "message": r.message}
    except Exception as e:
        results["slsqp"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}
    # trust-constr（ヘッセをゼロ指定で安定化）
    try:
        r = minimize(cost_flat, x0=x0, method="trust-constr",
                     bounds=bounds, constraints=[lc_all],
                     hess=lambda x: np.zeros((2*H,2*H)),
                     options={"maxiter":400, "gtol":1e-6, "xtol":1e-6, "verbose":0})
        results["trust_constr"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                                   "success": bool(r.success), "message": r.message}
    except Exception as e:
        results["trust_constr"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}
    # Powell（境界あり・線形拘束はペナルティに吸収済み）
    try:
        r = minimize(cost_flat, x0=x0, method="Powell",
                     bounds=bounds, options={"maxiter":600, "xtol":1e-6, "ftol":1e-9, "disp":False})
        results["powell"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                             "success": bool(r.success), "message": r.message}
    except Exception as e:
        results["powell"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}
    # Differential Evolution（大域探索）
    try:
        def feas_penalty(x):
            return cost_flat(x)  # cost_flat 内で強いペナルティ実装済み
        bnds = [(0.0,2*pmax), (-2*pmax,2*pmax)]*H
        r = differential_evolution(feas_penalty, bounds=bnds, maxiter=120, tol=1e-6, polish=True)
        results["de"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                         "success": bool(r.success), "message": "differential_evolution"}
    except Exception as e:
        results["de"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}

    # --- Post-process: rollout & metrics for each result ---
    enriched = {}
    best_name = None; best_val = np.inf
    for name, r in results.items():
        if r.get("x") is None: enriched[name]=r; continue
        x = np.array(r["x"], dtype=float)
        ps_seq = x[0::2]; pd_seq = x[1::2]
        theta_roll = rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                                   ps_seq, pd_seq, df_win["theta[rad]"].to_numpy())
        J_theta = float((theta_roll[-1]-theta_target)**2)
        J_z = float(np.sum([fit_static_z_model(args.z_static_csv)(ps_seq[i],pd_seq[i])**2 for i in range(len(ps_seq))]))
        J_rate = 0.0
        for k in range(len(ps_seq)):
            ref_ps = ps_prev0 if k==0 else ps_seq[k-1]
            ref_pd = pd_prev0 if k==0 else pd_seq[k-1]
            J_rate += (ps_seq[k]-ref_ps)**2 + (pd_seq[k]-ref_pd)**2
        r2 = dict(r)
        r2.update({
            "theta_roll": theta_roll.tolist(),
            "theta_terminal": float(theta_roll[-1]),
            "theta_err_terminal": float(theta_roll[-1]-theta_target),
            "ps_seq": ps_seq.tolist(),
            "pd_seq": pd_seq.tolist(),
            "J_theta": J_theta, "J_z": J_z, "J_rate": float(J_rate)
        })
        enriched[name]=r2
        if r["fun"] is not None and r["fun"]<best_val:
            best_val = r["fun"]; best_name = name
    enriched["best"] = {"name": best_name, "fun": best_val}

    # save JSON
    with open(os.path.join(args.out_dir,"report.json"),"w") as f:
        json.dump(enriched, f, indent=2)

    # plot best trajectories
    if best_name and enriched[best_name].get("x"):
        x = np.array(enriched[best_name]["x"], dtype=float)
        ps_seq = x[0::2]; pd_seq = x[1::2]
        theta_roll = np.array(enriched[best_name]["theta_roll"], dtype=float)
        plot_trajectories(os.path.join(args.out_dir,"trajectories.png"),
                          theta_roll, theta_target, ps_seq, pd_seq)

    print(f"Saved: {os.path.join(args.out_dir,'report.json')}")
    print(f"Saved: {os.path.join(args.out_dir,'landscape.png')}")
    if best_name:
        print(f"Best: {best_name} (J={best_val:.3e})")
    else:
        print("No solution found.")
if __name__ == "__main__":
    main()
