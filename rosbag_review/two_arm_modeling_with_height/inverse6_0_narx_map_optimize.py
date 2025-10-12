#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse mapping for antagonistic McKibben system:
Given target theta*, find pressures (p_sum, p_diff) that achieve it while
keeping z small (proxied by small p_sum unless a static z model is supplied).

Inputs:
  - narx_model.pt, narx_meta.json (from the training script)
  - context CSV providing the latest history window to build the NARX input
    (must contain: t[s], p_sum[MPa], p_diff[MPa], theta[rad], dz[m] columns at least)

Outputs:
  - out_dir/solution_report.json (per-solver results)
  - out_dir/solution.png (contour plot with feasible region and solver hits)

Usage example:
python inverse6_narx_map_optimize.py \
    --meta out_narx/narx_meta.json \
    --model out_narx/narx_model.pt \
    --context_csv out/dynamic_prbs_data.csv \
    --theta_target_deg 20.0 \
    --pmax 0.7 \
    --out_dir out_inv \
    --w_theta 1.0 --w_ps 0.01 --w_drate 0.0 \
    --rate_ps 5.0 --rate_pd 5.0 \
    --grid 81

Notes:
  - If you don't have a fresh context CSV, you can pass --ps_prev/--pd_prev/--theta_prev/--dz_prev
    to synthesize the last row, but for best results use a real context CSV.
  - If you have a static z-model CSV, you can add --z_static_csv ... to fit a simple f_z(p_s[,p_d]).
"""

import os, json, math, argparse, warnings
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, differential_evolution

# -------------------- Load NARX --------------------
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
    # minimal fields
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
    sd = torch.load(model_path, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m

# -------------------- Context builder --------------------
REQ = ["t[s]","p_sum[MPa]","p_diff[MPa]","theta[rad]","dz[m]"]
def estimate_dt(df):
    t = df["t[s]"].to_numpy()
    if len(t)>=3:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt<=0: dt=0.01
    else:
        dt = 0.01
    return dt

def load_context_window(context_csv, lags, delay, feat_cols):
    df = pd.read_csv(context_csv).sort_values("t[s]").reset_index(drop=True)
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise ValueError(f"context_csv missing columns: {miss}")
    # derive dp_sum, dp_diff
    dt = estimate_dt(df)
    for col, dcol in [("p_sum[MPa]","dp_sum[MPa/s]"),("p_diff[MPa]","dp_diff[MPa/s]")]:
        x = df[col].to_numpy()
        dx = np.zeros_like(x)
        if len(x)>1:
            dx[1:] = (x[1:]-x[:-1])/dt
            dx[0]  = dx[1]
        df[dcol]=dx
    N = len(df)
    need = lags+delay
    if N<need:
        raise ValueError(f"context_csv too short: need >= {need} rows, got {N}")
    # we will use indices base = N-1 (current), build window from base-delay back to L samples
    return df.tail(need).reset_index(drop=True), dt

def make_feature_stack_from_window(df_win, feat_cols, lags, delay, mu, std, device,
                                   candidate_ps=None, candidate_pd=None, ps_prev=None, pd_prev=None, dt=0.01):
    """
    Build standardized feature vector for NARX at base index = len(df_win)-1 - (0),
    but using inputs up to (base - delay). For the most recent slice (k=0),
    optionally replace p_sum/p_diff and their derivatives with candidate values.
    """
    # base index in df_win for the delayed slice
    base = len(df_win)-1 - 0  # current row
    base_del = base - delay
    if base_del < 0:
        raise ValueError("window too short vs delay")
    # stack lags from base_del back
    fv_list=[]
    for k in range(lags):
        idx = base_del - k
        if idx<0:
            raise ValueError("window too short for given lags/delay")
        row = df_win.iloc[idx][feat_cols].to_numpy().astype(np.float32)
        # overwrite most-recent slice (k==0) with candidate ps/pd if provided
        if k==0 and (candidate_ps is not None) and (candidate_pd is not None):
            # indices
            name2idx = {n:i for i,n in enumerate(feat_cols)}
            # replace p_sum / p_diff
            if "p_sum[MPa]" in name2idx: row[name2idx["p_sum[MPa]"]] = candidate_ps
            if "p_diff[MPa]" in name2idx: row[name2idx["p_diff[MPa]"]] = candidate_pd
            # replace derivatives using previous (from df or arguments)
            if ps_prev is None:
                # previous available in df_win at idx-1 (delayed-1), fall back to same if idx==0
                ps_prev = float(df_win.iloc[idx-1]["p_sum[MPa]"]) if idx-1>=0 else float(row[name2idx["p_sum[MPa]"]])
            if pd_prev is None:
                pd_prev = float(df_win.iloc[idx-1]["p_diff[MPa]"]) if idx-1>=0 else float(row[name2idx["p_diff[MPa]"]])
            dps = (candidate_ps - ps_prev)/max(dt,1e-6)
            dpd = (candidate_pd - pd_prev)/max(dt,1e-6)
            if "dp_sum[MPa/s]" in name2idx: row[name2idx["dp_sum[MPa/s]"]] = dps
            if "dp_diff[MPa/s]" in name2idx: row[name2idx["dp_diff[MPa/s]"]] = dpd
            # theta[rad], dz[m] はそのまま（現時点値を使用）
        fv_list.append(row)
    x = np.concatenate(fv_list, axis=0)[None,:]
    x_std = (x - mu)/std
    xt = torch.from_numpy(x_std).float().to(device)
    return xt

# -------------------- z static model (optional) --------------------
def fit_static_z_model(static_csv=None):
    """
    If provided, fit a simple z_stat(ps[,pd]) = b0 + b1*ps + b2*ps^2 (+ b3*|pd|)
    Return callable f(ps,pd)-> z_hat. If not provided, return proxy z_hat = alpha*ps.
    """
    if static_csv is None:
        return lambda ps, pd: float(ps)  # proxy: minimize ps
    df = pd.read_csv(static_csv)
    need = ["p_sum[MPa]","p_diff[MPa]","z[m]"]
    for c in need:
        if c not in df.columns:
            warnings.warn(f"static z model: missing col {c}, fallback to z~ps")
            return lambda ps,pd: float(ps)
    X = np.c_[ np.ones(len(df)), df["p_sum[MPa]"].to_numpy(), df["p_sum[MPa]"].to_numpy()**2,
               np.abs(df["p_diff[MPa]"].to_numpy()) ]
    y = df["z[m]"].to_numpy()
    # ridge small
    lam=1e-6
    A = X.T@X + lam*np.eye(X.shape[1])
    b = X.T@y
    coef = np.linalg.solve(A,b)
    def f(ps,pd):
        return float(coef[0] + coef[1]*ps + coef[2]*ps*ps + coef[3]*abs(pd))
    return f

# -------------------- Feasible region --------------------
def linear_constraints_ps_pd(pmax):
    """
    Enforce: 0 <= ps <= 2 pmax
             |pd| <= ps
             |pd| <= 2pmax - ps
    These 4 inequalities become:
      +pd - ps <= 0
      -pd - ps <= 0
      +pd + ps <= 2pmax
      -pd + ps <= 2pmax
    """
    A = np.array([[ 1.0, -1.0],
                  [-1.0, -1.0],
                  [ 1.0,  1.0],
                  [-1.0,  1.0]], dtype=float)
    ub = np.array([0.0, 0.0, 2*pmax, 2*pmax], dtype=float)
    lb = -np.inf*np.ones_like(ub)
    # bounds for ps in [0, 2pmax], pd free but constrained by A
    bounds = Bounds([0.0, -2*pmax],[2*pmax, 2*pmax])
    lc = LinearConstraint(A, lb, ub)
    return bounds, lc

def rate_constraints(ps_prev, pd_prev, dt, rate_ps=None, rate_pd=None):
    cons=[]
    if rate_ps is not None and rate_ps>0 and np.isfinite(ps_prev):
        def c1(x): return rate_ps*dt - abs(x[0]-ps_prev)
        cons.append({"type":"ineq", "fun": c1})
    if rate_pd is not None and rate_pd>0 and np.isfinite(pd_prev):
        def c2(x): return rate_pd*dt - abs(x[1]-pd_prev)
        cons.append({"type":"ineq","fun": c2})
    return cons

# -------------------- Objective --------------------
def build_objective(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                    theta_target, w_theta, w_ps, w_drate,
                    ps_prev, pd_prev, z_model=None):
    if z_model is None:
        z_model = lambda ps,pd: float(ps)

    def predict_theta(ps, pd):
        xt = make_feature_stack_from_window(df_win, feat_cols, lags, delay, mu, std, device,
                                            candidate_ps=ps, candidate_pd=pd,
                                            ps_prev=ps_prev, pd_prev=pd_prev, dt=dt)
        with torch.no_grad():
            y = model(xt).cpu().numpy().reshape(-1)[0]
        return float(y)

    def cost(x):
        ps, pd = float(x[0]), float(x[1])
        theta_hat = predict_theta(ps, pd)
        j_theta = (theta_hat - theta_target)**2
        j_ps    = (z_model(ps,pd))**2  # proxy: penalize z via z_model
        j_rate  = 0.0
        if np.isfinite(ps_prev): j_rate += (ps-ps_prev)**2
        if np.isfinite(pd_prev): j_rate += (pd-pd_prev)**2
        return w_theta*j_theta + w_ps*j_ps + w_drate*j_rate
    return cost

# -------------------- Solvers --------------------
def run_solvers(cost, bounds, lin_con, nonlin_cons, x0, pmax, grid=81):
    results = {}

    # helper to clamp into bounds box
    def clamp_box(x):
        x = np.array(x, dtype=float)
        x[0] = np.clip(x[0], 0.0, 2*pmax)
        x[1] = np.clip(x[1], -2*pmax, 2*pmax)
        return x

    # 0) Coarse grid search for initial seeds + a report
    ps_lin = np.linspace(0.0, 2*pmax, grid)
    pd_lim = lambda ps: min(ps, 2*pmax-ps)
    best = (None, np.inf)
    grid_pts = []
    for ps in ps_lin:
        lim = pd_lim(ps)
        pds = np.linspace(-lim, lim, grid)
        for pd in pds:
            v = cost([ps,pd])
            grid_pts.append((ps,pd,v))
            if np.isfinite(v) and v<best[1]:
                best = ([ps,pd], v)
    results["grid"] = {"x": best[0], "fun": best[1], "nit": grid*grid, "success": True, "message":"coarse grid"}
    seeds = [best[0], [pmax,0.0], [0.1,0.0], [2*pmax-0.1, 0.0]]

    # 1) SLSQP (box + linear constraints + optional rate cons)
    for i,seed in enumerate(seeds):
        try:
            r = minimize(cost, x0=clamp_box(seed), method="SLSQP",
                         bounds=bounds, constraints=[lin_con]+nonlin_cons,
                         options={"maxiter":300, "ftol":1e-9, "disp":False})
            key=f"slsqp_{i}"
            results[key] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                            "success": bool(r.success), "message": r.message}
        except Exception as e:
            results[f"slsqp_{i}"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}

    # 2) trust-constr（線形拘束＋境界）
    try:
        r = minimize(cost, x0=clamp_box(seeds[0]), method="trust-constr",
                     bounds=bounds, constraints=[lin_con],
                     options={"maxiter":300, "gtol":1e-6, "xtol":1e-6, "verbose":0})
        results["trust_constr"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                                   "success": bool(r.success), "message": r.message}
    except Exception as e:
        results["trust_constr"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}

    # 3) Powell（境界のみ対応・線形拘束はコスト中でペナルティ化が必要だが、
    #            本実装では bounds だけで回し、可行性は後でチェック）
    try:
        r = minimize(cost, x0=clamp_box(seeds[1]), method="Powell",
                     bounds=bounds, options={"maxiter":500, "xtol":1e-6, "ftol":1e-9, "disp":False})
        results["powell"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                             "success": bool(r.success), "message": r.message}
    except Exception as e:
        results["powell"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}

    # 4) Differential Evolution（大域探索）
    try:
        def feas_penalty(x):
            # linear constraints: A x <= ub  (we ignore lb=-inf)
            A = lin_con.A; ub = lin_con.ub
            v = (A@np.array(x) - ub).astype(float)
            pen = np.sum(np.clip(v, 0.0, None)**2)
            return cost(x) + 1e6*pen
        bounds_de = [(bounds.lb[0], bounds.ub[0]), (bounds.lb[1], bounds.ub[1])]
        r = differential_evolution(feas_penalty, bounds=bounds_de, maxiter=100, tol=1e-6, polish=True)
        results["de"] = {"x": r.x.tolist(), "fun": float(r.fun), "nit": r.nit,
                         "success": bool(r.success), "message": "differential_evolution"}
    except Exception as e:
        results["de"] = {"x": None, "fun": None, "nit": 0, "success": False, "message": str(e)}

    return results, np.array(grid_pts)

# -------------------- Plotting --------------------
def plot_landscape(out_png, grid_pts, pmax, theta_target, solutions, title="Inverse mapping landscape"):
    # grid_pts: (N,3) columns ps,pd,cost
    ps = grid_pts[:,0]; pd = grid_pts[:,1]; J = grid_pts[:,2]
    # build grid for contour
    # it's a triangular feasible region; we fake a grid by binning
    n = int(np.sqrt(len(ps)))
    # Scatter-based contour:
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.tricontourf(ps, pd, J, levels=30)
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Objective (lower is better)")
    # feasible polygon outlines (diamond boundaries)
    ps_line = np.linspace(0, 2*pmax, 200)
    ax.plot(ps_line,  ps_line, 'k--', lw=1)              # +pd = ps
    ax.plot(ps_line, -ps_line, 'k--', lw=1)              # -pd = ps
    ax.plot(ps_line,  2*pmax-ps_line, 'k--', lw=1)       # +pd = 2pmax - ps
    ax.plot(ps_line, -2*pmax+ps_line, 'k--', lw=1)       # -pd = 2pmax - ps
    # solutions
    for name,sol in solutions.items():
        if not sol.get("x"): continue
        x = sol["x"]
        ax.plot([x[0]],[x[1]], marker='o', ms=7, label=f"{name} (J={sol.get('fun'):.3e})")
    ax.set_xlim(0, 2*pmax); ax.set_ylim(-2*pmax, 2*pmax)
    ax.set_xlabel("p_sum [MPa]"); ax.set_ylabel("p_diff [MPa]")
    ax.set_title(f"{title}\n(theta_target={theta_target:.3f} rad)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context_csv", required=True, help="CSV with recent history to form NARX features")
    ap.add_argument("--theta_target_deg", type=float, default=None, help="target in degrees (optional)")
    ap.add_argument("--theta_target_rad", type=float, default=None, help="target in radians (optional)")
    ap.add_argument("--pmax", type=float, default=0.7)
    ap.add_argument("--w_theta", type=float, default=1.0)
    ap.add_argument("--w_ps", type=float, default=0.01)
    ap.add_argument("--w_drate", type=float, default=0.0)
    ap.add_argument("--rate_ps", type=float, default=None, help="MPa/s (optional)")
    ap.add_argument("--rate_pd", type=float, default=None, help="MPa/s (optional)")
    ap.add_argument("--z_static_csv", type=str, default=None, help="optional CSV to fit z_stat(ps,pd)")
    ap.add_argument("--out_dir", type=str, default="out_inv")
    ap.add_argument("--grid", type=int, default=81, help="coarse grid resolution (odd recommended)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    warnings.filterwarnings("ignore", message="delta_grad == 0.0", module="scipy.optimize._hessian_update_strategy")

    # target theta
    if args.theta_target_rad is None and args.theta_target_deg is None:
        raise ValueError("Provide --theta_target_rad or --theta_target_deg")
    theta_target = float(args.theta_target_rad if args.theta_target_rad is not None
                         else math.radians(args.theta_target_deg))

    # load meta/model
    meta, feat_cols, lags, delay, mu, std, hidden, dropout = load_meta(args.meta)
    in_dim = len(feat_cols)*lags
    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    model = load_model(args.model, in_dim, hidden, dropout, device)

    # context
    df_win, dt = load_context_window(args.context_csv, lags, delay, feat_cols)

    # previous pressures from the window (last delayed-1 slice)
    base = len(df_win)-1
    base_del = base - delay
    ps_prev = float(df_win.iloc[base_del-1]["p_sum[MPa]"]) if (base_del-1)>=0 else float(df_win.iloc[base_del]["p_sum[MPa]"])
    pd_prev = float(df_win.iloc[base_del-1]["p_diff[MPa]"]) if (base_del-1)>=0 else float(df_win.iloc[base_del]["p_diff[MPa]"])

    # z model (optional)
    z_model = fit_static_z_model(args.z_static_csv)

    # objective
    cost = build_objective(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                           theta_target, args.w_theta, args.w_ps, args.w_drate,
                           ps_prev, pd_prev, z_model=z_model)
    # constraints
    bounds, lin_con = linear_constraints_ps_pd(args.pmax)
    nonlin_cons = rate_constraints(ps_prev, pd_prev, dt, rate_ps=args.rate_ps, rate_pd=args.rate_pd)

    # initial guess at current ps,pd (delayed slice)
    x0 = np.array([float(df_win.iloc[base_del]["p_sum[MPa]"]),
                   float(df_win.iloc[base_del]["p_diff[MPa]"])], dtype=float)

    # run solvers
    results, grid_pts = run_solvers(cost, bounds, lin_con, nonlin_cons, x0, args.pmax, grid=args.grid)

    # enrich results with predicted theta / p1,p2 / feasibility flags
    def feasible(ps,pd,pmax):
        return (0<=ps<=2*pmax) and (abs(pd)<=min(ps, 2*pmax-ps))
    enriched = {}
    for k,v in results.items():
        if not v.get("x"):
            enriched[k]=v; continue
        ps, pd = float(v["x"][0]), float(v["x"][1])
        p1 = 0.5*(ps+pd); p2 = 0.5*(ps-pd)
        ok = feasible(ps,pd,args.pmax)
        # predict theta for the found point
        xt = make_feature_stack_from_window(df_win, feat_cols, lags, delay, mu, std, device,
                                            candidate_ps=ps, candidate_pd=pd,
                                            ps_prev=ps_prev, pd_prev=pd_prev, dt=dt)
        with torch.no_grad():
            th_hat = float(model(xt).cpu().numpy().reshape(-1)[0])
        v2 = dict(v)
        v2.update({
            "theta_hat": th_hat,
            "theta_err": th_hat - theta_target,
            "p1[MPa]": p1, "p2[MPa]": p2,
            "is_feasible": bool(ok)
        })
        enriched[k]=v2

    # choose best by objective among feasible ones
    best_name = None; best_val = np.inf
    for k,v in enriched.items():
        if v.get("x") is None or not v.get("is_feasible"): continue
        if v.get("fun") is None: continue
        if v["fun"] < best_val:
            best_val = v["fun"]; best_name = k
    enriched["best"] = {"name": best_name, "fun": best_val}

    # save JSON
    with open(os.path.join(args.out_dir,"solution_report.json"),"w") as f:
        json.dump(enriched, f, indent=2)

    # plot
    sol_points = {k:v for k,v in enriched.items() if isinstance(v,dict) and v.get("x") is not None}
    plot_landscape(os.path.join(args.out_dir,"solution.png"), grid_pts, args.pmax,
                   theta_target, sol_points, title="Inverse mapping objective")

    print(f"Saved: {os.path.join(args.out_dir,'solution_report.json')}")
    print(f"Saved: {os.path.join(args.out_dir,'solution.png')}")
    bn = enriched['best']['name']
    if bn:
        print(f"Best solver: {bn} ; J={enriched['best']['fun']:.3e}")
        bx = enriched[bn]['x']; print(f"ps*={bx[0]:.4f} MPa, pd*={bx[1]:.4f} MPa; "
                                      f"theta_hat={enriched[bn]['theta_hat']:.4f} rad")
    else:
        print("No feasible solution found (check target, weights, or constraints).")

if __name__ == "__main__":
    main()
