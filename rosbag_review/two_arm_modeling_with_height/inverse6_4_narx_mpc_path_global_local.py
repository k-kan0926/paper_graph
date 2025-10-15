#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global->Local pipeline for path-tracking MPC inverse mapping with NARX:
- Global solvers: Differential Evolution (DE), CMA-ES (pycma), Bayesian Opt (skopt)
- Local polish: SLSQP (always), trust-constr (optional)
- Keeps previous feasibility, rate limits, figures, JSON outputs.

Usage (example):
  python inverse6_4_narx_mpc_path_global_local.py \
    --meta out_narx/out_narx3_z/narx_meta.json \
    --model out_narx/out_narx3_z/narx_model.pt \
    --context_csv out/dynamic_prbs_data.csv \
    --theta_target_deg 20 \
    --horizon 12 --path smooth \
    --pmax 0.7 \
    --w_theta_term 8.0 --w_theta_path 3.0 --w_ps 0.02 --w_rate 0.05 \
    --rate_ps 2.0 --rate_pd 2.0 \
    --run_de --run_cma --run_bo \
    --out_dir model_sim/out_inv_mpc_global_local
"""

import os, json, math, argparse, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from scipy.optimize import minimize, Bounds, LinearConstraint, differential_evolution
import matplotlib.pyplot as plt

# Optional deps
HAVE_CMA = False
HAVE_SKOPT = False
try:
    import cma
    HAVE_CMA = True
except Exception:
    pass
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    HAVE_SKOPT = True
except Exception:
    pass

# ---------------- NARX ----------------
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
    lags      = int(meta["lags"]); delay = int(meta["delay"])
    mu        = np.array(meta["mu"], dtype=np.float32)
    std       = np.array(meta["std"], dtype=np.float32)
    hidden    = int(meta.get("hidden",128)); dropout=float(meta.get("dropout",0.0))
    fz_meta   = meta.get("fz_ps_quadratic", None)
    return meta, feat_cols, lags, delay, mu, std, hidden, dropout, fz_meta

def load_model(model_path, in_dim, hidden, dropout, device):
    m = MLP_NARX(in_dim, hidden=[hidden,hidden], out_dim=1, dropout=dropout).to(device)
    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(model_path, map_location=device)
    m.load_state_dict(sd); m.eval()
    return m

# -------------- Context & features --------------
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
    if miss: raise ValueError(f"context_csv missing: {miss}")
    dt = estimate_dt(df)
    for col, dcol in [("p_sum[MPa]","dp_sum[MPa/s]"),("p_diff[MPa]","dp_diff[MPa/s]")]:
        x = df[col].to_numpy(); dx = np.zeros_like(x)
        if len(x)>1:
            dx[1:] = (x[1:]-x[:-1])/dt; dx[0]=dx[1]
        df[dcol]=dx
    need = lags+delay
    if len(df)<need: raise ValueError(f"context too short: need >= {need}")
    return df.tail(need).reset_index(drop=True), dt

def build_xt_from_window(df_win, feat_cols, lags, delay, mu, std, device,
                         ps_now=None, pd_now=None, ps_prev=None, pd_prev=None, dt=0.01,
                         theta_used_override=None):
    base=len(df_win)-1; b_del=base-delay
    fv=[]; name2idx={n:i for i,n in enumerate(feat_cols)}
    for k in range(lags):
        idx=b_del-k
        row = df_win.iloc[idx][feat_cols].to_numpy().astype(np.float32)
        if theta_used_override is not None and "theta[rad]" in name2idx:
            row[name2idx["theta[rad]"]] = float(theta_used_override[idx])
        if k==0 and ps_now is not None and pd_now is not None:
            if "p_sum[MPa]" in name2idx:  row[name2idx["p_sum[MPa]"]] = ps_now
            if "p_diff[MPa]" in name2idx: row[name2idx["p_diff[MPa]"]] = pd_now
            psp = float(df_win.iloc[idx-1]["p_sum[MPa]"]) if ps_prev is None else ps_prev
            pdp = float(df_win.iloc[idx-1]["p_diff[MPa]"]) if pd_prev is None else pd_prev
            dps=(ps_now-psp)/max(dt,1e-6); dpd=(pd_now-pdp)/max(dt,1e-6)
            if "dp_sum[MPa/s]" in name2idx:  row[name2idx["dp_sum[MPa/s]"]]  = dps
            if "dp_diff[MPa/s]" in name2idx: row[name2idx["dp_diff[MPa/s]"]] = dpd
        fv.append(row)
    x=np.concatenate(fv,axis=0)[None,:]
    x_std=(x-mu)/std
    return torch.from_numpy(x_std).float().to(device)

# -------------- z model --------------
def z_model_from_meta(fz_meta):
    if fz_meta is None: return None
    a0=float(fz_meta.get("a0",0.0)); a1=float(fz_meta.get("a1",0.0)); a2=float(fz_meta.get("a2",0.0))
    def f(ps,pd): return a0 + a1*ps + a2*(ps**2)
    return f

# -------------- Feasible region --------------
def bounds_lin_for_ps_pd(pmax):
    A=np.array([[ 1,-1],[-1,-1],[ 1, 1],[-1, 1]],float)
    ub=np.array([0,0,2*pmax,2*pmax],float); lb=-np.inf*np.ones_like(ub)
    lc = LinearConstraint(A, lb, ub)
    bounds = Bounds([0,-2*pmax],[2*pmax,2*pmax])
    return bounds, lc

# -------------- Rollout & objective --------------
def rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt, seq_ps, seq_pd):
    H=len(seq_ps)
    theta_used=df_win["theta[rad]"].to_numpy().astype(np.float32).copy()
    preds=[]
    base=len(df_win)-1; b_del=base-delay
    ps_prev=float(df_win.iloc[b_del-1]["p_sum[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_sum[MPa]"])
    pd_prev=float(df_win.iloc[b_del-1]["p_diff[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_diff[MPa]"])
    for k in range(H):
        xt=build_xt_from_window(df_win, feat_cols, lags, delay, mu, std, device,
                                ps_now=float(seq_ps[k]), pd_now=float(seq_pd[k]),
                                ps_prev=ps_prev, pd_prev=pd_prev, dt=dt,
                                theta_used_override=theta_used)
        with torch.no_grad():
            th=float(model(xt).cpu().numpy().reshape(-1)[0])
        preds.append(th)
        theta_used=np.roll(theta_used,-1); theta_used[-1]=th
        ps_prev=float(seq_ps[k]); pd_prev=float(seq_pd[k])
    return np.array(preds,float)

def make_theta_path(theta0, theta_star, H, kind="smooth"):
    a=np.linspace(1/H,1.0,H)
    s=0.5*(1-np.cos(np.pi*a)) if kind=="smooth" else a
    return theta0 + s*(theta_star-theta0)

def build_objective(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                    theta_target, H, w_term, w_path, w_ps, w_rate,
                    z_model, pmax, ps_prev0, pd_prev0, theta_path_kind):
    theta0=float(df_win["theta[rad]"].iloc[-1])
    theta_ref=make_theta_path(theta0, theta_target, H, theta_path_kind)
    def cost_flat(x):
        x=np.asarray(x,float); ps_seq=x[0::2]; pd_seq=x[1::2]
        pen=0.0
        for k in range(H):
            ps, pd = ps_seq[k], pd_seq[k]
            if ps<0: pen+=(0-ps)**2
            if ps>2*pmax: pen+=(ps-2*pmax)**2
            if abs(pd)>ps: pen+=(abs(pd)-ps)**2
            if abs(pd)>(2*pmax-ps): pen+=(abs(pd)-(2*pmax-ps))**2
        theta_roll=rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt, ps_seq, pd_seq)
        j_term=(theta_roll[-1]-theta_target)**2
        j_path=np.sum((theta_roll-theta_ref)**2)
        j_z=0.0
        if z_model is not None:
            for k in range(H): j_z += z_model(ps_seq[k], pd_seq[k])**2
        j_rate=0.0
        for k in range(H):
            rps = ps_prev0 if k==0 else ps_seq[k-1]
            rpd = pd_prev0 if k==0 else pd_seq[k-1]
            j_rate += (ps_seq[k]-rps)**2 + (pd_seq[k]-rpd)**2
        return w_term*j_term + w_path*j_path + w_ps*j_z + w_rate*j_rate + 1e6*pen
    return cost_flat, theta_ref

# -------------- Plots --------------
def plot_landscape_first_step(out_png, cost_first_step, pmax, theta_target):
    ps_lin=np.linspace(0,2*pmax,161); pts_ps=[]; pts_pd=[]; pts_J=[]
    for ps in ps_lin:
        lim=min(ps,2*pmax-ps); pds=np.linspace(-lim,lim,161)
        for pd in pds:
            pts_ps.append(ps); pts_pd.append(pd); pts_J.append(cost_first_step(ps,pd))
    ps=np.array(pts_ps); pd=np.array(pts_pd); J=np.array(pts_J)
    fig,ax=plt.subplots(figsize=(8,6))
    cf=ax.tricontourf(ps,pd,J,levels=30); cbar=plt.colorbar(cf,ax=ax); cbar.set_label("Objective (first-step proxy)")
    line=np.linspace(0,2*pmax,200)
    ax.plot(line, line,'k--',lw=1); ax.plot(line,-line,'k--',lw=1)
    ax.plot(line, 2*pmax-line,'k--',lw=1); ax.plot(line,-2*pmax+line,'k--',lw=1)
    ax.set_xlim(0,2*pmax); ax.set_ylim(-2*pmax,2*pmax)
    ax.set_xlabel("p_sum [MPa]"); ax.set_ylabel("p_diff [MPa]")
    ax.set_title(f"First-step landscape (theta*={theta_target:.3f} rad)")
    fig.tight_layout(); fig.savefig(out_png,dpi=160); plt.close(fig)

def plot_trajectories(out_png, theta_roll, theta_ref, theta_target, ps_seq, pd_seq):
    H=len(ps_seq); t=np.arange(1,H+1)
    fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
    ax[0].plot(t,theta_roll,marker='o',label="theta_hat")
    ax[0].plot(t,theta_ref, ls='--',label="theta_ref")
    ax[0].axhline(theta_target,ls=':',label="theta*"); ax[0].legend(); ax[0].set_ylabel("theta [rad]")
    ax[1].plot(t,ps_seq,marker='o',label="p_sum"); ax[1].plot(t,pd_seq,marker='o',label="p_diff")
    ax[1].legend(); ax[1].set_xlabel("step"); ax[1].set_ylabel("MPa")
    fig.tight_layout(); fig.savefig(out_png,dpi=160); plt.close(fig)

# -------------- Main --------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--meta",required=True); ap.add_argument("--model",required=True); ap.add_argument("--context_csv",required=True)
    ap.add_argument("--theta_target_deg",type=float,default=None); ap.add_argument("--theta_target_rad",type=float,default=None)
    ap.add_argument("--horizon",type=int,default=10); ap.add_argument("--path",choices=["ramp","smooth"],default="smooth")
    ap.add_argument("--pmax",type=float,default=0.7)
    ap.add_argument("--w_theta_term",type=float,default=8.0)
    ap.add_argument("--w_theta_path",type=float,default=3.0)
    ap.add_argument("--w_ps",type=float,default=0.02)
    ap.add_argument("--w_rate",type=float,default=0.05)
    ap.add_argument("--rate_ps",type=float,default=2.0)
    ap.add_argument("--rate_pd",type=float,default=2.0)
    ap.add_argument("--out_dir",type=str,default="out_inv_mpc_global_local")
    ap.add_argument("--cpu",action="store_true")
    # which global solvers to run
    ap.add_argument("--run_de",action="store_true")
    ap.add_argument("--run_cma",action="store_true")
    ap.add_argument("--run_bo",action="store_true")
    # options
    ap.add_argument("--de_maxiter",type=int,default=150)
    ap.add_argument("--cma_restarts",type=int,default=2)
    ap.add_argument("--bo_calls",type=int,default=120)
    args=ap.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)

    theta_target = (args.theta_target_rad if args.theta_target_rad is not None
                    else math.radians(args.theta_target_deg))

    meta, feat_cols, lags, delay, mu, std, hidden, dropout, fz_meta = load_meta(args.meta)
    device=torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    model=load_model(args.model, in_dim=len(feat_cols)*lags, hidden=hidden, dropout=dropout, device=device)
    df_win, dt = load_context_window(args.context_csv, lags, delay, feat_cols)

    base=len(df_win)-1; b_del=base-delay
    ps_prev0=float(df_win.iloc[b_del-1]["p_sum[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_sum[MPa]"])
    pd_prev0=float(df_win.iloc[b_del-1]["p_diff[MPa]"]) if (b_del-1)>=0 else float(df_win.iloc[b_del]["p_diff[MPa]"])

    z_model = z_model_from_meta(fz_meta)  # meta優先（必要ならCSV回帰を追加実装可）

    H=args.horizon; pmax=args.pmax
    cost_flat, theta_ref = build_objective(model, df_win, feat_cols, lags, delay, mu, std, device, dt,
                                           theta_target, H, args.w_theta_term, args.w_theta_path,
                                           args.w_ps, args.w_rate, z_model, pmax,
                                           ps_prev0, pd_prev0, args.path)

    # block constraints & bounds for local solvers
    A=np.array([[ 1,-1],[-1,-1],[ 1, 1],[-1, 1]],float)
    ub=np.array([0,0,2*pmax,2*pmax],float); lb=-np.inf*np.ones_like(ub)
    Ablk=np.zeros((4*H,2*H)); 
    for k in range(H): Ablk[4*k:4*k+4,2*k:2*k+2]=A
    lc_all = LinearConstraint(Ablk, np.tile(lb,H), np.tile(ub,H))
    bounds = Bounds([0,-2*pmax]*H, [2*pmax,2*pmax]*H)

    # seed (hold)
    ps0=float(df_win.iloc[b_del]["p_sum[MPa]"]); pd0=float(df_win.iloc[b_del]["p_diff[MPa]"])
    x0=np.array([v for _ in range(H) for v in (ps0,pd0)],float)

    def unpack(x): x=np.asarray(x,float); return x[0::2], x[1::2]
    def enrich_record(rec):
        if rec.get("x") is None: return rec
        x=np.array(rec["x"],float); ps_seq,pd_seq=unpack(x)
        theta_roll=rollout_theta(model, df_win, feat_cols, lags, delay, mu, std, device, dt, ps_seq, pd_seq)
        j_term=(theta_roll[-1]-theta_target)**2
        j_path=np.sum((theta_roll-theta_ref)**2)
        j_z=0.0
        if z_model is not None:
            for k in range(H): j_z += z_model(ps_seq[k], pd_seq[k])**2
        j_rate=0.0
        for k in range(H):
            rps=ps_prev0 if k==0 else ps_seq[k-1]
            rpd=pd_prev0 if k==0 else pd_seq[k-1]
            j_rate += (ps_seq[k]-rps)**2 + (pd_seq[k]-rpd)**2
        rec.update({
            "theta_roll": theta_roll.tolist(),
            "theta_ref": theta_ref.tolist(),
            "theta_terminal": float(theta_roll[-1]),
            "theta_err_terminal": float(theta_roll[-1]-theta_target),
            "ps_seq": ps_seq.tolist(),
            "pd_seq": pd_seq.tolist(),
            "J_term": float(j_term), "J_path": float(j_path),
            "J_z": float(j_z), "J_rate": float(j_rate)
        })
        return rec

    # First-step landscape
    def cost_first_step(ps,pd):
        x=x0.copy(); x[0]=ps; x[1]=pd; return cost_flat(x)
    plot_landscape_first_step(os.path.join(args.out_dir,"landscape.png"), cost_first_step, pmax, theta_target)

    results = {}

    # ---------- Global: DE ----------
    if args.run_de:
        try:
            bnds=[(0,2*pmax), (-2*pmax,2*pmax)]*H
            r=differential_evolution(lambda x: cost_flat(x), bounds=bnds,
                                     maxiter=args.de_maxiter, tol=1e-6, polish=True)
            results["de_raw"]={"x":r.x.tolist(),"fun":float(r.fun),"nit":r.nit,"success":bool(r.success),"message":"de"}
        except Exception as e:
            results["de_raw"]={"x":None,"fun":None,"nit":0,"success":False,"message":str(e)}

    # ---------- Global: CMA-ES ----------
    if args.run_cma:
        if not HAVE_CMA:
            warnings.warn("CMA-ES requested but `cma` not installed; skipping.")
        else:
            try:
                lb=np.array([0,-2*pmax]*H); ub=np.array([2*pmax,2*pmax]*H)
                sigma = 0.2*(ub-lb).mean()  # coarse
                es = cma.CMAEvolutionStrategy(x0, sigma,
                      {'bounds':[lb.tolist(), ub.tolist()], 'maxiter':200, 'verb_disp':0})
                x_best=None; f_best=np.inf
                for _ in range(200):
                    X = es.ask()
                    F = [cost_flat(xi) for xi in X]
                    es.tell(X,F)
                    if es.result.fbest < f_best:
                        x_best = es.result.xbest; f_best = es.result.fbest
                    if es.stop(): break
                # simple restarts
                for _ in range(max(0,args.cma_restarts)):
                    es = cma.CMAEvolutionStrategy(x_best, sigma*0.5,
                          {'bounds':[lb.tolist(), ub.tolist()], 'maxiter':120, 'verb_disp':0})
                    es.optimize(cost_flat)
                    if es.result.fbest < f_best:
                        x_best = es.result.xbest; f_best = es.result.fbest
                results["cma_raw"]={"x":np.array(x_best).tolist(),"fun":float(f_best),
                                    "nit":int(es.result.iterations),"success":True,"message":"cma-es"}
            except Exception as e:
                results["cma_raw"]={"x":None,"fun":None,"nit":0,"success":False,"message":str(e)}

    # ---------- Global: Bayesian Optimization ----------
    if args.run_bo:
        if not HAVE_SKOPT:
            warnings.warn("BayesOpt requested but `scikit-optimize` not installed; skipping.")
        else:
            try:
                space=[]
                for k in range(H):
                    space.append(Real(0,2*pmax, name=f"ps{k}"))
                    space.append(Real(-2*pmax,2*pmax, name=f"pd{k}"))
                @use_named_args(space)
                def fobj(**kwargs):
                    x=np.array([kwargs[f"ps{k}"] if i%2==0 else kwargs[f"pd{k//2}"]
                                for i in range(2*H) for k in ([i//2] if i%2==0 else [i//2])], float)
                    return float(cost_flat(x))
                r = gp_minimize(fobj, dimensions=space, n_calls=args.bo_calls,
                                n_initial_points=min(20, args.bo_calls//4), acq_func="EI", noise=1e-10, verbose=False)
                x_best=np.array(r.x, float)
                results["bo_raw"]={"x":x_best.tolist(),"fun":float(r.fun),"nit":len(r.func_vals),
                                   "success":True,"message":"gp_minimize"}
            except Exception as e:
                results["bo_raw"]={"x":None,"fun":None,"nit":0,"success":False,"message":str(e)}

    # ---------- Local polish (SLSQP + trust) ----------
    def polish(name_raw, name_polish):
        r_raw=results.get(name_raw)
        if not r_raw or (r_raw.get("x") is None): 
            results[name_polish]={"x":None,"fun":None,"nit":0,"success":False,"message":"no raw"}
            return
        x_start=np.array(r_raw["x"],float)
        try:
            r=minimize(cost_flat, x0=x_start, method="SLSQP",
                       bounds=bounds, constraints=[lc_all],
                       options={"maxiter":400,"ftol":1e-9,"disp":False})
            results[name_polish]={"x":r.x.tolist(),"fun":float(r.fun),"nit":r.nit,
                                  "success":bool(r.success),"message":r.message}
        except Exception as e:
            results[name_polish]={"x":None,"fun":None,"nit":0,"success":False,"message":str(e)}
        # also trust-constr as reference
        try:
            r=minimize(cost_flat, x0=x_start, method="trust-constr",
                       bounds=bounds, constraints=[lc_all],
                       hess=lambda x: np.zeros((2*H,2*H)),
                       options={"maxiter":400,"gtol":1e-6,"xtol":1e-6})
            results[name_polish+"_trust"]={"x":r.x.tolist(),"fun":float(r.fun),"nit":r.nit,
                                           "success":bool(r.success),"message":r.message}
        except Exception as e:
            results[name_polish+"_trust"]={"x":None,"fun":None,"nit":0,"success":False,"message":str(e)}

    for raw, pol in [("de_raw","de_polished"), ("cma_raw","cma_polished"), ("bo_raw","bo_polished")]:
        polish(raw, pol)

    # ---------- Always include Powell & plain SLSQP as baselines ----------
    try:
        r=minimize(cost_flat, x0=x0, method="Powell",
                   bounds=bounds, options={"maxiter":700,"xtol":1e-6,"ftol":1e-9,"disp":False})
        results["powell"]={"x":r.x.tolist(),"fun":float(r.fun),"nit":r.nit,"success":bool(r.success),"message":r.message}
    except Exception as e:
        results["powell"]={"x":None,"fun":None,"nit":0,"success":False,"message":str(e)}
    try:
        r=minimize(cost_flat, x0=x0, method="SLSQP",
                   bounds=bounds, constraints=[lc_all],
                   options={"maxiter":400,"ftol":1e-9,"disp":False})
        results["slsqp"]={"x":r.x.tolist(),"fun":float(r.fun),"nit":r.nit,"success":bool(r.success),"message":r.message}
    except Exception as e:
        results["slsqp"]={"x":None,"fun":None,"nit":0,"success":False,"message":str(e)}

    # ---------- Enrich & select best ----------
    enriched={}
    best_name=None; best_val=np.inf
    for k,v in results.items():
        enriched[k]=enrich_record(dict(v))
        f=v.get("fun")
        if f is not None and np.isfinite(f) and f<best_val:
            best_val=f; best_name=k
    enriched["best"]={"name":best_name,"fun":best_val}

    with open(os.path.join(args.out_dir,"report.json"),"w") as f:
        json.dump(enriched,f,indent=2)

    # plot best traj
    if best_name and enriched[best_name].get("x"):
        x=np.array(enriched[best_name]["x"],float); ps, pd = x[0::2], x[1::2]
        theta_roll=np.array(enriched[best_name]["theta_roll"],float)
        plot_trajectories(os.path.join(args.out_dir,"trajectories.png"),
                          theta_roll, np.array(enriched[best_name]["theta_ref"],float),
                          theta_target, ps, pd)

    print(f"Saved: {os.path.join(args.out_dir,'report.json')}")
    print(f"Saved: {os.path.join(args.out_dir,'landscape.png')}")
    print(f"Best: {best_name} (J={best_val:.3e})" if best_name else "No solution found.")

if __name__=="__main__":
    main()
