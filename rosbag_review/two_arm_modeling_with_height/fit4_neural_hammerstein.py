#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Hammerstein (monotone-in-Delta static NN) + delay/1st-order dynamics fit.

Usage:
  python fit4_neural_hammerstein.py --csv out/diff_run1_h_data.csv --out-prefix out/model_k4/model_k4 \--dt 0.01 --delay-grid 0,1,2,3 --epochs 400 --batch-size 512 --lr 1e-3 --M 6 --hidden 32 --val-ratio 0.2 --sg-win 17 --l2-dyn 5e-3 --kdelta-min -2.0

Outputs:
  out-prefix + "_nh.pt"         (torch state_dict)
  out-prefix + "_nh_meta.npz"   (meta/dynamics/normalizers)
"""
import argparse, os, json, math
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------- helpers ----------
def smooth_and_deriv(x, dt, win=17, poly=3):
    x = np.asarray(x, float)
    N = len(x)
    win = min(win, (N//2)*2-1)
    if win < 5:
        return x, np.gradient(x, dt)
    try:
        from scipy.signal import savgol_filter
        xs = savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
        dx = savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=dt, mode="interp")
        return xs, dx
    except Exception:
        k = win; w = np.ones(k)/k
        xs = np.convolve(x, w, mode="same")
        dx = np.gradient(xs, dt)
        return xs, dx

def rollout_rmse(ps, pd, th, dt, d, alpha, kS, kD, theta_stat_fn):
    N = len(ps)
    start = d + 1
    th_pred = th.copy()
    for k in range(start, N-1):
        ku = k - d; kup = k - d - 1
        ths = float(theta_stat_fn(ps[ku], pd[ku]))
        dS  = (ps[ku] - ps[kup]) / dt
        dD  = (pd[ku] - pd[kup]) / dt
        rhs = alpha*(ths - th_pred[k]) + kS*dS + kD*dD
        th_pred[k+1] = th_pred[k] + dt*rhs
    err = th_pred[start:] - th[start:]
    return float(np.sqrt(np.mean(err**2)))

# ---------- NN (monotone in Delta) ----------
class MonoDeltaNN(nn.Module):
    """
    θ_stat(Σ,Δ) = A(Σ) + Σ_j softplus(w_j(Σ)) * tanh( softplus(s_j(Σ)) * (Δ̂ - c_j) )
    where Δ̂ is normalized Δ, centers c_j are fixed grid in normalized Δ.
    """
    def __init__(self, M=6, hidden=32, c_grid=None):
        super().__init__()
        self.M = M
        self.c = nn.Parameter(torch.tensor(c_grid, dtype=torch.float32), requires_grad=False)
        self.enc = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU()
        )
        self.head_A = nn.Linear(hidden, 1)        # A(Σ)
        self.head_w = nn.Linear(hidden, M)        # -> softplus >=0
        self.head_s = nn.Linear(hidden, M)        # -> softplus >=0
        # tiny bias so softplus(...) is not zero at init
        nn.init.zeros_(self.head_w.weight); nn.init.zeros_(self.head_w.bias)
        nn.init.zeros_(self.head_s.weight); nn.init.constant_(self.head_s.bias, math.log(math.e-1.0))

    def forward(self, Sigma_hat, Delta_hat):
        # Sigma_hat, Delta_hat: shape [N,1]
        h = self.enc(Sigma_hat)
        A = self.head_A(h)                        # [N,1]
        w = torch.nn.functional.softplus(self.head_w(h)) + 1e-6  # [N,M] >=0
        s = torch.nn.functional.softplus(self.head_s(h)) + 1e-6  # [N,M] >=0
        # Δ̂ - c_j: broadcast
        # [N, M] = [N,1] - [M] -> add batch dimension
        d = Delta_hat - self.c.view(1, -1)       # [N,M]
        bank = torch.tanh(s * d)                  # [N,M], tanh is increasing
        out = A + (w * bank).sum(dim=1, keepdim=True)  # [N,1]
        return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--sg-win", type=int, default=17)
    ap.add_argument("--sg-poly", type=int, default=3)
    ap.add_argument("--delay-grid", type=str, default="0,1,2,3")
    ap.add_argument("--l2-dyn", type=float, default=5e-3)
    ap.add_argument("--kdelta-min", type=float, default=-2.0)
    ap.add_argument("--pmax", type=float, default=None, help="override pmax in csv meta (optional)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # load csv
    df = pd.read_csv(args.csv, comment="#")
    for col in ["p_sum[MPa]","p_diff[MPa]","theta[deg]"]:
        if col not in df.columns:
            raise RuntimeError(f"missing column {col}")
    ps = df["p_sum[MPa]"].to_numpy(float)
    pdv= df["p_diff[MPa]"].to_numpy(float)
    th = df["theta[deg]"].to_numpy(float)

    # infer dt
    if args.dt is not None:
        dt = float(args.dt)
    elif "time[s]" in df.columns:
        t = df["time[s]"].to_numpy(float)
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = 0.01

    # normalizers
    muS, sdS = float(np.mean(ps)), float(np.std(ps)+1e-8)
    muD, sdD = float(np.mean(pdv)), float(np.std(pdv)+1e-8)
    S_hat = (ps - muS)/sdS
    D_hat = (pdv- muD)/sdD

    # Δ centers (normalized space)
    M = int(args.M)
    c_grid = np.linspace(np.percentile(D_hat, 5), np.percentile(D_hat, 95), M).astype(np.float32)

    # train/val split（後方を検証に）
    N = len(ps)
    val_n = max( int(N*args.val_ratio), 1 )
    tr_idx = np.arange(0, N-val_n)
    va_idx = np.arange(N-val_n, N)

    Xtr_S = torch.tensor(S_hat[tr_idx], dtype=torch.float32).view(-1,1)
    Xtr_D = torch.tensor(D_hat[tr_idx], dtype=torch.float32).view(-1,1)
    ytr   = torch.tensor(th[tr_idx],    dtype=torch.float32).view(-1,1)

    Xva_S = torch.tensor(S_hat[va_idx], dtype=torch.float32).view(-1,1)
    Xva_D = torch.tensor(D_hat[va_idx], dtype=torch.float32).view(-1,1)
    yva   = torch.tensor(th[va_idx],    dtype=torch.float32).view(-1,1)

    ds_tr = TensorDataset(Xtr_S, Xtr_D, ytr)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # model & opt
    model = MonoDeltaNN(M=M, hidden=int(args.hidden), c_grid=c_grid)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    loss_fn = torch.nn.HuberLoss(delta=3.0)  # robust to outliers

    # train
    best = (1e18, None)
    patience, wait = 40, 0
    for ep in range(args.epochs):
        model.train()
        for Sb,Db,yb in dl_tr:
            opt.zero_grad()
            yhat = model(Sb, Db)
            loss = loss_fn(yhat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            yhat_va = model(Xva_S, Xva_D)
            val_rmse = float(torch.sqrt(((yhat_va - yva)**2).mean()))
        print(f"[ep {ep:3d}] val RMSE={val_rmse:.3f} deg")
        if val_rmse < best[0] - 1e-3:
            best = (val_rmse, {k:v.cpu().clone() for k,v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break
    # load best
    if best[1] is not None:
        model.load_state_dict(best[1])

    # ----- dynamics fit (α,kΣ,kΔ, delay d) -----
    # smooth & derivs for rollout
    ps_s, _ = smooth_and_deriv(ps, dt, win=args.sg_win, poly=args.sg_poly)
    pd_s, _ = smooth_and_deriv(pdv, dt, win=args.sg_win, poly=args.sg_poly)
    th_s, dth = smooth_and_deriv(th, dt, win=args.sg_win, poly=args.sg_poly)

    def theta_stat_fn(ps_val, pd_val):
        S = torch.tensor([(ps_val-muS)/sdS], dtype=torch.float32).view(1,1)
        D = torch.tensor([(pd_val-muD)/sdD], dtype=torch.float32).view(1,1)
        with torch.no_grad():
            y = model(S,D).item()
        return y

    # build X,y per delay & solve with bounds
    try:
        from scipy.optimize import lsq_linear
    except Exception:
        raise RuntimeError("scipy が必要です（lsq_linear を使用）。")

    delays = [int(x.strip()) for x in args.delay_grid.split(",") if x.strip()!=""]
    best_dyn = None
    results = []
    for d in delays:
        start = d + 1
        ks = np.arange(start, N)
        ku = ks - d
        kup= ks - d - 1
        # features
        ths = np.array([theta_stat_fn(ps_s[i], pd_s[i]) for i in ku], float)
        phi1 = ths - th_s[ks]
        dS   = (ps_s[ku] - ps_s[kup]) / dt
        dD   = (pd_s[ku] - pd_s[kup]) / dt
        y    = dth[ks]

        X = np.column_stack([phi1, dS, dD])
        # standardize columns
        scales = np.std(X, axis=0, ddof=1); scales[scales<1e-8] = 1.0
        Xs = X / scales

        lam = float(args.l2_dyn)
        X_aug = np.vstack([Xs, math.sqrt(lam)*np.eye(3)])
        y_aug = np.hstack([y, np.zeros(3)])

        lb = [0.0, -np.inf, float(args.kdelta_min)]
        ub = [np.inf, np.inf, np.inf]
        sol = lsq_linear(X_aug, y_aug, bounds=(lb,ub), method="trf", lsmr_tol='auto', verbose=0)
        alpha, kS, kD = (sol.x / scales).tolist()

        rmse = rollout_rmse(ps_s, pd_s, th_s, d, alpha, kS, kD, theta_stat_fn)
        results.append((d, alpha, kS, kD, rmse))
        if (best_dyn is None) or (rmse < best_dyn[-1]):
            best_dyn = (d, alpha, kS, kD, rmse)

    print("=== Neural Hammerstein + delay fit ===")
    for d, A,Ks,Kd,R in results:
        print(f" d={d}: alpha={A:.4f} [1/s] (T={1.0/max(A,1e-9):.3f}s), kΣ={Ks:.4f}, kΔ={Kd:.4f}, RMSE_rollout={R:.3f} deg")
    d, alpha, kS, kD, rmse = best_dyn
    print(f" -> selected d={d}  (RMSE_rollout={rmse:.3f} deg)")

    # ----- save -----
    pt_path = args.out_prefix + "_nh.pt"
    torch.save(model.state_dict(), pt_path)

    meta = dict(
        dt=dt, muS=muS, sdS=sdS, muD=muD, sdD=sdD,
        c_grid=c_grid.tolist(), M=M, hidden=int(args.hidden),
        alpha=float(alpha), kSigma=float(kS), kDelta=float(kD), delay=int(d),
        rmse_rollout=float(rmse),
        pmax=float(args.pmax) if args.pmax is not None else float(np.nanmax(ps)*1.05),
        sigma_ref=float(np.median(ps)),
        sg_win=int(args.sg_win), l2_dyn=float(args.l2_dyn), kdelta_min=float(args.kdelta_min)
    )
    npz_path = args.out_prefix + "_nh_meta.npz"
    np.savez(npz_path, **{k:np.array(v) if isinstance(v,list) else v for k,v in meta.items()})
    print(f"[OK] saved: {pt_path}")
    print(f"[OK] saved: {npz_path}")
    print(json.dumps({k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in meta.items() if k not in ['c_grid']}, indent=2))

if __name__ == "__main__":
    main()
