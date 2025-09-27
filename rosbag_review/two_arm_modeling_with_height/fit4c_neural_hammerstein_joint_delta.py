#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Hammerstein with joint options:

(A) learnable Delta centers (on/off)
(B) slope floor penalty in Delta direction (on/off)
(C) one-step dynamic consistency loss (on/off, warmup/ramp/batch/refit)
(D) Delta-slope sign consistency:
    - hard:  kDelta <= 0   (--kdelta-hard-neg)
    - soft:  lambda * max(0, kDelta_est)  (--kdelta-sign-penalty L)

Usage examples:
  # baseline
  python fit4b_neural_hammerstein_joint.py --csv out/diff_run1_h_data.csv --out-prefix out/model_k4b/model_k4b

  # (A)+(B)
  python fit4b_neural_hammerstein_joint.py --csv out/diff_run1_h_data.csv --out-prefix out/model_k4b/model_k4b \
    --learn-centers --slope-floor --slope-eps 0.05 --slope-lambda 1e-3

  # (C) with delay search & dynamic ridge
  python fit4b_neural_hammerstein_joint.py --csv out/diff_run1_h_data.csv --out-prefix out/model_k4b/model_k4b \
    --one-step --one-step-warmup 50 --one-step-refit 20 \
    --one-step-weight 0.03 --one-step-ramp 20 --one-step-batch 2048 \
    --delay-grid 0,1,2,3,4,5 --l2-dyn 5e-3

  # (D) hard + soft sign options
  python fit4b_neural_hammerstein_joint.py --csv out/diff_run1_h_data.csv --out-prefix out/model_k4b/model_k4b \
    --one-step --kdelta-hard-neg --kdelta-sign-penalty 1e-2
"""

import argparse, os, json, math
import numpy as np, pandas as pd

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ----------------- helpers -----------------
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

def fit_dynamics_with_delay(theta_stat_fn, ps, pd, th, dt, sg_win, sg_poly,
                            delay_grid, l2_dyn, kdelta_min,
                            hard_neg=False):
    """最小二乗で α,kΣ,kΔ を推定。delay はグリッド探索で最良RMSEを選択。
       hard_neg=True で kΔ<=0 をハード拘束。"""
    try:
        from scipy.optimize import lsq_linear
    except Exception:
        raise RuntimeError("scipy が必要です（lsq_linear を使用）。")

    ps_s, _  = smooth_and_deriv(ps, dt, win=sg_win, poly=sg_poly)
    pd_s, _  = smooth_and_deriv(pd, dt, win=sg_win, poly=sg_poly)
    th_s, dth= smooth_and_deriv(th, dt, win=sg_win, poly=sg_poly)

    delays = [int(x.strip()) for x in delay_grid.split(",") if x.strip()!=""]
    best = None; results = []
    N = len(ps)

    for d in delays:
        if d < 0: continue
        start = d+1
        if start+1 >= N:  # too short
            continue
        ks = np.arange(start, N); ku = ks - d; kup = ks - d - 1
        ths = np.array([theta_stat_fn(ps_s[i], pd_s[i]) for i in ku], float)
        phi1= ths - th_s[ks]
        dS  = (ps_s[ku] - ps_s[kup]) / dt
        dD  = (pd_s[ku] - pd_s[kup]) / dt
        y   = dth[ks]
        X   = np.column_stack([phi1, dS, dD])  # [N,3]

        # 標準化
        scales = np.std(X, axis=0, ddof=1); scales[scales<1e-8]=1.0
        Xs = X / scales

        # リッジ（ダミー行）
        lam = float(l2_dyn)
        X_aug = np.vstack([Xs, math.sqrt(lam)*np.eye(3)])
        y_aug = np.hstack([y, np.zeros(3)])

        # 境界
        lb = [0.0, -np.inf, float(kdelta_min)]
        ub = [np.inf, np.inf, (0.0 if hard_neg else np.inf)]  # ★ hard_neg で kΔ<=0
        sol = lsq_linear(X_aug, y_aug, bounds=(lb,ub), method="trf", lsmr_tol='auto', verbose=0)
        alpha, kS, kD = (sol.x / scales).tolist()

        rmse = rollout_rmse(ps_s, pd_s, th_s, dt, d, alpha, kS, kD, theta_stat_fn)
        results.append((d, alpha, kS, kD, rmse))
        if (best is None) or (rmse < best[-1]):
            best = (d, alpha, kS, kD, rmse)

    return best, results

# ----------------- NN: monotone in Delta -----------------
class MonoDeltaNN(nn.Module):
    """
    θ_stat(Σ,Δ) = A(Σ̂) + Σ_j softplus(w_j(Σ̂)) * tanh( softplus(s_j(Σ̂)) * (Δ̂ - c_j) )
    Δ̂, Σ̂ are normalized. If learn_centers=True, c_j are trainable.
    """
    def __init__(self, M=6, hidden=32, c_grid=None, learn_centers=False):
        super().__init__()
        self.M = M
        self.c = nn.Parameter(torch.tensor(c_grid, dtype=torch.float32),
                              requires_grad=bool(learn_centers))
        self.enc = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU()
        )
        self.head_A = nn.Linear(hidden, 1)
        self.head_w = nn.Linear(hidden, M)
        self.head_s = nn.Linear(hidden, M)
        nn.init.zeros_(self.head_w.weight); nn.init.zeros_(self.head_w.bias)
        nn.init.zeros_(self.head_s.weight); nn.init.constant_(self.head_s.bias, 1.0)

    def forward(self, Sigma_hat, Delta_hat):
        h = self.enc(Sigma_hat)
        A = self.head_A(h)  # [N,1]
        w = torch.nn.functional.softplus(self.head_w(h)) + 1e-6  # [N,M] >=0
        s = torch.nn.functional.softplus(self.head_s(h)) + 1e-6  # [N,M] >=0
        d = Delta_hat - self.c.view(1,-1)                        # [N,M]
        bank = torch.tanh(s * d)                                 # [N,M]
        out = A + (w*bank).sum(dim=1, keepdim=True)
        # derivative wrt Delta (normalized) for slope-floor
        dtheta_dDhat = (w * (s * (1.0 - torch.tanh(s*d)**2))).sum(dim=1, keepdim=True)
        return out, dtheta_dDhat

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    # data & io
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--dt", type=float, default=None)
    # NN
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    # options A/B
    ap.add_argument("--learn-centers", action="store_true")
    ap.add_argument("--slope-floor", action="store_true")
    ap.add_argument("--slope-eps", type=float, default=0.05)
    ap.add_argument("--slope-lambda", type=float, default=1e-3)
    # option C: one-step
    ap.add_argument("--one-step", action="store_true")
    ap.add_argument("--one-step-warmup", type=int, default=50)
    ap.add_argument("--one-step-refit", type=int, default=20)
    ap.add_argument("--one-step-weight", type=float, default=0.05)
    ap.add_argument("--one-step-ramp", type=int, default=10, help="epochs to ramp one-step weight from 0 to target")
    ap.add_argument("--one-step-batch", type=int, default=2048, help="subsample size for one-step residual per epoch")
    # option D: sign consistency for kDelta
    ap.add_argument("--kdelta-hard-neg", action="store_true", help="enforce kDelta <= 0 in dynamics refit")
    ap.add_argument("--kdelta-sign-penalty", type=float, default=0.0, help="λ for soft penalty on positive kDelta in one-step loss")
    # dynamics fit options
    ap.add_argument("--sg-win", type=int, default=17)
    ap.add_argument("--sg-poly", type=int, default=3)
    ap.add_argument("--delay-grid", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--l2-dyn", type=float, default=5e-3)
    ap.add_argument("--kdelta-min", type=float, default=-2.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # ---------- load data ----------
    df = pd.read_csv(args.csv, comment="#")
    for col in ["p_sum[MPa]","p_diff[MPa]","theta[deg]"]:
        if col not in df.columns:
            raise RuntimeError(f"missing column {col}")
    ps  = df["p_sum[MPa]"].to_numpy(float)
    pdv = df["p_diff[MPa]"].to_numpy(float)
    th  = df["theta[deg]"].to_numpy(float)

    # dt
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

    # centers in normalized Delta
    M = int(args.M)
    c_grid = np.linspace(np.percentile(D_hat, 5), np.percentile(D_hat, 95), M).astype(np.float32)

    # train/val split（末尾を検証に）
    N = len(ps)
    val_n = max(int(N*args.val_ratio), 1)
    tr_idx = np.arange(0, N-val_n)
    va_idx = np.arange(N-val_n, N)

    Xtr_S = torch.tensor(S_hat[tr_idx], dtype=torch.float32).view(-1,1)
    Xtr_D = torch.tensor(D_hat[tr_idx], dtype=torch.float32).view(-1,1)
    ytr   = torch.tensor(th[tr_idx],    dtype=torch.float32).view(-1,1)

    Xva_S = torch.tensor(S_hat[va_idx], dtype=torch.float32).view(-1,1)
    Xva_D = torch.tensor(D_hat[va_idx], dtype=torch.float32).view(-1,1)
    yva   = torch.tensor(th[va_idx],    dtype=torch.float32).view(-1,1)

    ds_tr = TensorDataset(Xtr_S, Xtr_D, ytr)
    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    # model & opt
    model = MonoDeltaNN(M=M, hidden=int(args.hidden), c_grid=c_grid, learn_centers=args.learn_centers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    huber = torch.nn.HuberLoss(delta=3.0)

    # buffers for one-step/dynamics
    dyn_ready = False
    alpha = kS = kD = 0.0
    delay = 0

    # precompute smoothed series for one-step
    ps_s, _  = smooth_and_deriv(ps, dt, win=args.sg_win, poly=args.sg_poly)
    pd_s, _  = smooth_and_deriv(pdv, dt, win=args.sg_win, poly=args.sg_poly)
    th_s, dth= smooth_and_deriv(th, dt, win=args.sg_win, poly=args.sg_poly)

    def theta_stat_fn_numpy(ps_val, pd_val):
        S = torch.tensor([(ps_val-muS)/sdS], dtype=torch.float32).view(1,1)
        D = torch.tensor([(pd_val-muD)/sdD], dtype=torch.float32).view(1,1)
        with torch.no_grad():
            y, _ = model(S,D)
        return float(y.item())

    best_val = (1e18, None)
    patience, wait = 60, 0

    for ep in range(int(args.epochs)):
        model.train()
        # one-step weight ramp
        w1 = 0.0
        if args.one_step and dyn_ready and ep >= args.one_step_warmup:
            prog = max(0, ep - args.one_step_warmup + 1)
            w1 = float(args.one_step_weight) * min(1.0, prog / max(1, int(args.one_step_ramp)))

        for Sb,Db,yb in dl_tr:
            opt.zero_grad()
            yhat, dth_dDh = model(Sb, Db)  # [N,1]
            loss = huber(yhat, yb)

            # (B) slope floor
            if args.slope_floor:
                slope_shortage = torch.relu(args.slope_eps - dth_dDh)
                loss = loss + args.slope_lambda * slope_shortage.mean()

            # (C) one-step term
            if w1 > 0.0:
                # サブサンプリングして高速化
                start = delay + 1
                ks_full = np.arange(start, N)
                if len(ks_full) > int(args.one_step_batch):
                    sel = np.random.choice(ks_full, size=int(args.one_step_batch), replace=False)
                else:
                    sel = ks_full
                ku  = sel - delay
                kup = sel - delay - 1

                with torch.no_grad():
                    ths = np.array([theta_stat_fn_numpy(ps_s[i], pd_s[i]) for i in ku], float)
                    phi1= ths - th_s[sel]
                    dS  = (ps_s[ku] - ps_s[kup]) / dt
                    dD  = (pd_s[ku] - pd_s[kup]) / dt
                    y1  = dth[sel]
                    res = y1 - (alpha*phi1 + kS*dS + kD*dD)

                res = torch.tensor(res, dtype=torch.float32)
                loss = loss + w1 * (res**2).mean()

                # (D-soft) kΔ>0 にペナルティ
                if args.kdelta_sign_penalty > 0.0:
                    kdelta_pos = max(0.0, float(kD))
                    loss = loss + float(args.kdelta_sign_penalty) * kdelta_pos

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        # refit dynamics when needed (after warmup)
        if args.one_step and (ep+1) >= args.one_step_warmup:
            do_refit = (not dyn_ready) or (args.one_step_refit>0 and (ep+1-args.one_step_warmup)%args.one_step_refit==0)
            if do_refit:
                best_dyn, _ = fit_dynamics_with_delay(theta_stat_fn_numpy, ps, pdv, th,
                                                      dt, args.sg_win, args.sg_poly,
                                                      args.delay_grid, args.l2_dyn, args.kdelta_min,
                                                      hard_neg=args.kdelta-hard-neg if False else args.kdelta_hard_neg)
                # ↑ 上の1行のタイポ防止：正しい変数名に直下で再代入
                best_dyn, _ = fit_dynamics_with_delay(theta_stat_fn_numpy, ps, pdv, th,
                                                      dt, args.sg_win, args.sg_poly,
                                                      args.delay_grid, args.l2_dyn, args.kdelta_min,
                                                      hard_neg=args.kdelta_hard_neg)
                delay, alpha, kS, kD, rmse = best_dyn
                dyn_ready = True
                if w1 > 0:
                    print(f"[ep {ep:3d}] refit dynamics: d={delay}, alpha={alpha:.3f}, kS={kS:.3f}, kD={kD:.3f}, rolloutRMSE={rmse:.3f}")
                else:
                    print(f"[ep {ep:3d}] refit dynamics: d={delay}, alpha={alpha:.3f}, kS={kS:.3f}, kD={kD:.3f}, rolloutRMSE={rmse:.3f}")

        # validation
        model.eval()
        with torch.no_grad():
            yhat_va, _ = model(Xva_S, Xva_D)
            val_rmse = float(torch.sqrt(((yhat_va - yva)**2).mean()))
        if dyn_ready and w1>0:
            print(f"[ep {ep:3d}] val RMSE={val_rmse:.3f} deg, d={delay}, α={alpha:.2f}, kΣ={kS:.2f}, kΔ={kD:.2f}, oneStep={w1:.4f}")
        else:
            print(f"[ep {ep:3d}] val RMSE={val_rmse:.3f} deg")

        improved = val_rmse < best_val[0] - 1e-3
        if improved:
            best_val = (val_rmse, {k:v.cpu().clone() for k,v in model.state_dict().items()},
                        dict(delay=delay, alpha=alpha, kS=kS, kD=kD))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # load best
    if best_val[1] is not None:
        model.load_state_dict(best_val[1])
        if best_val[2]:
            delay = int(best_val[2]["delay"]); alpha=float(best_val[2]["alpha"])
            kS=float(best_val[2]["kS"]); kD=float(best_val[2]["kD"])

    # final dynamics refit (once more, on final network)
    def theta_stat_fn_numpy_final(ps_val, pd_val):
        S = torch.tensor([(ps_val-muS)/sdS], dtype=torch.float32).view(1,1)
        D = torch.tensor([(pd_val-muD)/sdD], dtype=torch.float32).view(1,1)
        with torch.no_grad():
            y, _ = model(S,D)
        return float(y.item())

    best_dyn, dyn_log = fit_dynamics_with_delay(
        theta_stat_fn_numpy_final, ps, pdv, th,
        dt, args.sg_win, args.sg_poly,
        args.delay_grid, args.l2_dyn, args.kdelta_min,
        hard_neg=args.kdelta_hard_neg
    )
    delay, alpha, kS, kD, rmse = best_dyn
    print(f"[final fit] d={delay}, alpha={alpha:.4f}, kΣ={kS:.4f}, kΔ={kD:.4f}, rolloutRMSE={rmse:.3f}")

    # report
    print("=== Final Neural Hammerstein (joint options) ===")
    print(f" delay={delay}, alpha={alpha:.4f} [1/s] (T={1.0/max(alpha,1e-9):.3f}s), kΣ={kS:.4f}, kΔ={kD:.4f}")

    # save
    pt_path = args.out_prefix + "_nh.pt"
    torch.save(model.state_dict(), pt_path)

    meta = dict(
        dt=dt, muS=float(muS), sdS=float(sdS), muD=float(muD), sdD=float(sdD),
        c_grid=(model.c.detach().cpu().numpy()),
        M=M, hidden=int(args.hidden),
        alpha=float(alpha), kSigma=float(kS), kDelta=float(kD), delay=int(delay),
        # options
        learn_centers=bool(args.learn_centers),
        slope_floor=bool(args.slope_floor), slope_eps=float(args.slope_eps), slope_lambda=float(args.slope_lambda),
        one_step=bool(args.one_step), one_step_warmup=int(args.one_step_warmup), one_step_refit=int(args.one_step_refit),
        one_step_weight=float(args.one_step_weight), one_step_ramp=int(args.one_step_ramp), one_step_batch=int(args.one_step_batch),
        # sign options
        kdelta_hard_neg=bool(args.kdelta_hard_neg), kdelta_sign_penalty=float(args.kdelta_sign_penalty),
        # dyn fit setup
        sg_win=int(args.sg_win), sg_poly=int(args.sg_poly),
        l2_dyn=float(args.l2_dyn), kdelta_min=float(args.kdelta_min),
    )
    npz_path = args.out_prefix + "_nh_meta.npz"
    # 推定 pmax/sigma_ref（運用で上書き可）
    pmax = float(np.nanmax(ps)*1.05)
    sigma_ref = float(np.median(ps))
    meta["pmax"] = pmax; meta["sigma_ref"] = sigma_ref
    np.savez(npz_path, **{k:(v if isinstance(v,np.ndarray) else np.array(v)) for k,v in meta.items()})

    print(f"[OK] saved: {pt_path}")
    print(f"[OK] saved: {npz_path}")
    print(json.dumps({k:(float(v) if isinstance(v,(int,float,np.floating)) else
                         (v.tolist() if isinstance(v,np.ndarray) else v))
                      for k,v in meta.items() if k!='c_grid'}, indent=2))

if __name__ == "__main__":
    main()
