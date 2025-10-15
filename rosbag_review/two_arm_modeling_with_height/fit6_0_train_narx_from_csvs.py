#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a NARX-like MLP for antagonistic McKibben system from CSV sessions.

Required CSV header columns:
  t[s], p_sum[MPa], p_diff[MPa], p1[MPa], p2[MPa], p1_cmd[MPa], p2_cmd[MPa],
  theta[rad], theta[deg], z[m], dz[m]

Notes:
- dz[m] is a relative displacement (NOT velocity). We use dz history as a feature.
- Static consistency pool uses quasi-static detection by |dθ/dt| and |d(dz)/dt| thresholds.

Example:
  python fit6_train_narx_from_csvs.py \
    --dyn_csvs out/dynamic_prbs_data.csv out/dynamic_multi_data.csv out/dynamic_cyrip_data.csv \
    --stat_csvs out/static1_data.csv out/static2_data.csv \
    --out_dir out_narx --lags 12 --delay 2 --epochs 250 \
    --ss_lambda 0.03 --ss_theta_eps 0.01 --ss_z_eps 0.002 \
    --rollout_teacher_beta 0.2
"""

import os, json, argparse, math, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------- Globals ----------------------
REQUIRED_COLS = [
    "t[s]","p_sum[MPa]","p_diff[MPa]","p1[MPa]","p2[MPa]","p1_cmd[MPa]","p2_cmd[MPa]",
    "theta[rad]","theta[deg]","z[m]","dz[m]"
]

def torch_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------- IO & Preproc ----------------------
def load_csv(path: str):
    df = pd.read_csv(path)
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"{path}: missing columns {miss}")
    df = df.sort_values("t[s]").dropna().reset_index(drop=True)

    t = df["t[s]"].to_numpy()
    dt = np.median(np.diff(t)) if len(t) > 2 else 0.01
    if not np.isfinite(dt) or dt <= 0: dt = 0.01

    # Derivatives of ps/pd (features)
    for col, dcol in [("p_sum[MPa]","dp_sum[MPa/s]"), ("p_diff[MPa]","dp_diff[MPa/s]")]:
        x = df[col].to_numpy()
        dx = np.zeros_like(x)
        if len(x) > 1:
            dx[1:] = (x[1:] - x[:-1]) / dt
            dx[0]  = dx[1]
        df[dcol] = dx
    return df, dt

def build_sequences_from_df(df: pd.DataFrame, lags: int, delay: int,
                            feat_cols, y_col="theta[rad]"):
    """Stack lagged features and 1-step-ahead target."""
    df = df.reset_index(drop=True)
    N = len(df)
    if N < (lags + delay + 2):
        return None, None
    y = df[y_col].to_numpy().astype(np.float32)
    X_list, idx_list = [], []
    for t_idx in range(lags + delay, N - 1):
        base = t_idx - delay
        fv = []
        ok = True
        for k in range(lags):
            i = base - k
            if i < 0: ok = False; break
            row = df.iloc[i][feat_cols].to_numpy().astype(np.float32)
            fv.append(row)
        if not ok: continue
        X_list.append(np.concatenate(fv, axis=0))
        idx_list.append(t_idx)
    if not X_list: return None, None
    X = np.vstack(X_list).astype(np.float32)
    y_out = y[np.array(idx_list) + 1]
    return X, y_out

def standardize_fit(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mu, std

def standardize_apply(X, mu, std):
    return (X - mu) / std

def make_feature_cols():
    # dz is relative displacement (m). Use it as a state-related feature.
    return ["theta[rad]",
            "p_sum[MPa]", "p_diff[MPa]",
            "dp_sum[MPa/s]", "dp_diff[MPa/s]",
            "dz[m]"]

def stack_sessions(csv_list, lags, delay, feat_cols):
    X_all, y_all, slices = [], [], []
    offset = 0
    for path in csv_list:
        df, _ = load_csv(path)
        X, y = build_sequences_from_df(df, lags, delay, feat_cols)
        if X is None: continue
        X_all.append(X); y_all.append(y)
        slices.append((path, offset, offset+len(y)))
        offset += len(y)
    if not X_all:
        raise ValueError("No usable samples from given CSVs.")
    return np.vstack(X_all).astype(np.float32), np.concatenate(y_all).astype(np.float32), slices

def theta_range_from_csvs(csvs):
    th_min, th_max = +np.inf, -np.inf
    for p in csvs:
        df, _ = load_csv(p)
        th = df["theta[rad]"].to_numpy()
        th_min = min(th_min, float(np.nanmin(th)))
        th_max = max(th_max, float(np.nanmax(th)))
    if not np.isfinite(th_min) or not np.isfinite(th_max):
        th_min, th_max = -math.pi, math.pi
    return th_min, th_max

# ---------------------- Model ----------------------
class MLP_NARX(nn.Module):
    def __init__(self, in_dim, hidden=[128,128], out_dim=1, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------------------- Rollout ----------------------
def rollout_predict(model, lags, delay, feat_cols, mu, std, df, steps=400, device="cpu",
                    clamp_theta=True, theta_minmax=None, teacher_beta=0.0):
    """
    Robust rollout evaluation (free-run 1-step recursion):
      - clamp predictions within observed theta range
      - optional leaky teacher forcing with factor 'teacher_beta' (0..0.3)
      - NaN/Inf & divergence guards
    """
    model.eval()
    df = df.reset_index(drop=True)
    N = len(df)
    start_idx = lags + delay
    end_idx   = N - 1
    if end_idx <= start_idx:
        return None

    # sanity on theta units
    th_abs_max = float(np.nanmax(np.abs(df["theta[rad]"].to_numpy())))
    if th_abs_max > 4*np.pi:
        print(f"[warn] theta[rad] abs max {th_abs_max:.3f} (>4π). Units/unwrap may be wrong.")

    if theta_minmax is None:
        tmin = float(np.nanpercentile(df["theta[rad]"].to_numpy(), 0.1))
        tmax = float(np.nanpercentile(df["theta[rad]"].to_numpy(), 99.9))
        theta_minmax = (min(tmin, -2*math.pi), max(tmax, 2*math.pi))
    th_lo, th_hi = theta_minmax

    t0 = max(start_idx, N - steps - 2)
    preds = []
    theta_true = df["theta[rad]"].to_numpy()
    theta_used = theta_true.copy()  # overwritten as we roll

    for base in range(t0, end_idx):
        fv = []
        for k in range(lags):
            i = base - k
            row = df.iloc[i][feat_cols].to_numpy().astype(np.float32)
            # overwrite theta entry with current "used" value
            for j, name in enumerate(feat_cols):
                if name == "theta[rad]":
                    row[j] = theta_used[i]
            fv.append(row)
        x = np.concatenate(fv, axis=0)[None, :]
        x_std = standardize_apply(x, mu, std)
        xt = torch.from_numpy(x_std).float().to(device)

        with torch.no_grad():
            y_hat = model(xt).cpu().numpy().reshape(-1)[0]

        if not np.isfinite(y_hat):
            print("[warn] NaN/Inf in prediction; abort rollout.")
            break
        if clamp_theta:
            y_hat = float(np.clip(y_hat, th_lo, th_hi))

        # Leaky teacher forcing (evaluation-time only)
        y_used = (1.0 - teacher_beta) * y_hat + teacher_beta * theta_true[base+1]

        theta_used[base+1] = y_used
        preds.append(y_hat)

        # hard divergence guard
        if abs(y_used) > 10.0 * max(abs(th_lo), abs(th_hi)) + 1.0:
            print(f"[warn] rollout diverged at step {base}; stopping early.")
            break

    y_true_seq = theta_true[t0+1 : t0+1+len(preds)]
    if len(preds) == 0:
        return None
    preds = np.asarray(preds)
    err   = preds - y_true_seq
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae":  float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "n":    int(len(preds))
    }

# ---------------------- Training ----------------------
def train(args):
    torch_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    feat_cols = make_feature_cols()

    # session split by files
    dyn_csvs = list(args.dyn_csvs)
    if len(dyn_csvs) == 1:
        train_dyn, val_dyn, test_dyn = dyn_csvs, dyn_csvs, dyn_csvs
    elif len(dyn_csvs) == 2:
        train_dyn, val_dyn, test_dyn = [dyn_csvs[0]], [dyn_csvs[1]], [dyn_csvs[1]]
    else:
        train_dyn, val_dyn, test_dyn = dyn_csvs[:-2], [dyn_csvs[-2]], [dyn_csvs[-1]]

    # datasets
    X_tr, y_tr, tr_slices = stack_sessions(train_dyn, args.lags, args.delay, feat_cols)
    X_va, y_va, va_slices = stack_sessions(val_dyn,   args.lags, args.delay, feat_cols)
    X_te, y_te, te_slices = stack_sessions(test_dyn,  args.lags, args.delay, feat_cols)

    # normalization on train only
    mu, std = standardize_fit(X_tr)
    X_tr_s = standardize_apply(X_tr, mu, std)
    X_va_s = standardize_apply(X_va, mu, std)
    X_te_s = standardize_apply(X_te, mu, std)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # tensors
    Xtr = torch.from_numpy(X_tr_s).float().to(device)
    ytr = torch.from_numpy(y_tr[:,None]).float().to(device)
    Xva = torch.from_numpy(X_va_s).float().to(device)
    yva = torch.from_numpy(y_va[:,None]).float().to(device)
    Xte = torch.from_numpy(X_te_s).float().to(device)
    yte = torch.from_numpy(y_te[:,None]).float().to(device)

    # model
    in_dim = Xtr.shape[1]
    model = MLP_NARX(in_dim, hidden=[args.hidden, args.hidden], out_dim=1, dropout=args.dropout).to(device)
    crit = nn.MSELoss()
    opt  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # static consistency pool (optional; quasi-static by slope thresholds)
    static_pool = []
    if args.stat_csvs and args.ss_lambda > 0.0:
        for spath in args.stat_csvs:
            sdf, _ = load_csv(spath)
            if len(sdf) >= 3:
                dt_est = np.median(np.diff(sdf["t[s]"].to_numpy()))
            else:
                dt_est = 0.01
            dtheta_dt = np.gradient(sdf["theta[rad]"].to_numpy(), dt_est)
            ddz_dt    = np.gradient(sdf["dz[m]"].to_numpy(),       dt_est)
            mask = (np.abs(dtheta_dt) < args.ss_theta_eps) & (np.abs(ddz_dt) < args.ss_z_eps)
            sdf_qs = sdf.loc[mask].copy().reset_index(drop=True)
            Xss, yss = build_sequences_from_df(sdf_qs, args.lags, args.delay, feat_cols)
            if Xss is not None and len(yss) > 0:
                Xss_s = standardize_apply(Xss, mu, std)
                static_pool.append((
                    torch.from_numpy(Xss_s).float().to(device),
                    torch.from_numpy(yss[:,None]).float().to(device)
                ))

    def evaluate(X, Y):
        model.eval()
        with torch.no_grad():
            Yh = model(X)
            mse = torch.mean((Yh - Y)**2).item()
            rmse = math.sqrt(mse)
            mae  = torch.mean(torch.abs(Yh - Y)).item()
            bias = torch.mean(Yh - Y).item()
        return rmse, mae, bias

    # training loop + early stopping
    best_val = float("inf"); best_state = None
    no_improve = 0; bs = args.batch_size
    for ep in range(1, args.epochs+1):
        model.train()
        N = Xtr.shape[0]
        idx = torch.randperm(N, device=device)
        total_loss = 0.0

        for i0 in range(0, N, bs):
            sel = idx[i0:i0+bs]
            xb, yb = Xtr[sel], ytr[sel]
            yhat = model(xb)
            loss = crit(yhat, yb)
            if static_pool and args.ss_lambda > 0.0:
                Xss_t, yss_t = random.choice(static_pool)
                take = min(bs, Xss_t.shape[0])
                ridx = torch.randint(0, Xss_t.shape[0], (take,), device=device)
                ss_loss = crit(model(Xss_t[ridx]), yss_t[ridx])
                loss = loss + args.ss_lambda * ss_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        rmse_tr, mae_tr, bias_tr = evaluate(Xtr, ytr)
        rmse_va, mae_va, bias_va = evaluate(Xva, yva)
        print(f"[{ep:03d}] loss={total_loss/max(1,N):.6f} | TR rmse={rmse_tr:.4e} mae={mae_tr:.4e} bias={bias_tr:.4e} "
              f"| VA rmse={rmse_va:.4e} mae={mae_va:.4e} bias={bias_va:.4e}")

        if rmse_va + 1e-6 < best_val:
            best_val = rmse_va
            best_state = {"model": model.state_dict()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {ep} (no improve {args.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # final metrics
    tr_metrics = evaluate(Xtr, ytr)
    va_metrics = evaluate(Xva, yva)
    te_metrics = evaluate(Xte, yte)

    # rollout on last test file (robust mode)
    ro_metrics = None
    try:
        last_te_path = te_slices[-1][0]
        df_te, _ = load_csv(last_te_path)
        theta_minmax = theta_range_from_csvs([p for p,_,_ in tr_slices])  # clamp to train range
        dev = next(model.parameters()).device
        ro_metrics = rollout_predict(
            model=model, lags=args.lags, delay=args.delay, feat_cols=feat_cols,
            mu=mu, std=std, df=df_te, steps=min(1200, len(df_te)),
            device=dev, clamp_theta=True, theta_minmax=theta_minmax,
            teacher_beta=args.rollout_teacher_beta
        )
    except Exception as e:
        print(f"[warn] rollout failed: {e}")

    # save artifacts
    meta = {
        "feature_names_single_slice": feat_cols,
        "lags": args.lags,
        "delay": args.delay,
        "mu": mu.tolist(),
        "std": std.tolist(),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_dyn_csvs": train_dyn,
        "val_dyn_csvs": val_dyn,
        "test_dyn_csvs": test_dyn,
        "static_csvs": list(args.stat_csvs) if args.stat_csvs else [],
        "ss_lambda": args.ss_lambda,
        "ss_theta_eps": args.ss_theta_eps,   # [rad/s]
        "ss_z_eps": args.ss_z_eps,           # [m/s] (slope of dz)
        "rollout_teacher_beta": args.rollout_teacher_beta
    }
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "narx_model.pt"))
    with open(os.path.join(args.out_dir, "narx_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    metrics = {
        "train": {"rmse": tr_metrics[0], "mae": tr_metrics[1], "bias": tr_metrics[2]},
        "val":   {"rmse": va_metrics[0], "mae": va_metrics[1], "bias": va_metrics[2]},
        "test":  {"rmse": te_metrics[0], "mae": te_metrics[1], "bias": te_metrics[2]},
        "rollout_test": ro_metrics
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n== Final Metrics ==")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved: {os.path.join(args.out_dir,'narx_model.pt')} and narx_meta.json")

# ---------------------- CLI ----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn_csvs", nargs="+", required=True, help="Dynamic session CSVs (D1..D3)")
    ap.add_argument("--stat_csvs", nargs="*", default=[], help="Static session CSVs (S1..S2); optional")
    ap.add_argument("--out_dir", type=str, default="out_narx")
    ap.add_argument("--lags", type=int, default=10, help="history length per signal")
    ap.add_argument("--delay", type=int, default=2, help="IO delay (steps)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    # static consistency (quasi-static by slopes)
    ap.add_argument("--ss_lambda", type=float, default=0.05, help="Weight of static consistency term (0 disables)")
    ap.add_argument("--ss_theta_eps", type=float, default=1.0*np.pi/180.0,
                    help="|dθ/dt| threshold [rad/s] for quasi-static selection")
    ap.add_argument("--ss_z_eps", type=float, default=2e-3,
                    help="|d(dz)/dt| threshold [m/s] for quasi-static selection")
    # rollout evaluation
    ap.add_argument("--rollout_teacher_beta", type=float, default=0.0,
                    help="Leaky teacher forcing for rollout (0..0.3 typical)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
