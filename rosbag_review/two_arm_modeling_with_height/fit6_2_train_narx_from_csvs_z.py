#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a multi-output NARX-like MLP for antagonistic McKibben system from CSV sessions.
Outputs: [theta_{t+1}[rad], dz_{t+1}[m]]

Required CSV columns:
  t[s], p_sum[MPa], p_diff[MPa], p1[MPa], p2[MPa], p1_cmd[MPa], p2_cmd[MPa],
  theta[rad], theta[deg], z[m], dz[m]

Notes:
- dz[m] is a relative displacement (NOT velocity). We use dz as a state-like feature.
- Quasi-static samples for the static consistency term are selected by slope thresholds:
  |dθ/dt| < ss_theta_eps [rad/s] and |d(dz)/dt| < ss_z_eps [m/s].

Example:
  python fit6_2_train_narx_from_csvs_z.py \
    --dyn_csvs out/dynamic_prbs_data.csv out/dynamic_multi_data.csv out/dynamic_cyrip_data.csv \
    --stat_csvs out/static1_data.csv out/static2_data.csv \
    --out_dir out_narx/out_narx3_znarx --lags 12 --delay 2 --epochs 250 \
    --ss_lambda 0.03 --ss_theta_eps 0.01 --ss_z_eps 0.002 \
    --rollout_teacher_beta 0.2 \
    --w_theta 1.0 --w_dz 1.0
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

OUTPUT_NAMES = ["theta[rad]","dz[m]"]  # predict next-step of these (order matters)

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
                            feat_cols, y_cols=("theta[rad]","dz[m]")):
    """Stack lagged features and 1-step-ahead multi-output target [theta, dz]."""
    df = df.reset_index(drop=True)
    N = len(df)
    if N < (lags + delay + 2):
        return None, None
    Yfull = df[list(y_cols)].to_numpy().astype(np.float32)
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
    Y = Yfull[np.array(idx_list) + 1]  # next-step targets for both outputs
    return X, Y  # shapes: (M, lags*len(feats)), (M, 2)

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
    X_all, Y_all, slices = [], [], []
    offset = 0
    for path in csv_list:
        df, _ = load_csv(path)
        X, Y = build_sequences_from_df(df, lags, delay, feat_cols, y_cols=OUTPUT_NAMES)
        if X is None: continue
        X_all.append(X); Y_all.append(Y)
        slices.append((path, offset, offset+len(Y)))
        offset += len(Y)
    if not X_all:
        raise ValueError("No usable samples from given CSVs.")
    return np.vstack(X_all).astype(np.float32), np.vstack(Y_all).astype(np.float32), slices

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

def dz_range_from_csvs(csvs):
    zmin, zmax = +np.inf, -np.inf
    for p in csvs:
        df, _ = load_csv(p)
        z = df["dz[m]"].to_numpy()
        zmin = min(zmin, float(np.nanmin(z)))
        zmax = max(zmax, float(np.nanmax(z)))
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        zmin, zmax = -0.1, 0.1
    return zmin, zmax

# ---------------------- Static dz surrogate fit: dz ≈ a0 + a1*ps + a2*ps^2 ----------------------
def fit_fz_quadratic_from_static(stat_csvs, ss_theta_eps, ss_z_eps):
    """
    Collect quasi-static samples from static CSVs and fit dz ≈ a0 + a1*ps + a2*ps^2.
    Returns dict with coefficients and diagnostics, or None if insufficient samples.
    """
    ps_list, dz_list = [], []
    for spath in stat_csvs:
        df, _ = load_csv(spath)
        if len(df) < 3:
            continue
        t = df["t[s]"].to_numpy()
        dt = np.median(np.diff(t)) if len(t) > 2 else 0.01
        dtheta_dt = np.gradient(df["theta[rad]"].to_numpy(), dt)
        ddz_dt    = np.gradient(df["dz[m]"].to_numpy(),       dt)
        mask = (np.abs(dtheta_dt) < ss_theta_eps) & (np.abs(ddz_dt) < ss_z_eps)
        qs = df.loc[mask]
        if len(qs) == 0:
            continue
        ps_list.append(qs["p_sum[MPa]"].to_numpy())
        dz_list.append(qs["dz[m]"].to_numpy())

    if not ps_list:
        return None

    ps = np.concatenate(ps_list).astype(np.float64)
    dz = np.concatenate(dz_list).astype(np.float64)

    # remove NaN/Inf
    ok = np.isfinite(ps) & np.isfinite(dz)
    ps, dz = ps[ok], dz[ok]
    if len(ps) < 20:
        return None

    # robust-ish trimming to suppress outliers (2–98 percentile)
    lo, hi = np.percentile(ps, [2.0, 98.0])
    keep = (ps >= lo) & (ps <= hi)
    ps_t = ps[keep]; dz_t = dz[keep]
    if len(ps_t) < 20:
        ps_t, dz_t = ps, dz

    # quadratic fit
    coeffs = np.polyfit(ps_t, dz_t, deg=2)  # returns [a2, a1, a0]
    a2, a1, a0 = map(float, coeffs)

    # diagnostics
    dz_hat = a0 + a1*ps_t + a2*(ps_t**2)
    ss_res = float(np.sum((dz_t - dz_hat)**2))
    ss_tot = float(np.sum((dz_t - np.mean(dz_t))**2))
    r2 = float(1.0 - ss_res/max(ss_tot, 1e-12))

    return {
        "a0": a0, "a1": a1, "a2": a2,
        "ps_min": float(np.min(ps)), "ps_max": float(np.max(ps)),
        "n": int(len(ps)),
        "r2": r2
    }

# ---------------------- Model ----------------------
class MLP_NARX(nn.Module):
    def __init__(self, in_dim, hidden=[128,128], out_dim=2, dropout=0.0):
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
                    clamp_theta=True, clamp_dz=True, theta_minmax=None, dz_minmax=None,
                    teacher_beta=0.0):
    """
    Robust rollout evaluation (free-run 1-step recursion) for multi-output [theta, dz]:
      - clamp predictions within observed ranges
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
    if dz_minmax is None:
        zmin = float(np.nanpercentile(df["dz[m]"].to_numpy(), 0.1))
        zmax = float(np.nanpercentile(df["dz[m]"].to_numpy(), 99.9))
        dz_minmax = (min(zmin, -1.0), max(zmax, 1.0))

    th_lo, th_hi = theta_minmax
    dz_lo, dz_hi = dz_minmax

    t0 = max(start_idx, N - steps - 2)
    preds_th, preds_dz = [], []
    theta_true = df["theta[rad]"].to_numpy()
    dz_true    = df["dz[m]"].to_numpy()
    theta_used = theta_true.copy()  # overwritten as we roll
    dz_used    = dz_true.copy()

    for base in range(t0, end_idx):
        fv = []
        for k in range(lags):
            i = base - k
            row = df.iloc[i][feat_cols].to_numpy().astype(np.float32)
            # overwrite theta & dz entries with current "used" values
            for j, name in enumerate(feat_cols):
                if name == "theta[rad]":
                    row[j] = theta_used[i]
                elif name == "dz[m]":
                    row[j] = dz_used[i]
            fv.append(row)
        x = np.concatenate(fv, axis=0)[None, :]
        x_std = standardize_apply(x, mu, std)
        xt = torch.from_numpy(x_std).float().to(device)

        with torch.no_grad():
            y_hat = model(xt).cpu().numpy().reshape(-1)  # [theta_hat, dz_hat]

        if not np.all(np.isfinite(y_hat)):
            print("[warn] NaN/Inf in prediction; abort rollout.")
            break
        th_hat, dz_hat = float(y_hat[0]), float(y_hat[1])

        if clamp_theta:
            th_hat = float(np.clip(th_hat, th_lo, th_hi))
        if clamp_dz:
            dz_hat = float(np.clip(dz_hat, dz_lo, dz_hi))

        # Leaky teacher forcing
        th_used = (1.0 - teacher_beta) * th_hat + teacher_beta * theta_true[base+1]
        dz_used_next = (1.0 - teacher_beta) * dz_hat + teacher_beta * dz_true[base+1]

        theta_used[base+1] = th_used
        dz_used[base+1]    = dz_used_next
        preds_th.append(th_hat)
        preds_dz.append(dz_hat)

        # hard divergence guard
        mag_bound = 10.0 * max(abs(th_lo), abs(th_hi)) + 1.0
        if abs(th_used) > mag_bound or not np.isfinite(th_used) or not np.isfinite(dz_used_next):
            print(f"[warn] rollout diverged at step {base}; stopping early.")
            break

    y_true_th_seq = theta_true[t0+1 : t0+1+len(preds_th)]
    y_true_dz_seq = dz_true[t0+1 : t0+1+len(preds_dz)]
    if len(preds_th) == 0:
        return None

    preds_th = np.asarray(preds_th)
    preds_dz = np.asarray(preds_dz)

    err_th = preds_th - y_true_th_seq
    err_dz = preds_dz - y_true_dz_seq
    return {
        "theta": {
            "rmse": float(np.sqrt(np.mean(err_th**2))),
            "mae":  float(np.mean(np.abs(err_th))),
            "bias": float(np.mean(err_th)),
            "n":    int(len(preds_th))
        },
        "dz": {
            "rmse": float(np.sqrt(np.mean(err_dz**2))),
            "mae":  float(np.mean(np.abs(err_dz))),
            "bias": float(np.mean(err_dz)),
            "n":    int(len(preds_dz))
        }
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

    # datasets (multi-output)
    X_tr, Y_tr, tr_slices = stack_sessions(train_dyn, args.lags, args.delay, feat_cols)
    X_va, Y_va, va_slices = stack_sessions(val_dyn,   args.lags, args.delay, feat_cols)
    X_te, Y_te, te_slices = stack_sessions(test_dyn,  args.lags, args.delay, feat_cols)

    # normalization on train only (inputs only)
    mu, std = standardize_fit(X_tr)
    X_tr_s = standardize_apply(X_tr, mu, std)
    X_va_s = standardize_apply(X_va, mu, std)
    X_te_s = standardize_apply(X_te, mu, std)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # tensors
    Xtr = torch.from_numpy(X_tr_s).float().to(device)
    ytr = torch.from_numpy(Y_tr).float().to(device)   # (N,2)
    Xva = torch.from_numpy(X_va_s).float().to(device)
    yva = torch.from_numpy(Y_va).float().to(device)
    Xte = torch.from_numpy(X_te_s).float().to(device)
    yte = torch.from_numpy(Y_te).float().to(device)

    # model
    in_dim = Xtr.shape[1]
    model = MLP_NARX(in_dim, hidden=[args.hidden, args.hidden], out_dim=2, dropout=args.dropout).to(device)
    opt  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # static consistency pool (optional; quasi-static by slope thresholds)
    static_pool = []
    if args.stat_csvs and args.ss_lambda > 0.0:
        for spath in args.stat_csvs:
            sdf, _ = load_csv(spath)
            if len(sdf) >= 3:
                dt_est_local = np.median(np.diff(sdf["t[s]"].to_numpy()))
            else:
                dt_est_local = 0.01
            dtheta_dt = np.gradient(sdf["theta[rad]"].to_numpy(), dt_est_local)
            ddz_dt    = np.gradient(sdf["dz[m]"].to_numpy(),       dt_est_local)
            mask = (np.abs(dtheta_dt) < args.ss_theta_eps) & (np.abs(ddz_dt) < args.ss_z_eps)
            sdf_qs = sdf.loc[mask].copy().reset_index(drop=True)
            Xss, Yss = build_sequences_from_df(sdf_qs, args.lags, args.delay, feat_cols, y_cols=OUTPUT_NAMES)
            if Xss is not None and len(Yss) > 0:
                Xss_s = standardize_apply(Xss, mu, std)
                static_pool.append((
                    torch.from_numpy(Xss_s).float().to(device),
                    torch.from_numpy(Yss).float().to(device)
                ))

    # weighted multi-output MSE
    w = torch.tensor([args.w_theta, args.w_dz], dtype=torch.float32, device=device)  # (2,)
    def weighted_mse(yhat, ytrue):
        mse_cols = torch.mean((yhat - ytrue)**2, dim=0)  # (2,)
        return torch.sum(w * mse_cols) / torch.sum(w)

    def evaluate(X, Y):
        model.eval()
        with torch.no_grad():
            Yh = model(X)
            err = Yh - Y
            mse_cols = torch.mean(err**2, dim=0)
            mae_cols = torch.mean(torch.abs(err), dim=0)
            bias_cols= torch.mean(err, dim=0)
            out = {
                "theta": {"rmse": float(torch.sqrt(mse_cols[0]).item()),
                          "mae":  float(mae_cols[0].item()),
                          "bias": float(bias_cols[0].item())},
                "dz":    {"rmse": float(torch.sqrt(mse_cols[1]).item()),
                          "mae":  float(mae_cols[1].item()),
                          "bias": float(bias_cols[1].item())}
            }
        return out

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
            loss = weighted_mse(yhat, yb)

            if static_pool and args.ss_lambda > 0.0:
                Xss_t, Yss_t = random.choice(static_pool)
                take = min(bs, Xss_t.shape[0])
                ridx = torch.randint(0, Xss_t.shape[0], (take,), device=device)
                ss_loss = weighted_mse(model(Xss_t[ridx]), Yss_t[ridx])
                loss = loss + args.ss_lambda * ss_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        tr_metrics = evaluate(Xtr, ytr)
        va_metrics = evaluate(Xva, yva)
        print(f"[{ep:03d}] loss={total_loss/max(1,N):.6f} | "
              f"TR θ rmse={tr_metrics['theta']['rmse']:.4e} dz rmse={tr_metrics['dz']['rmse']:.4e} | "
              f"VA θ rmse={va_metrics['theta']['rmse']:.4e} dz rmse={va_metrics['dz']['rmse']:.4e}")

        # early stopping uses weighted sum of val RMSEs
        val_score = (args.w_theta * va_metrics['theta']['rmse'] +
                     args.w_dz    * va_metrics['dz']['rmse']) / (args.w_theta + args.w_dz)
        if val_score + 1e-6 < best_val:
            best_val = val_score
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
        df_te, dt_est = load_csv(last_te_path)
        # clamp ranges from train CSVs
        theta_minmax = theta_range_from_csvs([p for p,_,_ in tr_slices])
        dz_minmax    = dz_range_from_csvs([p for p,_,_ in tr_slices])
        dev = next(model.parameters()).device
        ro_metrics = rollout_predict(
            model=model, lags=args.lags, delay=args.delay, feat_cols=feat_cols,
            mu=mu, std=std, df=df_te, steps=min(1200, len(df_te)),
            device=dev, clamp_theta=True, clamp_dz=True,
            theta_minmax=theta_minmax, dz_minmax=dz_minmax,
            teacher_beta=args.rollout_teacher_beta
        )
    except Exception as e:
        print(f"[warn] rollout failed: {e}")
        dt_est = float(np.median(np.diff(df_te['t[s]'].to_numpy()))) if 'df_te' in locals() else 0.01
        theta_minmax = theta_range_from_csvs([p for p,_,_ in tr_slices]) if 'tr_slices' in locals() else (-math.pi, math.pi)
        dz_minmax    = dz_range_from_csvs([p for p,_,_ in tr_slices]) if 'tr_slices' in locals() else (-0.1, 0.1)

    # ---- fit dz surrogate f_z(ps) on quasi-static samples from stat CSVs (optional)
    fz_meta = None
    if args.stat_csvs:
        try:
            fz_meta = fit_fz_quadratic_from_static(args.stat_csvs, args.ss_theta_eps, args.ss_z_eps)
        except Exception as e:
            print(f"[warn] f_z(ps) fit failed: {e}")
            fz_meta = None

    # save artifacts (include optional physical constraints for inverse solver)
    meta = {
        "feature_names_single_slice": make_feature_cols(),
        "output_names": OUTPUT_NAMES,         # ["theta[rad]", "dz[m]"]
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
        "ss_z_eps": args.ss_z_eps,           # [m/s] slope of dz
        "rollout_teacher_beta": args.rollout_teacher_beta,
        "loss_weights": {"theta": args.w_theta, "dz": args.w_dz},
        # helpful for inverse solver
        "theta_train_minmax": list(theta_minmax) if 'theta_minmax' in locals() else [-math.pi, math.pi],
        "dz_train_minmax": list(dz_minmax) if 'dz_minmax' in locals() else [-0.1, 0.1],
        "dt_est": float(dt_est),
        "pressure_limits": {
            "p_max_each_side_MPa": args.p_max_each_side_MPa,   # optional
            "ps_rate_limit_MPa_s": args.ps_rate_limit_MPa_s,   # optional
            "pd_rate_limit_MPa_s": args.pd_rate_limit_MPa_s    # optional
        },
        # ---- dz surrogate coefficients (if available)
        "fz_ps_quadratic": fz_meta  # dict or None
    }
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "narx_model.pt"))
    with open(os.path.join(args.out_dir, "narx_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    metrics = {
        "train": tr_metrics,   # dict with 'theta' and 'dz'
        "val":   va_metrics,
        "test":  te_metrics,
        "rollout_test": ro_metrics
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n== Final Metrics ==")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved: {os.path.join(args.out_dir,'narx_model.pt')} and narx_meta.json")
    if fz_meta is not None:
        print(f"dz surrogate f_z(ps): a0={fz_meta['a0']:.6g}, a1={fz_meta['a1']:.6g}, a2={fz_meta['a2']:.6g}, "
              f"R2={fz_meta['r2']:.3f}, n={fz_meta['n']}, range=[{fz_meta['ps_min']:.3f},{fz_meta['ps_max']:.3f}]")
    else:
        print("dz surrogate f_z(ps): not available (insufficient or invalid static data).")

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
    # optional physical constraints to store in meta (for inverse solver)
    ap.add_argument("--p_max_each_side_MPa", type=float, default=None,
                    help="Optional: per-side pressure upper bound [MPa] to record into meta")
    ap.add_argument("--ps_rate_limit_MPa_s", type=float, default=None,
                    help="Optional: |dp_sum/dt| limit [MPa/s] to record into meta")
    ap.add_argument("--pd_rate_limit_MPa_s", type=float, default=None,
                    help="Optional: |dp_diff/dt| limit [MPa/s] to record into meta")
    # multi-output loss weights
    ap.add_argument("--w_theta", type=float, default=1.0, help="loss weight for theta")
    ap.add_argument("--w_dz", type=float, default=1.0, help="loss weight for dz")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
