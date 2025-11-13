#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_narx_p1p2_cmd_final.py
最終版NARX訓練スクリプト（p1_cmd, p2_cmd ベース）

Features per lag:
  - theta[rad]
  - p1_cmd[MPa], p2_cmd[MPa]
  - dp1_cmd_dt[MPa/s], dp2_cmd_dt[MPa/s]
  - dz[m]  (optional)

Output:
  - theta_next[rad]

Usage:
  python train_narx_p1p2_cmd_final.py \
    --dyn_csvs processed_data/dyn_*.csv \
    --stat_csvs processed_data/static_*.csv \
    --out_dir models/narx_p1p2_final \
    --lags 24 --delay 17 --hidden 192 --epochs 400
"""

import argparse
import json
import math
import os
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ========== Config ==========
REQUIRED_COLS = [
    "t[s]",
    "p1_cmd[MPa]",
    "p2_cmd[MPa]",
    "dp1_cmd_dt[MPa/s]",
    "dp2_cmd_dt[MPa/s]",
    "theta[rad]",
    "dz[m]",
]


def torch_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========== Data Loading ==========
def load_csv(path: str, use_dz: bool = True):
    """CSVロード & 検証"""
    df = pd.read_csv(path)

    # 必須カラム
    required = ["t[s]", "p1_cmd[MPa]", "p2_cmd[MPa]", "theta[rad]"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing {missing}")

    # 微分項がなければ計算
    if "dp1_cmd_dt[MPa/s]" not in df.columns:
        print(f"[WARN] {path}: dp1_cmd_dt not found, computing...")
        t = df["t[s]"].values
        dt = np.median(np.diff(t)) if len(t) > 2 else 0.005
        df["dp1_cmd_dt[MPa/s]"] = np.gradient(df["p1_cmd[MPa]"].values, dt)
        df["dp2_cmd_dt[MPa/s]"] = np.gradient(df["p2_cmd[MPa]"].values, dt)

    # dz補完
    if "dz[m]" not in df.columns:
        df["dz[m]"] = 0.0

    df = df.sort_values("t[s]").dropna().reset_index(drop=True)

    t = df["t[s]"].values
    dt = np.median(np.diff(t)) if len(t) > 2 else 0.005

    return df, float(dt)


def make_feature_cols(use_dz: bool = True):
    """特徴量リスト（1ラグ分）"""
    base = [
        "theta[rad]",
        "p1_cmd[MPa]",
        "p2_cmd[MPa]",
        "dp1_cmd_dt[MPa/s]",
        "dp2_cmd_dt[MPa/s]",
    ]
    if use_dz:
        base.append("dz[m]")
    return base


def build_sequences_from_df(
    df: pd.DataFrame,
    lags: int,
    delay: int,
    feat_cols,
):
    """ラグ付き特徴量 → 1-step ahead theta予測"""
    df = df.reset_index(drop=True)
    N = len(df)
    if N < (lags + delay + 2):
        return None, None

    Y_full = df["theta[rad]"].values.astype(np.float32)
    X_list, idx_list = [], []

    for t_idx in range(lags + delay, N - 1):
        base = t_idx - delay
        fv = []
        ok = True
        for k in range(lags):
            i = base - k
            if i < 0:
                ok = False
                break
            row = df.iloc[i][feat_cols].values.astype(np.float32)
            fv.append(row)
        if not ok:
            continue
        X_list.append(np.concatenate(fv, axis=0))
        idx_list.append(t_idx)

    if not X_list:
        return None, None

    X = np.vstack(X_list).astype(np.float32)
    Y = Y_full[np.array(idx_list) + 1].reshape(-1, 1)  # (M, 1)
    return X, Y


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mu, std


def standardize_apply(X: np.ndarray, mu: np.ndarray, std: np.ndarray):
    return (X - mu) / std


def stack_sessions(csv_list, lags, delay, feat_cols):
    """複数CSVを統合"""
    X_all, Y_all, slices = [], [], []
    offset = 0
    for path in csv_list:
        df, _ = load_csv(path)
        X, Y = build_sequences_from_df(df, lags, delay, feat_cols)
        if X is None:
            continue
        X_all.append(X)
        Y_all.append(Y)
        slices.append((path, offset, offset + len(Y)))
        offset += len(Y)
    if not X_all:
        raise ValueError("No usable samples")
    return (
        np.vstack(X_all).astype(np.float32),
        np.vstack(Y_all).astype(np.float32),
        slices,
    )


def theta_range_from_csvs(csvs):
    """訓練データのtheta範囲"""
    th_min, th_max = +np.inf, -np.inf
    for p in csvs:
        df, _ = load_csv(p)
        th = df["theta[rad]"].values
        th_min = min(th_min, float(np.nanmin(th)))
        th_max = max(th_max, float(np.nanmax(th)))
    if not np.isfinite(th_min):
        th_min, th_max = -math.pi, math.pi
    return th_min, th_max


# ========== Model ==========
class MLP_NARX(nn.Module):
    def __init__(self, in_dim, hidden=None, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = [192, 192]

        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ========== Rollout ==========
def rollout_predict(
    model,
    lags,
    delay,
    feat_cols,
    mu,
    std,
    df,
    steps: int = 400,
    device: str = "cpu",
    clamp_theta: bool = True,
    theta_minmax=None,
    teacher_beta: float = 0.0,
):
    """Free-run rollout評価（teacher forcing optional）"""
    model.eval()
    df = df.reset_index(drop=True)
    N = len(df)
    start_idx = lags + delay
    end_idx = N - 1
    if end_idx <= start_idx:
        return None

    if theta_minmax is None:
        tmin = float(np.nanpercentile(df["theta[rad]"].values, 0.1))
        tmax = float(np.nanpercentile(df["theta[rad]"].values, 99.9))
        theta_minmax = (min(tmin, -2 * math.pi), max(tmax, 2 * math.pi))

    th_lo, th_hi = theta_minmax

    t0 = max(start_idx, N - steps - 2)
    preds_th = []
    theta_true = df["theta[rad]"].values
    theta_used = theta_true.copy()

    # Buffers for lagged features
    hist_theta = deque(
        [theta_true[i] for i in range(t0, t0 - lags, -1)],
        maxlen=lags,
    )

    for base in range(t0, end_idx):
        # Build feature vector with lagged values
        fv = []
        for k in range(lags):
            row = df.iloc[base - k][feat_cols].values.astype(np.float32).copy()
            # Override theta with predicted/used value
            theta_idx = feat_cols.index("theta[rad]")
            row[theta_idx] = list(hist_theta)[k]
            fv.append(row)

        x = np.concatenate(fv, axis=0)[None, :]
        x_std = standardize_apply(x, mu, std)
        xt = torch.from_numpy(x_std).float().to(device)

        with torch.no_grad():
            y_hat = model(xt).cpu().numpy().reshape(-1)

        if not np.all(np.isfinite(y_hat)):
            print("[WARN] NaN in prediction, abort rollout")
            break

        th_hat = float(y_hat[0])

        if clamp_theta:
            th_hat = float(np.clip(th_hat, th_lo, th_hi))

        # Leaky teacher forcing
        th_used_next = (1.0 - teacher_beta) * th_hat + teacher_beta * theta_true[base + 1]

        hist_theta.appendleft(th_used_next)
        preds_th.append(th_hat)

        # Divergence guard
        if abs(th_used_next) > 10.0 * max(abs(th_lo), abs(th_hi)):
            print(f"[WARN] Diverged at step {base}, stopping")
            break

    y_true_seq = theta_true[t0 + 1 : t0 + 1 + len(preds_th)]
    if len(preds_th) == 0:
        return None

    preds_th = np.array(preds_th)
    err_th = preds_th - y_true_seq

    return {
        "rmse": float(np.sqrt(np.mean(err_th**2))),
        "mae": float(np.mean(np.abs(err_th))),
        "bias": float(np.mean(err_th)),
        "n": int(len(preds_th)),
    }


# ========== Training ==========
def train(args):
    torch_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    feat_cols = make_feature_cols(use_dz=args.use_dz)

    # Dataset split
    dyn_csvs = list(args.dyn_csvs)
    if len(dyn_csvs) == 1:
        train_dyn, val_dyn, test_dyn = dyn_csvs, dyn_csvs, dyn_csvs
    elif len(dyn_csvs) == 2:
        train_dyn, val_dyn, test_dyn = [dyn_csvs[0]], [dyn_csvs[1]], [dyn_csvs[1]]
    else:
        train_dyn = dyn_csvs[:-2]
        val_dyn = [dyn_csvs[-2]]
        test_dyn = [dyn_csvs[-1]]

    print("\n[Dataset Split]")
    print(f"  Train: {len(train_dyn)} files")
    print(f"  Val:   {len(val_dyn)} files")
    print(f"  Test:  {len(test_dyn)} files\n")

    # Build datasets
    X_tr, Y_tr, tr_slices = stack_sessions(train_dyn, args.lags, args.delay, feat_cols)
    X_va, Y_va, va_slices = stack_sessions(val_dyn, args.lags, args.delay, feat_cols)
    X_te, Y_te, te_slices = stack_sessions(test_dyn, args.lags, args.delay, feat_cols)

    print("[Data Shapes]")
    print(f"  Train: X={X_tr.shape}, Y={Y_tr.shape}")
    print(f"  Val:   X={X_va.shape}, Y={Y_va.shape}")
    print(f"  Test:  X={X_te.shape}, Y={Y_te.shape}\n")

    # Normalize
    mu, std = standardize_fit(X_tr)
    X_tr_s = standardize_apply(X_tr, mu, std)
    X_va_s = standardize_apply(X_va, mu, std)
    X_te_s = standardize_apply(X_te, mu, std)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}\n")

    # Tensors
    Xtr = torch.from_numpy(X_tr_s).float().to(device)
    ytr = torch.from_numpy(Y_tr).float().to(device)
    Xva = torch.from_numpy(X_va_s).float().to(device)
    yva = torch.from_numpy(Y_va).float().to(device)
    Xte = torch.from_numpy(X_te_s).float().to(device)
    yte = torch.from_numpy(Y_te).float().to(device)

    # Model
    in_dim = Xtr.shape[1]
    model = MLP_NARX(
        in_dim,
        hidden=[args.hidden, args.hidden],
        out_dim=1,
        dropout=args.dropout,
    ).to(device)
    opt = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("[Model]")
    print(f"  Input dim:  {in_dim}")
    print(f"  Hidden:     {args.hidden}")
    print("  Output dim: 1 (theta)")
    print(f"  Params:     {sum(p.numel() for p in model.parameters())}\n")

    # Static consistency pool (quasi-static by slopes)
    static_pool = []
    if args.stat_csvs and args.ss_lambda > 0.0:
        for spath in args.stat_csvs:
            sdf, dt_est = load_csv(spath)
            if len(sdf) < 3:
                continue
            dtheta_dt = np.gradient(sdf["theta[rad]"].values, dt_est)
            mask = np.abs(dtheta_dt) < args.ss_theta_eps
            sdf_qs = sdf.loc[mask].copy().reset_index(drop=True)
            Xss, Yss = build_sequences_from_df(sdf_qs, args.lags, args.delay, feat_cols)
            if Xss is not None and len(Yss) > 0:
                Xss_s = standardize_apply(Xss, mu, std)
                static_pool.append(
                    (
                        torch.from_numpy(Xss_s).float().to(device),
                        torch.from_numpy(Yss).float().to(device),
                    )
                )
        print(f"[Static Pool] {len(static_pool)} quasi-static subsets loaded\n")

    # Loss function
    def mse_loss(yhat, ytrue):
        return torch.mean((yhat - ytrue) ** 2)

    def evaluate(X, Y):
        model.eval()
        with torch.no_grad():
            Yh = model(X)
            err = Yh - Y
            mse = torch.mean(err**2)
            mae = torch.mean(torch.abs(err))
            bias = torch.mean(err)
        return {
            "rmse": float(torch.sqrt(mse).item()),
            "mae": float(mae.item()),
            "bias": float(bias.item()),
        }

    # Training loop
    best_val = float("inf")
    best_state = None
    no_improve = 0
    bs = args.batch_size

    print(f"[Training] epochs={args.epochs}, batch_size={bs}, lr={args.lr}\n")

    for ep in range(1, args.epochs + 1):
        model.train()
        N = Xtr.shape[0]
        idx = torch.randperm(N, device=device)
        total_loss = 0.0

        for i0 in range(0, N, bs):
            sel = idx[i0 : i0 + bs]
            xb, yb = Xtr[sel], ytr[sel]
            yhat = model(xb)
            loss = mse_loss(yhat, yb)

            # Static consistency
            if static_pool and args.ss_lambda > 0.0:
                Xss_t, Yss_t = random.choice(static_pool)
                take = min(bs, Xss_t.shape[0])
                ridx = torch.randint(0, Xss_t.shape[0], (take,), device=device)
                ss_loss = mse_loss(model(Xss_t[ridx]), Yss_t[ridx])
                loss = loss + args.ss_lambda * ss_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        # Evaluate
        tr_metrics = evaluate(Xtr, ytr)
        va_metrics = evaluate(Xva, yva)

        if ep % 10 == 0 or ep == 1:
            print(
                f"[{ep:03d}] loss={total_loss / max(1, N):.6f} | "
                f"TR rmse={tr_metrics['rmse']:.5f} | VA rmse={va_metrics['rmse']:.5f}"
            )

        # Early stopping
        if va_metrics["rmse"] + 1e-6 < best_val:
            best_val = va_metrics["rmse"]
            best_state = {"model": model.state_dict()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {ep} (patience={args.patience})\n")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Final metrics
    tr_metrics = evaluate(Xtr, ytr)
    va_metrics = evaluate(Xva, yva)
    te_metrics = evaluate(Xte, yte)

    print("\n[Final Metrics]")
    print(f"  Train: rmse={tr_metrics['rmse']:.5f}, mae={tr_metrics['mae']:.5f}")
    print(f"  Val:   rmse={va_metrics['rmse']:.5f}, mae={va_metrics['mae']:.5f}")
    print(f"  Test:  rmse={te_metrics['rmse']:.5f}, mae={te_metrics['mae']:.5f}")

    # Rollout
    ro_metrics = None
    try:
        last_te_path = te_slices[-1][0]
        df_te, dt_est = load_csv(last_te_path)
        theta_minmax = theta_range_from_csvs([p for p, _, _ in tr_slices])
        ro_metrics = rollout_predict(
            model=model,
            lags=args.lags,
            delay=args.delay,
            feat_cols=feat_cols,
            mu=mu,
            std=std,
            df=df_te,
            steps=min(1200, len(df_te)),
            device=device,
            clamp_theta=True,
            theta_minmax=theta_minmax,
            teacher_beta=args.rollout_teacher_beta,
        )
        if ro_metrics:
            print(f"  Rollout: rmse={ro_metrics['rmse']:.5f}, n={ro_metrics['n']} steps")
    except Exception as e:
        print(f"[WARN] Rollout failed: {e}")
        dt_est = 0.005
        theta_minmax = (
            theta_range_from_csvs([p for p, _, _ in tr_slices])
            if tr_slices
            else (-math.pi, math.pi)
        )

    # Save artifacts
    meta = {
        "model_type": "NARX_MLP_p1p2_cmd",
        "feature_names_single_slice": feat_cols,
        "lags": args.lags,
        "delay": args.delay,
        "delay_measured_ms": args.delay_measured_ms,
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
        "ss_theta_eps": args.ss_theta_eps,
        "rollout_teacher_beta": args.rollout_teacher_beta,
        "theta_train_minmax": list(theta_minmax)
        if "theta_minmax" in locals()
        else [-math.pi, math.pi],
        "dt_est": float(dt_est),
        "use_dz": args.use_dz,
        "pressure_limits": {
            "p_max_each_side_MPa": args.p_max_each_side_MPa,
            "p1_rate_limit_MPa_s": args.p1_rate_limit_MPa_s,
            "p2_rate_limit_MPa_s": args.p2_rate_limit_MPa_s,
        },
    }

    torch.save(model.state_dict(), os.path.join(args.out_dir, "narx_model.pt"))
    with open(os.path.join(args.out_dir, "narx_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    metrics = {
        "train": tr_metrics,
        "val": va_metrics,
        "test": te_metrics,
        "rollout_test": ro_metrics,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[Saved]")
    print(f"  Model:   {os.path.join(args.out_dir, 'narx_model.pt')}")
    print(f"  Meta:    {os.path.join(args.out_dir, 'narx_meta.json')}")
    print(f"  Metrics: {os.path.join(args.out_dir, 'metrics.json')}")
    print(f"\n{'=' * 70}\n")


# ========== CLI ==========
def parse_args():
    ap = argparse.ArgumentParser(
        description="NARX training with p1_cmd, p2_cmd features",
    )

    # Data
    ap.add_argument(
        "--dyn_csvs",
        nargs="+",
        required=True,
        help="Dynamic session CSVs",
    )
    ap.add_argument(
        "--stat_csvs",
        nargs="*",
        default=[],
        help="Static session CSVs (optional)",
    )
    ap.add_argument("--out_dir", type=str, default="out_narx_p1p2")

    # Model architecture
    ap.add_argument(
        "--lags",
        type=int,
        default=24,
        help="History length per signal",
    )
    ap.add_argument(
        "--delay",
        type=int,
        default=17,
        help="I/O delay (steps)",
    )
    ap.add_argument(
        "--delay_measured_ms",
        type=float,
        default=None,
        help="Measured delay in ms (for metadata)",
    )
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument(
        "--use_dz",
        action="store_true",
        default=True,
        help="Include dz[m] in features",
    )

    # Training
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-6)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA available",
    )

    # Static consistency
    ap.add_argument(
        "--ss_lambda",
        type=float,
        default=0.12,
        help="Static consistency weight (0=disable)",
    )
    ap.add_argument(
        "--ss_theta_eps",
        type=float,
        default=0.008,
        help="|dθ/dt| threshold [rad/s] for quasi-static",
    )

    # Rollout evaluation
    ap.add_argument(
        "--rollout_teacher_beta",
        type=float,
        default=0.05,
        help="Leaky teacher forcing for rollout (0-0.3)",
    )

    # Physical constraints (for metadata)
    ap.add_argument("--p_max_each_side_MPa", type=float, default=0.70)
    ap.add_argument("--p1_rate_limit_MPa_s", type=float, default=3.5)
    ap.add_argument("--p2_rate_limit_MPa_s", type=float, default=3.5)

    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
