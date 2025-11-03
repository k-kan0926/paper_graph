#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute local input sensitivities (Jacobian) for multi-output NARX MLP:
  d[theta_hat]/d x_j and d[dz_hat]/d x_j at a chosen CSV time index.

- Loads model + meta (mu/std, lags, delay, feature order).
- Rebuilds the exact lagged feature vector at time t, standardizes, and
  computes per-input gradients via autograd.
- Reports both gradients w.r.t. standardized inputs and converted to original units.

Usage examples:
  python narx_local_sensitivities.py \
      --meta out_narx/out_narx3_znarx/narx_meta.json \
      --state out_narx/out_narx3_znarx/narx_model.pt \
      --csv  out/dynamic_multi_data.csv \
      --t-idx -3 \
      --out-dir out_narx/sens/sens

  # pick last usable t automatically
  python narx_local_sensitivities.py --meta ... --state ... --csv ... --auto-last

Outputs:
  - CSV files: sensitivities_theta.csv, sensitivities_dz.csv (sorted by |grad_orig|)
  - Optional lag-aggregated CSVs
"""

import os, json, argparse, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------- Model def (must match training) ----------------------
class MLP_NARX(nn.Module):
    def __init__(self, in_dim, hidden=[128,128], out_dim=2, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout and dropout > 0: layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------------------- Helpers ----------------------
REQ_COLS = ["t[s]","p_sum[MPa]","p_diff[MPa]","p1[MPa]","p2[MPa]","p1_cmd[MPa]","p2_cmd[MPa]",
            "theta[rad]","theta[deg]","z[m]","dz[m]"]

def load_csv(path: str):
    df = pd.read_csv(path)
    miss = [c for c in REQ_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"{path}: missing columns {miss}")
    df = df.sort_values("t[s]").dropna().reset_index(drop=True)
    t = df["t[s]"].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.01
    if not np.isfinite(dt) or dt <= 0: dt = 0.01
    # dp_sum / dp_diff (same as training)
    for col, dcol in [("p_sum[MPa]","dp_sum[MPa/s]"), ("p_diff[MPa]","dp_diff[MPa/s]")]:
        x = df[col].to_numpy()
        dx = np.zeros_like(x)
        if len(x) > 1:
            dx[1:] = (x[1:] - x[:-1]) / dt
            dx[0]  = dx[1]
        df[dcol] = dx
    return df, dt

def build_feature_slice(df_row, feat_cols):
    return df_row[feat_cols].to_numpy().astype(np.float32)

def build_lagged_input(df: pd.DataFrame, t_idx: int, lags: int, delay: int, feat_cols):
    """
    Compose x(t) = vec(phi(t-d), phi(t-d-1), ..., phi(t-d-lags+1))
    Returns shape (in_dim,), and also (k_list, r_list) index maps.
    """
    fv = []
    in_map = []  # list of (k, r, colname, abs_index)
    for k in range(lags):
        i = t_idx - delay - k
        if i < 0:
            raise IndexError(f"t_idx={t_idx} too small for lags={lags}, delay={delay}")
        row = df.iloc[i]
        slice_vals = build_feature_slice(row, feat_cols)
        fv.append(slice_vals)
        for r, name in enumerate(feat_cols):
            in_map.append((k, r, name, len(in_map)))
    x = np.concatenate(fv, axis=0).astype(np.float32)
    return x, in_map

def standardize_apply(x, mu, std):
    return (x - mu) / std

def ensure_last_usable_index(df_len, lags, delay):
    """
    We need t in [lags+delay, N-2] because the original training used y(t+1).
    For local sensitivity we only need x(t), so safe choice is N-2.
    """
    t_idx = df_len - 2
    lo = lags + delay
    if t_idx < lo:
        raise ValueError(f"Series too short. Need at least lags+delay+2 = {lo+2} rows.")
    return t_idx

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="path to narx_meta.json")
    ap.add_argument("--state", required=True, help="path to narx_model.pt (state_dict)")
    ap.add_argument("--csv",   required=True, help="CSV to sample from")
    ap.add_argument("--t-idx", type=int, default=None, help="time index t for x(t) (can be negative)")
    ap.add_argument("--auto-last", action="store_true", help="pick last usable t automatically")
    ap.add_argument("--out-dir", type=str, default="out_sens")
    ap.add_argument("--aggregate-by-lag", action="store_true", help="also write lag-aggregated CSVs")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- load meta & model
    with open(args.meta, "r") as f:
        meta = json.load(f)
    feat_cols = meta["feature_names_single_slice"]
    lags      = int(meta["lags"])
    delay     = int(meta["delay"])
    mu        = np.asarray(meta["mu"], dtype=np.float32)
    std       = np.asarray(meta["std"], dtype=np.float32)
    hidden    = int(meta.get("hidden", 128))
    dropout   = float(meta.get("dropout", 0.0))
    in_dim    = len(mu)
    assert in_dim == 6*lags, f"in_dim={in_dim} but 6*lags={6*lags}"

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")

    model = MLP_NARX(in_dim=in_dim, hidden=[hidden, hidden], out_dim=2, dropout=dropout).to(device)

    # ↓ ここを安全ロードに変更
    try:
        state = torch.load(args.state, map_location=device, weights_only=True)  # PyTorch >= 2.4
    except TypeError:
        state = torch.load(args.state, map_location=device)  # 互換フォールバック
    model.load_state_dict(state)
    model.eval()

    # --- load CSV and build x(t)
    df, _ = load_csv(args.csv)
    N = len(df)

    # ↓ ここを auto_last に修正
    if args.auto_last or args.t_idx is None:
        t_idx = ensure_last_usable_index(N, lags, delay)
    else:
        t_idx = args.t_idx if args.t_idx >= 0 else (N + args.t_idx)
        lo = lags + delay
        if t_idx < lo or t_idx >= N-1:
            raise IndexError(f"t_idx must be in [{lo}, {N-2}] (you gave {t_idx})")


    x_raw, in_map = build_lagged_input(df, t_idx, lags, delay, feat_cols)
    assert x_raw.shape[0] == in_dim

    # --- standardize
    x_std = standardize_apply(x_raw, mu, std)
    x_t = torch.from_numpy(x_std[None, :]).float().to(device)
    x_t.requires_grad_(True)

    # --- forward
    y_hat = model(x_t)  # shape (1,2)
    # outputs: [0]=theta_hat, [1]=dz_hat
    # compute gradients wrt standardized inputs
    sens_std = []
    for out_i in [0, 1]:
        grad = torch.autograd.grad(y_hat[0, out_i], x_t, retain_graph=True)[0].detach().cpu().numpy().reshape(-1)
        sens_std.append(grad)
    sens_std = np.stack(sens_std, axis=0)  # (2, in_dim)

    # convert to original units: dy/dx = (dy/dx_std) * (1/std)
    inv_std = 1.0 / std
    sens_orig = sens_std * inv_std[None, :]

    # --- build DataFrames
    rows = []
    for j, (k, r, name, _) in enumerate(in_map):
        rows.append({
            "abs_col": j,
            "lag_k": k,
            "feat_name": name,
            "x_value_raw": float(x_raw[j]),
            "x_value_std": float(x_std[j]),
            "dtheta_d_xstd": float(sens_std[0, j]),
            "ddz_d_xstd":    float(sens_std[1, j]),
            "dtheta_d_xorig": float(sens_orig[0, j]),
            "ddz_d_xorig":    float(sens_orig[1, j]),
            "abs_dtheta_d_xorig": float(abs(sens_orig[0, j])),
            "abs_ddz_d_xorig":    float(abs(sens_orig[1, j])),
        })
    df_all = pd.DataFrame(rows)

    # sort and save per-output
    df_theta = df_all.sort_values("abs_dtheta_d_xorig", ascending=False)
    df_dz    = df_all.sort_values("abs_ddz_d_xorig",    ascending=False)

    f_theta = os.path.join(args.out_dir, "sensitivities_theta.csv")
    f_dz    = os.path.join(args.out_dir, "sensitivities_dz.csv")
    df_theta.to_csv(f_theta, index=False)
    df_dz.to_csv(f_dz, index=False)

    # optional: aggregate by lag (sum of absolute grads over the 6 features of each lag)
    if args.aggregate_by_lag:
        agg_theta = df_all.groupby("lag_k")["abs_dtheta_d_xorig"].sum().reset_index().sort_values("abs_dtheta_d_xorig", ascending=False)
        agg_dz    = df_all.groupby("lag_k")["abs_ddz_d_xorig"].sum().reset_index().sort_values("abs_ddz_d_xorig",    ascending=False)
        agg_theta.to_csv(os.path.join(args.out_dir, "sens_theta_by_lag.csv"), index=False)
        agg_dz.to_csv(   os.path.join(args.out_dir, "sens_dz_by_lag.csv"),    index=False)

    # console summary
    def topn(df, col, n=12):
        return df.head(n)[["lag_k","feat_name","abs_col",col,"x_value_raw"]]

    print(f"\n[info] t-idx = {t_idx}  (time t = df['t[s]'][t_idx] = {df['t[s]'].iloc[t_idx]:.6f} s)")
    print(f"[info] wrote: {f_theta}")
    print(topn(df_theta, "abs_dtheta_d_xorig").to_string(index=False))
    print(f"\n[info] wrote: {f_dz}")
    print(topn(df_dz, "abs_ddz_d_xorig").to_string(index=False))

if __name__ == "__main__":
    main()
