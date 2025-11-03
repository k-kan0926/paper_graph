#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI with NARX (multi-output) -- SAFE CLAMP VERSION
- clamp NARX outputs to training range (theta, dz)
- clamp reference to same range
- add heavy cost outside training range

python inverse7_1_narx_mppi.py \
    --meta out_narx/out_narx3_znarx/narx_meta.json \
    --model out_narx/out_narx3_znarx/narx_model.pt \
    --theta-target-deg 20 \
    --horizon 15 \
    --steps 40 \
    --K 256 \
    --lambda 1.0 \
    --pmax 0.7 \
    --plot \
    --context-csv out/dynamic_prbs_data.csv
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# -----------------------------------------------------------------------------
# 1. loading
# -----------------------------------------------------------------------------

def load_meta(meta_path: str):
    meta = json.loads(Path(meta_path).read_text())
    assert "lags" in meta
    assert "feature_names_single_slice" in meta
    assert "mu" in meta and "std" in meta
    return meta


def load_torch_state(pt_path: str):
    try:
        sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(pt_path, map_location="cpu")

    W1 = sd["net.0.weight"].cpu().numpy()
    b1 = sd["net.0.bias"].cpu().numpy()
    W2 = sd["net.2.weight"].cpu().numpy()
    b2 = sd["net.2.bias"].cpu().numpy()
    W3 = sd["net.4.weight"].cpu().numpy()
    b3 = sd["net.4.bias"].cpu().numpy()
    return (W1, b1, W2, b2, W3, b3)


# -----------------------------------------------------------------------------
# 2. NARX step (with SAFE CLAMP)
# -----------------------------------------------------------------------------

class NarxNumpy:
    def __init__(self, meta, weights, pmax: float, dt: float, act: str = "tanh"):
        self.meta = meta
        self.weights = weights
        self.pmax = float(pmax)
        self.dt = float(dt)
        self.L = int(meta["lags"])
        self.mu = np.asarray(meta["mu"], dtype=np.float32).reshape(1, -1)
        self.std = np.asarray(meta["std"], dtype=np.float32).reshape(1, -1)
        self.act = act

        # ここが追加: 学習時に書いた範囲をそのまま読む
        self.theta_lo, self.theta_hi = -math.pi, math.pi
        if "theta_train_minmax" in meta:
            self.theta_lo = float(meta["theta_train_minmax"][0])
            self.theta_hi = float(meta["theta_train_minmax"][1])

        self.dz_lo, self.dz_hi = -0.1, 0.1
        if "dz_train_minmax" in meta:
            self.dz_lo = float(meta["dz_train_minmax"][0])
            self.dz_hi = float(meta["dz_train_minmax"][1])

        # (optional) rate limit記録があれば使える
        self.ps_rate = float(meta.get("pressure_limits", {}).get("ps_rate_limit_MPa_s", 999.0))
        self.pd_rate = float(meta.get("pressure_limits", {}).get("pd_rate_limit_MPa_s", 999.0))

        self.idx_theta = lambda j: 6*j + 0
        self.idx_ps    = lambda j: 6*j + 1
        self.idx_pd    = lambda j: 6*j + 2
        self.idx_dps   = lambda j: 6*j + 3
        self.idx_dpd   = lambda j: 6*j + 4
        self.idx_dz    = lambda j: 6*j + 5

    def _act(self, x):
        if self.act == "tanh":
            return np.tanh(x)
        elif self.act == "relu":
            return np.maximum(x, 0.0)
        else:
            raise ValueError("act must be tanh or relu")

    def step(self, X: np.ndarray, U: np.ndarray):
        B = X.shape[0]
        L = self.L
        pmax = self.pmax
        dt = self.dt

        U = np.clip(U, 0.0, 1.0)
        a = U[:, 0:1]
        b = U[:, 1:2]
        ps = pmax * (a + b)
        pd = pmax * (a - b)

        ps_prev = X[:, self.idx_ps(0):self.idx_ps(0)+1]
        pd_prev = X[:, self.idx_pd(0):self.idx_pd(0)+1]

        dps = (ps - ps_prev) / dt
        dpd = (pd - pd_prev) / dt

        # --- ここでレート制限（metaにあれば） ---
        if self.ps_rate < 900.0:
            dps = np.clip(dps, -self.ps_rate, self.ps_rate)
            ps = ps_prev + dps * dt
        if self.pd_rate < 900.0:
            dpd = np.clip(dpd, -self.pd_rate, self.pd_rate)
            pd = pd_prev + dpd * dt

        # NARX入力
        z = X.copy()
        z_norm = (z - self.mu) / self.std

        W1, b1, W2, b2, W3, b3 = self.weights
        h1 = self._act(z_norm @ W1.T + b1[None, :])
        h2 = self._act(h1 @ W2.T + b2[None, :])
        y  = h2 @ W3.T + b3[None, :]   # (B,2)

        theta_next = y[:, 0:1]
        dz_next    = y[:, 1:2]

        # === 一番大事なところ ===
        theta_next = np.clip(theta_next, self.theta_lo, self.theta_hi)
        dz_next    = np.clip(dz_next,    self.dz_lo,    self.dz_hi)

        s_next = np.concatenate([theta_next, ps, pd, dps, dpd, dz_next], axis=1)

        if L > 1:
            X_next = np.concatenate([s_next, X[:, :6*(L-1)]], axis=1)
        else:
            X_next = s_next
        return X_next

    def unpack_front(self, X: np.ndarray):
        th = X[:, self.idx_theta(0)]
        ps = X[:, self.idx_ps(0)]
        pd = X[:, self.idx_pd(0)]
        dz = X[:, self.idx_dz(0)]
        return th, ps, pd, dz


# -----------------------------------------------------------------------------
# 3. utilities
# -----------------------------------------------------------------------------

def build_x0_from_context(meta, theta0=0.0, ps0=0.3, pd0=0.0):
    L = int(meta["lags"])
    s = np.array([theta0, ps0, pd0, 0.0, 0.0, 0.0], dtype=np.float32)
    x0 = np.tile(s, (L,))
    return x0


def infer_x0_from_csv(meta, csv_path: str):
    df = pd.read_csv(csv_path)
    L = int(meta["lags"])
    cols = ["theta[rad]", "p_sum[MPa]", "p_diff[MPa]", "dp_sum[MPa/s]", "dp_diff[MPa/s]", "dz[m]"]

    if "dp_sum[MPa/s]" not in df.columns or "dp_diff[MPa/s]" not in df.columns:
        t = df["t[s]"].to_numpy()
        dt = np.median(np.diff(t)) if len(t) > 1 else float(meta["dt_est"])
        ps = df["p_sum[MPa]"].to_numpy()
        pdv = df["p_diff[MPa]"].to_numpy()
        df["dp_sum[MPa/s]"] = np.gradient(ps, dt)
        df["dp_diff[MPa/s]"] = np.gradient(pdv, dt)

    tail = df.tail(L)
    if len(tail) < L:
        first = df.head(1).iloc[0]
        pads = [first for _ in range(L - len(tail))]
        tail = pd.concat([pd.DataFrame([r]) for r in pads] + [tail], ignore_index=True)
    rows = tail.iloc[::-1]
    x0 = rows[cols].to_numpy().astype(np.float32).reshape(-1,)
    return x0


def smooth_theta_ref(theta0, theta_target, T):
    t = np.linspace(0., 1., T+1)
    s = 3 * t**2 - 2 * t**3
    return theta0 + (theta_target - theta0) * s


# -----------------------------------------------------------------------------
# 4. MPPI core (with box penalty)
# -----------------------------------------------------------------------------

def mppi_control_step(model: NarxNumpy,
                      x0: np.ndarray,
                      theta_refs: np.ndarray,
                      U_nom: np.ndarray,
                      K: int,
                      lam: float,
                      sigma_u: float,
                      w_path: float,
                      w_rate: float,
                      w_z: float,
                      w_term: float,
                      w_box: float):
    T = U_nom.shape[0]
    Kb = K

    X = np.repeat(x0[None, :], Kb, axis=0)
    noise = sigma_u * np.random.randn(Kb, T, 2).astype(np.float32)
    J = np.zeros((Kb,), dtype=np.float32)

    theta_lo, theta_hi = model.theta_lo, model.theta_hi

    for t in range(T):
        U_t = U_nom[t][None, :] + noise[:, t, :]
        X_next = model.step(X, U_t)

        theta, ps, pd, dz = model.unpack_front(X_next)
        theta_ref_t = theta_refs[t+1]

        ps_prev = X[:, model.idx_ps(0)]
        pd_prev = X[:, model.idx_pd(0)]

        # 基本のコスト
        cost_t = (w_path * (theta - theta_ref_t)**2
                  + w_rate * (ps - ps_prev)**2
                  + w_rate * (pd - pd_prev)**2
                  + w_z    * (dz**2))

        # ----- BOXペナルティ -----
        over_hi = np.maximum(0.0, theta - theta_hi)
        over_lo = np.maximum(0.0, theta_lo - theta)
        cost_t += w_box * (over_hi**2 + over_lo**2)

        J += cost_t.astype(np.float32)
        X = X_next

    # 終端
    theta, _, _, _ = model.unpack_front(X)
    err_term = theta - theta_refs[-1]
    J += (w_term * err_term**2).astype(np.float32)

    # 終端でもBOX
    over_hi = np.maximum(0.0, theta - theta_hi)
    over_lo = np.maximum(0.0, theta_lo - theta)
    J += w_box * (over_hi**2 + over_lo**2)

    J_min = np.min(J)
    w = np.exp(-(J - J_min) / max(1e-6, lam))
    w_sum = np.sum(w) + 1e-8
    w_norm = w / w_sum

    U_new = np.zeros_like(U_nom)
    for t in range(T):
        U_t_samp = U_nom[t][None, :] + noise[:, t, :]
        U_t_samp = np.clip(U_t_samp, 0.0, 1.0)
        U_new[t] = np.sum(w_norm[:, None] * U_t_samp, axis=0)

    u0 = U_new[0].copy()
    info = {"J_min": float(J_min), "J_mean": float(np.mean(J))}
    return u0, U_new, info


# -----------------------------------------------------------------------------
# 5. main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-csv", default="")
    ap.add_argument("--theta-target-deg", type=float, default=20.0)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--sigma-u", type=float, default=0.15)
    ap.add_argument("--pmax", type=float, default=0.7)
    ap.add_argument("--dt-override", type=float, default=0.0)
    ap.add_argument("--act", choices=["tanh", "relu"], default="tanh")
    ap.add_argument("--w-term", type=float, default=8.0)
    ap.add_argument("--w-path", type=float, default=5.0)
    ap.add_argument("--w-rate", type=float, default=0.10)
    ap.add_argument("--w-z", type=float, default=0.10)
    # 新しく: 範囲外をとにかく嫌うコスト
    ap.add_argument("--w-box", type=float, default=200.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    meta = load_meta(args.meta)
    weights = load_torch_state(args.model)

    dt = float(meta["dt_est"]) if args.dt_override <= 0.0 else float(args.dt_override)
    model = NarxNumpy(meta, weights, pmax=args.pmax, dt=dt, act=args.act)

    # 初期状態
    if args.context_csv:
        x = infer_x0_from_csv(meta, args.context_csv)
        theta0 = float(x[0])
    else:
        theta0 = 0.0
        x = build_x0_from_context(meta, theta0=theta0, ps0=0.3, pd0=0.0)

    theta_target = math.radians(args.theta_target_deg)
    T = args.horizon

    # 初期U
    U_nom = np.zeros((T, 2), dtype=np.float32)
    a0 = (0.3 + 0.0) / (2 * args.pmax)
    b0 = (0.3 - 0.0) / (2 * args.pmax)
    U_nom[:] = np.clip([a0, b0], 0.0, 1.0)

    log_t = []
    log_theta = []
    log_theta_ref = []

    for k in range(args.steps):
        # いまのθからターゲットへS字
        theta_cur = float(x[0])
        theta_refs = smooth_theta_ref(theta_cur, theta_target, T)

        # ← ここで参照もクリップ
        theta_refs = np.clip(theta_refs,
                             model.theta_lo + 1e-3,
                             model.theta_hi - 1e-3)

        u0, U_nom, info = mppi_control_step(
            model=model,
            x0=x,
            theta_refs=theta_refs,
            U_nom=U_nom,
            K=args.K,
            lam=args.lam,
            sigma_u=args.sigma_u,
            w_path=args.w_path,
            w_rate=args.w_rate,
            w_z=args.w_z,
            w_term=args.w_term,
            w_box=args.w_box,
        )

        x_next = model.step(x[None, :], u0[None, :])[0]

        theta, _, _, _ = model.unpack_front(x_next[None, :])
        log_t.append(k * dt)
        log_theta.append(theta[0])
        log_theta_ref.append(theta_refs[1])

        x = x_next

        print(f"[step {k:03d}] theta={math.degrees(theta[0]):.2f} deg, "
              f"theta_ref={math.degrees(theta_refs[1]):.2f} deg, "
              f"u0={u0}, Jmin={info['J_min']:.3f}, Jmean={info['J_mean']:.3f}")

    if args.plot and _HAS_PLT:
        t = np.asarray(log_t)
        plt.figure()
        plt.title("theta tracking (clamped)")
        plt.plot(t, np.degrees(np.asarray(log_theta)), label="theta [deg]")
        plt.plot(t, np.degrees(np.asarray(log_theta_ref)), "--", label="theta_ref [deg]")
        plt.grid(True); plt.legend(); plt.xlabel("time [s]")
        plt.show()


if __name__ == "__main__":
    main()
