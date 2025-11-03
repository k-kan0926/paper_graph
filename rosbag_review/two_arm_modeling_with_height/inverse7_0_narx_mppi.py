#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI with NARX (multi-output) for antagonistic McKibben system.

- uses the SAME meta.json & .pt that the training script saved
- uses the SAME 6*L state layout and the SAME a,b -> (ps,pd) mapping
- objective is similar to your acados script:
    * theta tracking
    * dz suppression
    * pressure rate smoothness

Run (example):

python inverse7_narx_mppi.py \
    --meta out_narx/out_narx3_znarx/narx_meta.json \
    --model out_narx/out_narx3_znarx/narx_model.pt \
    --theta-target-deg 20 \
    --horizon 15 \
    --steps 40 \
    --K 256 \
    --lambda 1.0 \
    --pmax 0.7 \
    --plot \
    --context-csv out/dynamic_prbs_data.csv  # optional
      

If you have a context CSV (recent trajectory) and want to initialize the 6L buffer from it:

  python inverse7_narx_mppi.py \
      --meta .../narx_meta.json \
      --model .../narx_model.pt \
      --context-csv out/dynamic_prbs_data.csv \
      --theta-target-deg 20

This is a *test* script – no ROS publishing here. Integrate the "chosen_u" section
with your ROS node where you publish /mpa_cmd later.
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
    # same style as your acados script
    try:
        sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(pt_path, map_location="cpu")
    # extract linear layers
    W1 = sd["net.0.weight"].cpu().numpy()
    b1 = sd["net.0.bias"].cpu().numpy()
    W2 = sd["net.2.weight"].cpu().numpy()
    b2 = sd["net.2.bias"].cpu().numpy()
    W3 = sd["net.4.weight"].cpu().numpy()
    b3 = sd["net.4.bias"].cpu().numpy()
    return (W1, b1, W2, b2, W3, b3)


# -----------------------------------------------------------------------------
# 2. NARX step (vectorized for MPPI)
# -----------------------------------------------------------------------------

class NarxNumpy:
    """
    Pure-numpy version of your trained 3-layer MLP NARX.
    Input:  X : (B, 6L)
            U : (B, 2)  (a,b) in [0,1]
    Output: X_next : (B, 6L)
    """
    def __init__(self, meta, weights, pmax: float, dt: float, act: str = "tanh"):
        self.meta = meta
        self.weights = weights
        self.pmax = float(pmax)
        self.dt = float(dt)
        self.L = int(meta["lags"])
        self.mu = np.asarray(meta["mu"], dtype=np.float32).reshape(1, -1)
        self.std = np.asarray(meta["std"], dtype=np.float32).reshape(1, -1)
        self.act = act

        # index helpers (same as acados script)
        # slice j has offset 6*j
        # 0:theta,1:ps,2:pd,3:dps,4:dpd,5:dz
        self.idx_theta = lambda j: 6*j + 0
        self.idx_ps    = lambda j: 6*j + 1
        self.idx_pd    = lambda j: 6*j + 2
        self.idx_dps   = lambda j: 6*j + 3
        self.idx_dpd   = lambda j: 6*j + 4
        self.idx_dz    = lambda j: 6*j + 5

    # activation
    def _act(self, x):
        if self.act == "tanh":
            return np.tanh(x)
        elif self.act == "relu":
            return np.maximum(x, 0.0)
        else:
            raise ValueError("act must be tanh or relu")

    def step(self, X: np.ndarray, U: np.ndarray):
        """
        X: (B, 6L) current buffer
        U: (B, 2)  actions in [0,1]
        returns X_next: (B, 6L)
        """
        B = X.shape[0]
        L = self.L
        pmax = self.pmax
        dt = self.dt

        # 1) a,b -> ps,pd
        U = np.clip(U, 0.0, 1.0)
        a = U[:, 0:1]
        b = U[:, 1:2]
        ps = pmax * (a + b)      # (B,1)
        pd = pmax * (a - b)      # (B,1)

        # current ps/pd (front slice)
        ps_prev = X[:, self.idx_ps(0):self.idx_ps(0)+1]
        pd_prev = X[:, self.idx_pd(0):self.idx_pd(0)+1]

        dps = (ps - ps_prev) / dt
        dpd = (pd - pd_prev) / dt

        # 2) MLP input = concat 6L
        z = X.copy()  # (B, 6L)
        # normalize
        z_norm = (z - self.mu) / self.std

        # 3) run MLP for all batch
        W1, b1, W2, b2, W3, b3 = self.weights
        h1 = self._act(z_norm @ W1.T + b1[None, :])
        h2 = self._act(h1 @ W2.T + b2[None, :])
        y  = h2 @ W3.T + b3[None, :]   # (B,2): [theta_next, dz_next]
        theta_next = y[:, 0:1]
        dz_next = y[:, 1:2]

        # 4) build new front slice
        s_next = np.concatenate([theta_next, ps, pd, dps, dpd, dz_next], axis=1)  # (B,6)

        # 5) shift buffer (new → old)
        if L > 1:
            X_next = np.concatenate([s_next, X[:, :6*(L-1)]], axis=1)
        else:
            X_next = s_next
        return X_next

    def unpack_front(self, X: np.ndarray):
        """return (theta, ps, pd, dz) for front slice (B,)"""
        th = X[:, self.idx_theta(0)]
        ps = X[:, self.idx_ps(0)]
        pd = X[:, self.idx_pd(0)]
        dz = X[:, self.idx_dz(0)]
        return th, ps, pd, dz


# -----------------------------------------------------------------------------
# 3. utilities (init state)
# -----------------------------------------------------------------------------

def build_x0_from_context(meta, theta0=0.0, ps0=0.3, pd0=0.0):
    L = int(meta["lags"])
    # dps, dpd, dz = 0
    s = np.array([theta0, ps0, pd0, 0.0, 0.0, 0.0], dtype=np.float32)
    x0 = np.tile(s, (L,))  # (6L,)
    return x0


def infer_x0_from_csv(meta, csv_path: str):
    df = pd.read_csv(csv_path)
    L = int(meta["lags"])
    cols = ["theta[rad]", "p_sum[MPa]", "p_diff[MPa]", "dp_sum[MPa/s]", "dp_diff[MPa/s]", "dz[m]"]
    # ensure dp columns
    if "dp_sum[MPa/s]" not in df.columns or "dp_diff[MPa/s]" not in df.columns:
        # try to estimate
        t = df["t[s]"].to_numpy()
        dt = np.median(np.diff(t)) if len(t) > 1 else float(meta["dt_est"])
        ps = df["p_sum[MPa]"].to_numpy()
        pdv = df["p_diff[MPa]"].to_numpy()
        df["dp_sum[MPa/s]"] = np.gradient(ps, dt)
        df["dp_diff[MPa/s]"] = np.gradient(pdv, dt)

    tail = df.tail(L)
    if len(tail) < L:
        # pad with first row
        first = df.head(1).iloc[0]
        pads = [first for _ in range(L - len(tail))]
        tail = pd.concat([pd.DataFrame([r]) for r in pads] + [tail], ignore_index=True)
    # newest must be first slice
    rows = tail.iloc[::-1]  # reverse
    x0 = rows[cols].to_numpy().astype(np.float32).reshape(-1,)
    return x0


def smooth_theta_ref(theta0, theta_target, T):
    # same S-curve as acados version
    t = np.linspace(0., 1., T+1)
    s = 3 * t**2 - 2 * t**3
    return theta0 + (theta_target - theta0) * s


# -----------------------------------------------------------------------------
# 4. MPPI core
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
                      w_term: float):
    """
    One MPPI iteration.
    x0        : (6L,)
    theta_refs: (T+1,) desired theta sequence
    U_nom     : (T,2) nominal control sequence
    K         : #samples
    return: (u0, U_new, info)
    """
    T = U_nom.shape[0]
    L = model.L
    B = K

    # expand x0 to K
    X = np.repeat(x0[None, :], K, axis=0)  # (K, 6L)
    # sample noises for all horizon
    noise = sigma_u * np.random.randn(K, T, 2).astype(np.float32)
    # clip later

    J = np.zeros((K,), dtype=np.float32)

    # rollout every sample
    for t in range(T):
        # control for all samples at time t
        U_t = U_nom[t][None, :] + noise[:, t, :]
        # step
        X_next = model.step(X, U_t)
        # cost
        theta, ps, pd, dz = model.unpack_front(X_next)
        theta_ref_t = theta_refs[t+1]  # +1 because X_next is k+1
        err_th = theta - theta_ref_t
        # prev ps/pd are in X (before stepping), front slice 0
        ps_prev = X[:, model.idx_ps(0)]
        pd_prev = X[:, model.idx_pd(0)]
        cost_t = (w_path * err_th**2
                  + w_rate * (ps - ps_prev)**2
                  + w_rate * (pd - pd_prev)**2
                  + w_z * (dz**2))
        J += cost_t.astype(np.float32)
        X = X_next

    # terminal cost
    theta, _, _, _ = model.unpack_front(X)
    err_term = theta - theta_refs[-1]
    J += (w_term * err_term**2).astype(np.float32)

    # importance weights
    J_min = np.min(J)
    w = np.exp(-(J - J_min) / max(1e-6, lam))
    w_sum = np.sum(w) + 1e-8
    w_norm = w / w_sum

    # update U
    U_new = np.zeros_like(U_nom)
    for t in range(T):
        U_t_samp = U_nom[t][None, :] + noise[:, t, :]
        U_t_samp = np.clip(U_t_samp, 0.0, 1.0)
        U_new[t] = np.sum(w_norm[:, None] * U_t_samp, axis=0)

    # control to apply now
    u0 = U_new[0].copy()
    info = {
        "J_min": float(J_min),
        "J_mean": float(np.mean(J)),
    }
    return u0, U_new, info


# -----------------------------------------------------------------------------
# 5. main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-csv", default="", help="optional csv to init 6L buffer")
    ap.add_argument("--theta-target-deg", type=float, default=20.0)
    ap.add_argument("--horizon", type=int, default=15, help="MPPI horizon (T)")
    ap.add_argument("--steps", type=int, default=40, help="closed-loop simulation length")
    ap.add_argument("--K", type=int, default=256, help="# of MPPI samples")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--sigma-u", type=float, default=0.15, help="noise std for u")
    ap.add_argument("--pmax", type=float, default=0.7)
    ap.add_argument("--dt-override", type=float, default=0.0)
    ap.add_argument("--act", choices=["tanh", "relu"], default="tanh")
    # cost weights (make them similar to acados)
    ap.add_argument("--w-term", type=float, default=8.0)
    ap.add_argument("--w-path", type=float, default=5.0)
    ap.add_argument("--w-rate", type=float, default=0.10)
    ap.add_argument("--w-z", type=float, default=0.10)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    meta = load_meta(args.meta)
    weights = load_torch_state(args.model)

    dt = float(meta["dt_est"]) if args.dt_override <= 0.0 else float(args.dt_override)
    model = NarxNumpy(meta, weights, pmax=args.pmax, dt=dt, act=args.act)

    # initial state
    if args.context_csv:
        x = infer_x0_from_csv(meta, args.context_csv)
        theta0 = float(x[0])
    else:
        theta0 = 0.0
        x = build_x0_from_context(meta, theta0=theta0, ps0=0.3, pd0=0.0)

    theta_target = math.radians(args.theta_target_deg)
    T = args.horizon

    # initial nominal control (do nothing-ish)
    U_nom = np.zeros((T, 2), dtype=np.float32)
    # e.g. keep ps=0.3, pd=0.0 as initial guess
    a0 = (0.3 + 0.0) / (2 * args.pmax)  # ≈ ps/(2*pmax)
    b0 = (0.3 - 0.0) / (2 * args.pmax)
    U_nom[:] = np.clip([a0, b0], 0.0, 1.0)

    # logs
    log_t = []
    log_theta = []
    log_theta_ref = []
    log_ps = []
    log_pd = []
    log_u = []

    for k in range(args.steps):
        # build reference for this step (always from current theta)
        theta_cur = float(x[0])
        theta_refs = smooth_theta_ref(theta_cur, theta_target, T)

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
        )

        # apply chosen u0 to "plant" (here, the same NARX)
        x_next = model.step(x[None, :], u0[None, :])[0]

        # log
        th, ps, pd, dz = model.unpack_front(x_next[None, :])
        log_t.append(k * dt)
        log_theta.append(th[0])
        log_theta_ref.append(theta_targets := theta_refs[1])
        log_ps.append(ps[0])
        log_pd.append(pd[0])
        log_u.append(u0)

        # for next loop
        x = x_next

        print(f"[step {k:03d}] theta={math.degrees(th[0]):.2f} deg, "
              f"theta_ref={math.degrees(theta_targets):.2f} deg, "
              f"u0={u0}, Jmin={info['J_min']:.3f}, Jmean={info['J_mean']:.3f}")

    # show result
    if args.plot and _HAS_PLT:
        t = np.asarray(log_t)
        plt.figure()
        plt.title("theta tracking (MPPI+NARX)")
        plt.plot(t, np.degrees(np.asarray(log_theta)), label="theta [deg]")
        plt.plot(t, np.degrees(np.asarray(log_theta_ref)), "--", label="theta_ref [deg]")
        plt.grid(True); plt.legend()
        plt.xlabel("time [s]")

        plt.figure()
        plt.title("ps/pd")
        plt.plot(t, np.asarray(log_ps), label="ps [MPa]")
        plt.plot(t, np.asarray(log_pd), label="pd [MPa]")
        plt.grid(True); plt.legend()

        plt.figure()
        plt.title("u=(a,b)")
        u = np.asarray(log_u)
        plt.step(t, u[:, 0], where="post", label="a")
        plt.step(t, u[:, 1], where="post", label="b")
        plt.grid(True); plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
