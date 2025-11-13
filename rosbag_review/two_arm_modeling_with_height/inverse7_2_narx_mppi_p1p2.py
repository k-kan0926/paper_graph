#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI with NARX (multi-output) -- SAFE CLAMP VERSION (p1/p2 版, 完全版)

状態1スライス: [theta, p1, p2, dp1, dp2, dz]
出力: [theta_next, dz_next]（NARX予測は学習レンジでクランプ）
入力: U = [u1, u2] = [p1/pmax, p2/pmax] ∈ [0,1]^2
レート制限: |dp1/dt| <= p1_rate, |dp2/dt| <= p2_rate （メタで既定、CLIで上書き可）

主なオプション:
  --no-clip-ref                 : 参照theta_refを学習レンジでクランプしない（安全は w_box / w_soft で担保）
  --theta-min-deg / --theta-max-deg / --theta-margin-deg : 学習レンジの上書き＆余白付与
  --p1-rate / --p2-rate         : レート制限上書き [MPa/s]
  --ref-alpha                   : 参照S字の有効地平線を 1/alpha に圧縮（攻める）
  --theta-soft-margin-deg / --w-soft : 壁の内側からペナルティ（貼り付き回避）
  --u-diff-bias                 : 初期Uに差方向バイアス（u2+=b, u1-=b）
  --auto-udiff                  : 小さな試験入力で θ の符号方向を自動推定して、b の符号を自動決定
  --probe-eps                   : 自動推定で使う差入力量（u2+=eps, u1-=eps）
  --use-dpdt-cost               : レートコストを |dp/dt|^2 に（既定は |Δp|^2）
  --seed                        : 乱数再現
  --no-shift-warmstart          : MPPIウォームスタートのシフトを無効化
"""

import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd, torch

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# -------------------- utils --------------------

def set_seed(seed: int | None):
    if seed is None: return
    np.random.seed(seed); torch.manual_seed(seed)
    try:
        import random; random.seed(seed)
    except Exception:
        pass

def load_meta(meta_path: str):
    meta = json.loads(Path(meta_path).read_text())
    assert "lags" in meta and "feature_names_single_slice" in meta and "mu" in meta and "std" in meta
    return meta

def load_torch_state(pt_path: str):
    try:
        sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(pt_path, map_location="cpu")
    return (sd["net.0.weight"].cpu().numpy(),
            sd["net.0.bias"].cpu().numpy(),
            sd["net.2.weight"].cpu().numpy(),
            sd["net.2.bias"].cpu().numpy(),
            sd["net.4.weight"].cpu().numpy(),
            sd["net.4.bias"].cpu().numpy())


# -------------------- NARX numpy --------------------

class NarxNumpy:
    def __init__(self, meta, weights, pmax: float, dt: float, act: str = "relu",
                 p1_rate_override: float | None = None, p2_rate_override: float | None = None,
                 theta_min_override: float | None = None, theta_max_override: float | None = None):
        self.meta = meta
        self.weights = weights
        self.pmax = float(pmax)
        self.dt = float(dt)
        self.L = int(meta["lags"])
        self.mu = np.asarray(meta["mu"], dtype=np.float32).reshape(1, -1)
        self.std = np.asarray(meta["std"], dtype=np.float32).reshape(1, -1)
        self.act_name = act

        # theta range
        self.theta_lo, self.theta_hi = -math.pi, math.pi
        if "theta_train_minmax" in meta:
            self.theta_lo = float(meta["theta_train_minmax"][0])
            self.theta_hi = float(meta["theta_train_minmax"][1])
        if theta_min_override is not None: self.theta_lo = float(theta_min_override)
        if theta_max_override is not None: self.theta_hi = float(theta_max_override)

        # dz range
        self.dz_lo, self.dz_hi = -0.1, 0.1
        if "dz_train_minmax" in meta:
            self.dz_lo = float(meta["dz_train_minmax"][0])
            self.dz_hi = float(meta["dz_train_minmax"][1])

        # rate limits (prefer p1/p2; fallback ps/pd); allow override
        plim = meta.get("pressure_limits", {}) or {}
        self.p1_rate = float(plim.get("p1_rate_limit_MPa_s", plim.get("ps_rate_limit_MPa_s", 999.0)))
        self.p2_rate = float(plim.get("p2_rate_limit_MPa_s", plim.get("pd_rate_limit_MPa_s", 999.0)))
        if p1_rate_override is not None: self.p1_rate = float(p1_rate_override)
        if p2_rate_override is not None: self.p2_rate = float(p2_rate_override)

        # indices in a slice of 6
        self.idx_theta = lambda j: 6*j + 0
        self.idx_p1    = lambda j: 6*j + 1
        self.idx_p2    = lambda j: 6*j + 2
        self.idx_dp1   = lambda j: 6*j + 3
        self.idx_dp2   = lambda j: 6*j + 4
        self.idx_dz    = lambda j: 6*j + 5

    def _act(self, x):
        if self.act_name == "tanh": return np.tanh(x)
        if self.act_name == "relu": return np.maximum(x, 0.0)
        raise ValueError("act must be 'tanh' or 'relu'")

    def step(self, X: np.ndarray, U: np.ndarray):
        """
        X: (B, 6*L) [theta, p1, p2, dp1, dp2, dz]*L  (front slice j=0 が最新)
        U: (B, 2) with u1=p1/pmax, u2=p2/pmax in [0,1]
        """
        B, L, pmax, dt = X.shape[0], self.L, self.pmax, self.dt

        # action -> proposal pressures
        U = np.clip(U, 0.0, 1.0)
        u1 = U[:, 0:1]; u2 = U[:, 1:2]
        p1 = pmax * u1
        p2 = pmax * u2

        # previous
        p1_prev = X[:, self.idx_p1(0):self.idx_p1(0)+1]
        p2_prev = X[:, self.idx_p2(0):self.idx_p2(0)+1]
        dp1 = (p1 - p1_prev) / dt
        dp2 = (p2 - p2_prev) / dt

        # rate limit
        if self.p1_rate < 900.0: dp1 = np.clip(dp1, -self.p1_rate, self.p1_rate)
        if self.p2_rate < 900.0: dp2 = np.clip(dp2, -self.p2_rate, self.p2_rate)

        # integrate & clip to [0, pmax], then recompute dp from actual
        p1 = np.clip(p1_prev + dp1 * dt, 0.0, pmax)
        p2 = np.clip(p2_prev + dp2 * dt, 0.0, pmax)
        dp1 = (p1 - p1_prev) / dt
        dp2 = (p2 - p2_prev) / dt

        # NARX inference uses previous lag stack (I/O delay consistency)
        z = X.copy()
        z_norm = (z - self.mu) / self.std
        W1, b1, W2, b2, W3, b3 = self.weights
        h1 = self._act(z_norm @ W1.T + b1[None, :])
        h2 = self._act(h1 @ W2.T + b2[None, :])
        y  = h2 @ W3.T + b3[None, :]   # (B,2) -> [theta_next, dz_next]

        theta_next = np.clip(y[:, 0:1], self.theta_lo, self.theta_hi)
        dz_next    = np.clip(y[:, 1:2], self.dz_lo,    self.dz_hi)

        s_next = np.concatenate([theta_next, p1, p2, dp1, dp2, dz_next], axis=1)
        X_next = np.concatenate([s_next, X[:, :6*(L-1)]] , axis=1) if L > 1 else s_next
        return X_next

    def unpack_front(self, X: np.ndarray):
        return (X[:, self.idx_theta(0)],
                X[:, self.idx_p1(0)],
                X[:, self.idx_p2(0)],
                X[:, self.idx_dz(0)])


# -------------------- helpers --------------------

def build_x0_from_context(meta, theta0=0.0, p1_0=0.15, p2_0=0.15):
    L = int(meta["lags"])
    s = np.array([theta0, p1_0, p2_0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.tile(s, (L,))

def infer_x0_from_csv(meta, csv_path: str):
    df = pd.read_csv(csv_path); L = int(meta["lags"])
    need_dp1 = "dp1[MPa/s]" not in df.columns
    need_dp2 = "dp2[MPa/s]" not in df.columns
    if need_dp1 or need_dp2:
        t = df["t[s]"].to_numpy()
        dt = np.median(np.diff(t)) if len(t) > 1 else float(meta.get("dt_est", 0.01))
        if need_dp1: df["dp1[MPa/s]"] = np.gradient(df["p1[MPa]"].to_numpy(), dt)
        if need_dp2: df["dp2[MPa/s]"] = np.gradient(df["p2[MPa]"].to_numpy(), dt)
    cols = ["theta[rad]", "p1[MPa]", "p2[MPa]", "dp1[MPa/s]", "dp2[MPa/s]", "dz[m]"]
    tail = df.tail(L)
    if len(tail) < L:
        first = df.head(1).iloc[0]
        tail = pd.concat([pd.DataFrame([first])]*(L-len(tail)) + [tail], ignore_index=True)
    rows = tail.iloc[::-1]
    return rows[cols].to_numpy().astype(np.float32).reshape(-1,)

def smooth_theta_ref(theta0, theta_target, T):
    t = np.linspace(0., 1., T+1)
    s = 3*t**2 - 2*t**3
    return theta0 + (theta_target - theta0) * s

def infer_udiff_sign(model: NarxNumpy, x: np.ndarray, u_base: np.ndarray, eps: float) -> float:
    """
    差方向(u2+=eps, u1-=eps)が θ を増やすなら +1、減らすなら -1 を返す
    """
    ub = np.clip(u_base, 0.0, 1.0)
    up = np.array([[np.clip(ub[0]-eps, 0.0, 1.0), np.clip(ub[1]+eps, 0.0, 1.0)]], dtype=np.float32)
    um = np.array([[np.clip(ub[0]+eps, 0.0, 1.0), np.clip(ub[1]-eps, 0.0, 1.0)]], dtype=np.float32)
    th_p = model.unpack_front(model.step(x[None, :], up))[0][0]
    th_m = model.unpack_front(model.step(x[None, :], um))[0][0]
    sgn = np.sign(th_p - th_m)
    return 1.0 if sgn == 0 else float(sgn)


# -------------------- MPPI --------------------

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
                      w_box: float,
                      w_soft: float = 0.0,
                      theta_soft_margin_deg: float = 0.0,
                      use_dpdt_cost: bool = False):
    T, Kb, dt = U_nom.shape[0], K, model.dt
    X = np.repeat(x0[None, :], Kb, axis=0)
    noise = sigma_u * np.random.randn(Kb, T, 2).astype(np.float32)
    J = np.zeros((Kb,), dtype=np.float32)
    theta_lo, theta_hi = model.theta_lo, model.theta_hi

    for t in range(T):
        U_t = U_nom[t][None, :] + noise[:, t, :]
        X_next = model.step(X, U_t)

        theta, p1, p2, dz = model.unpack_front(X_next)
        theta_ref_t = theta_refs[t+1]

        p1_prev = X[:, model.idx_p1(0)]
        p2_prev = X[:, model.idx_p2(0)]

        if use_dpdt_cost:
            rate_cost = ((p1 - p1_prev)**2 + (p2 - p2_prev)**2) / (dt**2)
        else:
            rate_cost = ((p1 - p1_prev)**2 + (p2 - p2_prev)**2)

        cost_t = (w_path * (theta - theta_ref_t)**2
                  + w_rate * rate_cost
                  + w_z    * (dz**2))

        # hard box (outside)
        over_hi = np.maximum(0.0, theta - theta_hi)
        over_lo = np.maximum(0.0, theta_lo - theta)
        cost_t += w_box * (over_hi**2 + over_lo**2)

        # soft margins (inside)
        if w_soft > 0.0 and theta_soft_margin_deg > 0.0:
            m = np.radians(theta_soft_margin_deg)
            thr_hi = theta_hi - m
            thr_lo = theta_lo + m
            soft_hi = np.clip(theta - thr_hi, 0.0, None) / m
            soft_lo = np.clip(thr_lo - theta, 0.0, None) / m
            cost_t += w_soft * (soft_hi**2 + soft_lo**2)

        J += cost_t.astype(np.float32)
        X = X_next

    # terminal
    theta, _, _, _ = model.unpack_front(X)
    J += (w_term * (theta - theta_refs[-1])**2).astype(np.float32)
    over_hi = np.maximum(0.0, theta - theta_hi)
    over_lo = np.maximum(0.0, theta_lo - theta)
    J += w_box * (over_hi**2 + over_lo**2)

    lam = max(lam, 1e-6)
    J_min = float(np.min(J))
    w = np.exp(-(J - J_min) / lam)
    w /= (np.sum(w) + 1e-8)

    U_new = np.zeros_like(U_nom)
    for t in range(T):
        U_t_samp = np.clip(U_nom[t][None, :] + noise[:, t, :], 0.0, 1.0)
        U_new[t] = np.sum(w[:, None] * U_t_samp, axis=0)

    return U_new[0].copy(), U_new, {"J_min": J_min, "J_mean": float(np.mean(J))}


# -------------------- main --------------------

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
    ap.add_argument("--act", choices=["tanh", "relu"], default="relu")
    ap.add_argument("--w-term", type=float, default=8.0)
    ap.add_argument("--w-path", type=float, default=5.0)
    ap.add_argument("--w-rate", type=float, default=0.10)
    ap.add_argument("--w-z", type=float, default=0.10)
    ap.add_argument("--w-box", type=float, default=200.0)
    ap.add_argument("--plot", action="store_true")
    # extras
    ap.add_argument("--no-clip-ref", action="store_true")
    ap.add_argument("--theta-min-deg", type=float, default=None)
    ap.add_argument("--theta-max-deg", type=float, default=None)
    ap.add_argument("--theta-margin-deg", type=float, default=0.0)
    ap.add_argument("--p1-rate", type=float, default=None)
    ap.add_argument("--p2-rate", type=float, default=None)
    ap.add_argument("--ref-alpha", type=float, default=1.0)
    ap.add_argument("--theta-soft-margin-deg", type=float, default=0.0)
    ap.add_argument("--w-soft", type=float, default=0.0)
    ap.add_argument("--u-diff-bias", type=float, default=0.0)
    ap.add_argument("--auto-udiff", action="store_true",
                    help="差方向の符号を毎ステップ自動推定して u-diff-bias の符号を決める")
    ap.add_argument("--probe-eps", type=float, default=0.02,
                    help="自動推定で使う差入力量（0..1）")
    ap.add_argument("--use-dpdt-cost", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-shift-warmstart", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    meta = load_meta(args.meta)
    weights = load_torch_state(args.model)

    dt = float(meta["dt_est"]) if args.dt_override <= 0.0 else float(args.dt_override)
    model = NarxNumpy(
        meta, weights, pmax=args.pmax, dt=dt, act=args.act,
        p1_rate_override=args.p1_rate, p2_rate_override=args.p2_rate,
        theta_min_override=(math.radians(args.theta_min_deg) if args.theta_min_deg is not None else None),
        theta_max_override=(math.radians(args.theta_max_deg) if args.theta_max_deg is not None else None),
    )
    if args.theta_margin_deg:
        m = math.radians(args.theta_margin_deg)
        model.theta_lo -= m; model.theta_hi += m

    print(f"[info] dt = {dt:.6f} s")
    print(f"[info] theta range (deg): {math.degrees(model.theta_lo):.2f} .. {math.degrees(model.theta_hi):.2f}")
    print(f"[info] rate limits p1/p2 (MPa/s): {model.p1_rate} / {model.p2_rate}")

    # initial state
    if args.context_csv:
        x = infer_x0_from_csv(meta, args.context_csv)
        theta0 = float(x[0])
    else:
        theta0 = 0.0
        x = build_x0_from_context(meta, theta0=theta0, p1_0=0.15, p2_0=0.15)

    theta_target = math.radians(args.theta_target_deg)
    print(f"[info] target theta (deg): {args.theta_target_deg:.2f}")

    T = args.horizon
    # initial U_nom (with optional user bias)
    u1_0 = np.clip(0.15 / args.pmax - args.u_diff_bias, 0.0, 1.0)
    u2_0 = np.clip(0.15 / args.pmax + args.u_diff_bias, 0.0, 1.0)
    U_nom = np.tile(np.array([u1_0, u2_0], dtype=np.float32), (T, 1))

    log_t, log_theta, log_theta_ref = [], [], []

    for k in range(args.steps):
        theta_cur = float(x[0])

        # aggressive reference: compress effective horizon
        T_eff = max(2, int(T / max(1e-6, args.ref_alpha)))
        theta_refs = smooth_theta_ref(theta_cur, theta_target, T_eff)
        if len(theta_refs) < T + 1:
            theta_refs = np.concatenate([theta_refs, np.full((T+1-len(theta_refs),), theta_refs[-1])])

        if not args.no_clip_ref:
            theta_refs = np.clip(theta_refs, model.theta_lo + 1e-3, model.theta_hi - 1e-3)

        # auto sign for u-diff bias (per step)
        if args.auto-udiff:
            b_abs = abs(args.u_diff_bias)
            if b_abs > 0.0:
                sgn = infer_udiff_sign(model, x, U_nom[0], args.probe-eps if hasattr(args, "probe-eps") else args.probe_eps)
                # ↑ argparse が '-' を '_' に変換するので安全側で hasattr チェック
                sgn = infer_udiff_sign(model, x, U_nom[0], args.probe_eps)
                b = np.clip(b_abs * sgn, -0.5, 0.5)
                U_nom[:, 0] = np.clip(U_nom[:, 0] - b, 0.0, 1.0)  # u1 -= b
                U_nom[:, 1] = np.clip(U_nom[:, 1] + b, 0.0, 1.0)  # u2 += b
                if k == 0:
                    print(f"[info] auto-udiff sign = {int(np.sign(b)):+d}, bias={b:.3f}")

        u0, U_new, info = mppi_control_step(
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
            w_soft=args.w_soft,
            theta_soft_margin_deg=args.theta_soft_margin_deg,
            use_dpdt_cost=args.use_dpdt_cost,
        )

        # apply one step
        x_next = model.step(x[None, :], u0[None, :])[0]
        theta, _, _, _ = model.unpack_front(x_next[None, :])

        log_t.append(k * dt)
        log_theta.append(theta[0])
        log_theta_ref.append(theta_refs[1])

        # receding warm-start (shift)
        U_nom = U_new if args.no_shift_warmstart else np.vstack([U_new[1:], U_new[-1:]])
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
