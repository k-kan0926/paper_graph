#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-1: Model zoo benchmark (+ multi-CSV (dyn/stat) support, z(ps,pd) learning/design,
progress logs, vectorized static eval)

- Static + shared dynamics: HW_Poly / HW_RBF / PWA / GP / MonoNN
- Sequence: ARX / NARX / NLSS(GRU) / Koopman(EDMD)

Inputs (choose one):
  --csv <single.csv>
  or
  --dyn_csvs D1 D2 ... [--stat_csvs S1 S2 ...]

Outputs:
  out-dir/summary.json (or summary_partial.json on Ctrl+C)
  out-dir/{hw_poly|hw_rbf|pwa|gp|mononn}_meta.json
  out-dir/{hw_poly|hw_rbf|pwa|gp|mononn}_common_meta.npz
  out-dir/{arx|narx|nlss|koopman}_meta.json
  out-dir/narx_common_meta.npz
python phase1_model_zoo_multi.py \
  --dyn_csvs out/dyn_prbs.csv out/dyn_multi.csv out/dyn_cyrip.csv \
  --stat_csvs out/static1.csv out/static2.csv \
  --out-dir out_zoo_multi --enable-all \
  --delay-grid 0,1,2,3 \
  --z-mode learn --z-col dz[m] --z-source stat --z-qstatic-from-stat \
  --z-feat quad --z-ss-theta-eps 0.01 --z-ss-dz-eps 0.002
"""

import os, json, math, argparse, time, warnings
import numpy as np
import numpy.linalg as LA

# ---------- optional deps ----------
try:
    import sklearn
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    SKLEARN_GP_OK = True
except Exception:
    SKLEARN_GP_OK = False

try:
    import scipy
    from scipy.signal import savgol_filter
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(max(1, os.cpu_count()//2))

# ----------------- utils -----------------
def smooth_and_deriv(x, dt, win=17, poly=3):
    x = np.asarray(x, float)
    if not SCIPY_OK or len(x) < max(7, poly+2):
        xs = x
        dx = np.gradient(xs, dt)
        return xs, dx
    win = min(int(win)|1, max(5, (len(x)//2)*2-1))  # odd
    xs = savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
    dx = savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=dt, mode="interp")
    return xs, dx

def rollout_rmse_firstorder(ps, pd, th, dt, d, alpha, kS, kD, theta_stat_vec):
    """Single-session rollout RMSE for the first-order shared dynamics."""
    N = len(ps)
    start = d + 1
    th_pred = th.copy()
    for k in range(start, N-1):
        ku = k - d
        kup = k - d - 1
        ths = float(theta_stat_vec(ps[ku:ku+1], pd[ku:ku+1])[0])
        dS  = (ps[ku] - ps[kup]) / dt
        dD  = (pd[ku] - pd[kup]) / dt
        rhs = alpha*(ths - th_pred[k]) + kS*dS + kD*dD
        th_pred[k+1] = th_pred[k] + dt*rhs
    err = th_pred[start:] - th[start:]
    if len(err)==0: return float("inf")
    return float(np.sqrt(np.mean(err**2)))

def _fit_dyn_with_delay_multi(theta_stat_vec, sessions, sg_win, sg_poly,
                              delay_grid, l2_dyn=5e-3, kdelta_min=-2.0, log_prefix=""):
    """
    Multi-session variant.
    sessions: list of dicts with keys {'ps','pd','th','dt','name'}
    Returns: best(dict), logs(list), per_delay_detail(list)
    """
    if not SCIPY_OK:
        raise RuntimeError("scipy is required for fit_dyn_with_delay().")
    from scipy.optimize import lsq_linear

    t0 = time.time()
    delays = [int(x) for x in delay_grid.split(",") if x.strip()!=""]
    best = None; logs = []; per_delay_detail=[]
    print(f"{log_prefix}  [dyn] start grid-search over delays: {delays} | sessions={len(sessions)}")

    for d in delays:
        # stack across sessions
        X_stack = []; y_stack = []
        # also keep per-session RMSE for reporting
        sess_rmses = []
        scales_all = []  # we will standardize per-delay globally
        # first pass: build and stack
        for sess in sessions:
            ps = sess['ps']; pd = sess['pd']; th = sess['th']; dt = sess['dt']
            ps_s, _ = smooth_and_deriv(ps, dt, win=sg_win, poly=sg_poly)
            pd_s, _ = smooth_and_deriv(pd, dt, win=sg_win, poly=sg_poly)
            th_s, dth= smooth_and_deriv(th, dt, win=sg_win, poly=sg_poly)
            N = len(ps)
            start = d+1
            if N <= start+1:  # not enough length
                continue
            idx = np.arange(start, N)
            ku = idx - d
            kup = idx - d - 1
            ths = theta_stat_vec(ps_s[ku], pd_s[ku])  # array
            phi1= ths - th_s[idx]
            dS  = (ps_s[ku] - ps_s[kup]) / dt
            dD  = (pd_s[ku] - pd_s[kup]) / dt
            y   = dth[idx]
            X   = np.column_stack([phi1, dS, dD])
            X_stack.append(X); y_stack.append(y)

        if not X_stack:
            logs.append(dict(d=d, error="no usable samples"))
            continue

        X_all = np.vstack(X_stack)
        y_all = np.hstack(y_stack)

        # global column standardization for this delay
        scales = np.std(X_all, axis=0, ddof=1); scales[scales<1e-8]=1.0
        Xs = X_all / scales
        lam = float(l2_dyn)
        X_aug = np.vstack([Xs, math.sqrt(lam)*np.eye(3)])
        y_aug = np.hstack([y_all, np.zeros(3)])

        lb = [0.0, -np.inf, float(kdelta_min)]
        ub = [np.inf, np.inf, np.inf]
        sol = lsq_linear(X_aug, y_aug, bounds=(lb,ub), method="trf", lsmr_tol='auto', verbose=0)
        alpha, kS, kD = (sol.x / scales).tolist()

        # evaluate RMSE per session, then average
        rmse_list = []
        for sess in sessions:
            rm = rollout_rmse_firstorder(sess['ps'], sess['pd'], sess['th'], sess['dt'],
                                         d, alpha, kS, kD, theta_stat_vec)
            if np.isfinite(rm):
                rmse_list.append(rm)
        rmse_mean = float(np.mean(rmse_list)) if rmse_list else float("inf")
        detail = dict(d=d, alpha=alpha, kS=kS, kD=kD, rmse_mean=rmse_mean, rmse_each=rmse_list)
        per_delay_detail.append(detail)

        print(f"{log_prefix}    delay={d}: RMSE_mean={rmse_mean:.3f} "
              f"(alpha={alpha:.3f}, kS={kS:.3f}, kD={kD:.3f}, sessions={len(rmse_list)})")

        if (best is None) or (rmse_mean < best["rmse"]):
            best = dict(d=d, alpha=alpha, kS=kS, kD=kD, rmse=rmse_mean)

    print(f"{log_prefix}  [dyn] done in {time.time()-t0:.2f}s -> best: d={best['d']}, RMSE_mean={best['rmse']:.3f}")
    return best, per_delay_detail

def standardize_cols(S, D):
    muS, sdS = float(np.mean(S)), float(np.std(S)+1e-8)
    muD, sdD = float(np.mean(D)), float(np.std(D)+1e-8)
    Sh = (S-muS)/sdS
    Dh = (D-muD)/sdD
    return (muS, sdS, muD, sdD, Sh, Dh)

def train_val_split(N, val_ratio):
    val_n = max(1, int(N*val_ratio))
    tr = np.arange(0, N-val_n)
    va = np.arange(N-val_n, N)
    return tr, va

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ===== CSV loading (single or multi) =====
def _load_single_csv(path, col_sum, col_diff, col_theta, col_time):
    import pandas as pd
    df = pd.read_csv(path, comment="#").dropna().reset_index(drop=True)
    for c in [col_sum, col_diff, col_theta]:
        if c not in df.columns:
            raise RuntimeError(f"{path}: missing column {c}")
    if col_time in df.columns:
        t = df[col_time].to_numpy(float)
        dt = float(np.nanmedian(np.diff(t)))
        if not np.isfinite(dt) or dt<=0: dt = 0.01
    else:
        dt = 0.01
    return df, dt

def load_dynamic_csvs(paths, col_sum, col_diff, col_theta, col_time):
    """Return list of sessions [{'ps','pd','th','dt','name'}], and concatenated arrays for static fit."""
    sessions = []
    Ps_all, Pd_all, Th_all = [], [], []
    for p in paths:
        df, dt = _load_single_csv(p, col_sum, col_diff, col_theta, col_time)
        ps = df[col_sum].to_numpy(float)
        pd = df[col_diff].to_numpy(float)
        th = df[col_theta].to_numpy(float)
        sessions.append(dict(ps=ps, pd=pd, th=th, dt=dt, name=os.path.basename(p)))
        Ps_all.append(ps); Pd_all.append(pd); Th_all.append(th)
    if not sessions:
        raise RuntimeError("No dynamic CSVs loaded.")
    ps_all = np.concatenate(Ps_all)
    pd_all = np.concatenate(Pd_all)
    th_all = np.concatenate(Th_all)
    # representative dt for metadata: median over sessions
    dt_rep = float(np.median([s['dt'] for s in sessions]))
    return sessions, ps_all, pd_all, th_all, dt_rep

def load_static_csvs(paths, z_col, col_sum, col_diff, col_theta, col_time,
                     use_quasi_static=False, ss_theta_eps=None, ss_z_eps=None, dz_col=None):
    """Collect (ps,pd,z) from static CSVs for z-learning."""
    import pandas as pd
    Ps, Pd, Z = [], [], []
    for p in paths:
        df = pd.read_csv(p, comment="#").dropna().reset_index(drop=True)
        if z_col not in df.columns:
            continue
        if use_quasi_static and (dz_col is not None) and (col_time in df.columns):
            # select by small slopes of theta and dz
            t = df[col_time].to_numpy(float)
            dt = float(np.nanmedian(np.diff(t))) if len(t)>=2 else 0.01
            dth = np.gradient(df[col_theta].to_numpy(float), dt)
            ddz = np.gradient(df[dz_col].to_numpy(float), dt)
            mask = (np.abs(dth) < float(ss_theta_eps or 1e-2)) & (np.abs(ddz) < float(ss_z_eps or 2e-3))
            df = df.loc[mask].reset_index(drop=True)
            if len(df)==0:
                continue
        Ps.append(df[col_sum].to_numpy(float))
        Pd.append(df[col_diff].to_numpy(float))
        Z.append(df[z_col].to_numpy(float))
    if not Ps:
        return None, None, None
    return np.concatenate(Ps), np.concatenate(Pd), np.concatenate(Z)

# ===== z(ps,pd): feature builder & fitting / design =====
def _build_z_features(ps, pd, mode="quad"):
    ps = np.asarray(ps, float); pd = np.asarray(pd, float)
    if mode == "lin":
        Phi = np.column_stack([np.ones_like(ps), ps, pd, ps*pd])
        names = ["1", "ps", "pd", "ps*pd"]
    else:
        Phi = np.column_stack([np.ones_like(ps), ps, pd, ps*pd, ps**2, pd**2])
        names = ["1", "ps", "pd", "ps*pd", "ps^2", "pd^2"]
    return Phi, names

def _reg_metrics(y_true, y_pred):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    sst = float(np.sum((y_true - np.mean(y_true))**2))
    sse = float(np.sum(err**2))
    r2  = float(1.0 - (sse / (sst + 1e-12)))
    return dict(rmse=rmse, mae=mae, r2=r2)

def fit_z_learn(ps, pd, z_obs, feat_mode="quad", lam=1e-6, val_ratio=0.2):
    Phi, names = _build_z_features(ps, pd, feat_mode)
    N = len(ps)
    tr, va = train_val_split(N, val_ratio)
    A_tr = Phi[tr]; b_tr = np.asarray(z_obs, float)[tr]
    A_va = Phi[va]; b_va = np.asarray(z_obs, float)[va]
    A_aug = np.vstack([A_tr, np.sqrt(lam)*np.eye(A_tr.shape[1])])
    b_aug = np.hstack([b_tr, np.zeros(A_tr.shape[1])])
    w, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    pred_tr = A_tr @ w
    pred_va = A_va @ w
    m_tr = _reg_metrics(b_tr, pred_tr)
    m_va = _reg_metrics(b_va, pred_va)
    metrics = {f"{k}_tr": v for k,v in m_tr.items()}
    metrics.update({f"{k}_va": v for k,v in m_va.items()})
    return w, names, metrics

def fit_or_design_z(ps, pd, z_obs=None, feat_mode="quad",
                    w_sigma2=1.0, w_delta2=1.0, w_cross=0.0,
                    lam=1e-6, val_ratio=0.2):
    if z_obs is None:
        Phi, names = _build_z_features(ps, pd, feat_mode)
        coef = np.zeros(Phi.shape[1], float)
        if feat_mode == "lin":
            if "ps*pd" in names:
                coef[names.index("ps*pd")] = float(w_cross)
        else:
            coef[names.index("ps^2")] = float(w_sigma2)
            coef[names.index("pd^2")] = float(w_delta2)
            if "ps*pd" in names:
                coef[names.index("ps*pd")] = float(w_cross)
        return coef, names, None
    else:
        w, names, metrics = fit_z_learn(ps, pd, np.asarray(z_obs, float),
                                        feat_mode=feat_mode, lam=lam, val_ratio=val_ratio)
        return w, names, metrics

def save_common_meta_npz(out_path_npz, dt, muS, sdS, muD, sdD, dyn, z_coef, z_feat_names, z_metrics=None):
    z_rmse_va = np.array(z_metrics["rmse_va"]) if (z_metrics and "rmse_va" in z_metrics) else np.array(np.nan)
    z_mae_va  = np.array(z_metrics["mae_va"])  if (z_metrics and "mae_va"  in z_metrics) else np.array(np.nan)
    z_r2_va   = np.array(z_metrics["r2_va"])   if (z_metrics and "r2_va"   in z_metrics) else np.array(np.nan)
    np.savez(out_path_npz,
             dt=np.array(dt, dtype=np.float64),
             muS=np.array(muS, dtype=np.float64),
             sdS=np.array(sdS, dtype=np.float64),
             muD=np.array(muD, dtype=np.float64),
             sdD=np.array(sdD, dtype=np.float64),
             alpha=np.array(dyn["alpha"], dtype=np.float64),
             kSigma=np.array(dyn["kS"], dtype=np.float64),
             kDelta=np.array(dyn["kD"], dtype=np.float64),
             delay=np.array(dyn["d"], dtype=np.int64),
             z_coef=np.array(z_coef, dtype=np.float64),
             z_feat_names=np.array(z_feat_names),
             z_rmse_va=z_rmse_va,
             z_mae_va=z_mae_va,
             z_r2_va=z_r2_va)

# ----------------- Static models -----------------
class StaticModel:
    def fit(self, S, D, Y): ...
    def predict(self, S, D): ...

class StaticPoly(StaticModel):
    def __init__(self, deg=3, alpha=1e-3):
        if not SKLEARN_OK: raise RuntimeError("sklearn needed for StaticPoly.")
        self.poly = PolynomialFeatures(degree=deg, include_bias=True)
        self.ridge = Ridge(alpha=alpha, fit_intercept=False)
    def fit(self, S, D, Y):
        t0=time.time()
        X = self.poly.fit_transform(np.column_stack([S,D]))
        self.ridge.fit(X, Y)
        print(f"    [fit HW_Poly] {len(Y)} samples in {time.time()-t0:.2f}s")
        return self
    def predict(self, S, D):
        X = self.poly.transform(np.column_stack([S,D]))
        return self.ridge.predict(X)

class StaticRBF(StaticModel):
    def __init__(self, n_centers=40, length=0.6, alpha=1e-3, kmeans=True, seed=0):
        if not SKLEARN_OK: raise RuntimeError("sklearn needed for StaticRBF.")
        self.n_centers=n_centers; self.length=float(length); self.alpha=float(alpha)
        self.kmeans=bool(kmeans); self.seed=int(seed); self.W=None
    def _features(self, S, D, C):
        X = np.column_stack([S,D])
        diff = X[:,None,:] - C[None,:,:]
        r2 = np.sum(diff*diff, axis=2)
        Phi = np.exp(-0.5*r2/(self.length**2))
        Phi = np.hstack([np.ones((Phi.shape[0],1)), Phi])
        return Phi
    def fit(self, S, D, Y):
        t0=time.time()
        X = np.column_stack([S,D])
        if self.kmeans:
            km = KMeans(n_clusters=self.n_centers, random_state=self.seed).fit(X)
            C = km.cluster_centers_
        else:
            smin,smax = np.quantile(S,0.02), np.quantile(S,0.98)
            dmin,dmax = np.quantile(D,0.02), np.quantile(D,0.98)
            gs = int(np.sqrt(self.n_centers))
            Ss = np.linspace(smin,smax,gs); Ds = np.linspace(dmin,dmax,gs)
            C = np.array([(a,b) for a in Ss for b in Ds], float)
        Phi = self._features(S,D,C)
        lam = self.alpha
        A = Phi.T@Phi + lam*np.eye(Phi.shape[1]); b = Phi.T@Y
        self.W = LA.solve(A,b); self.C=C
        print(f"    [fit HW_RBF] {len(Y)} samples, centers={len(C)} in {time.time()-t0:.2f}s")
        return self
    def predict(self, S, D):
        Phi = self._features(S,D,self.C)
        return Phi@self.W

class StaticPWA(StaticModel):
    def __init__(self, n_bins=8, alpha=1e-3):
        self.n_bins=int(n_bins); self.alpha=float(alpha)
    def fit(self, S, D, Y):
        t0=time.time()
        edges = np.quantile(D, np.linspace(0,1,self.n_bins+1))
        self.edges = edges; self.W=[]
        for bi in range(self.n_bins):
            lo, hi = edges[bi], edges[bi+1]
            idx = np.where((D>=lo)&(D<=hi))[0]
            if len(idx) < 10:
                self.W.append(np.zeros((3,))); continue
            Xb = np.column_stack([np.ones(len(idx)), S[idx], D[idx]])
            Yb = Y[idx]
            A = Xb.T@Xb + self.alpha*np.eye(3); b = Xb.T@Yb
            self.W.append(LA.solve(A,b))
        print(f"    [fit PWA] {len(Y)} samples, bins={self.n_bins} in {time.time()-t0:.2f}s")
        return self
    def predict(self, S, D):
        Y = np.zeros_like(S)
        for i in range(len(S)):
            d = D[i]
            bi = min(self.n_bins-1, max(0, int(np.searchsorted(self.edges, d)-1)))
            w = self.W[bi]
            Y[i] = w[0] + w[1]*S[i] + w[2]*D[i]
        return Y

class StaticGP(StaticModel):
    def __init__(self, gp_maxn=8000, seed=0):
        if not SKLEARN_GP_OK: raise RuntimeError("sklearn GP needed for StaticGP.")
        self.gp_maxn=int(gp_maxn); self.seed=int(seed)
    def fit(self, S, D, Y):
        t0=time.time()
        X = np.column_stack([S,D])
        if len(X) > self.gp_maxn:
            rs = np.random.RandomState(self.seed)
            idx = rs.choice(len(X), self.gp_maxn, replace=False)
            X, Y = X[idx], Y[idx]
            print(f"    [GP] subsampled to {len(X)}")
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=[1.0,1.0], length_scale_bounds=(1e-2,1e2)) \
                 + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6,1e-1))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=2, normalize_y=True, random_state=self.seed)
        self.gp.fit(X, Y)
        print(f"    [fit GP] {len(X)} samples in {time.time()-t0:.2f}s | kernel={self.gp.kernel_}")
        return self
    def predict(self, S, D):
        X = np.column_stack([S,D])
        return self.gp.predict(X)

class MonoDeltaNN_torch(nn.Module):
    def __init__(self, M=6, hidden=32, c_grid=None, learn_centers=False):
        super().__init__()
        self.M=M
        self.c = nn.Parameter(torch.tensor(c_grid, dtype=torch.float32), requires_grad=bool(learn_centers))
        self.enc = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.head_A = nn.Linear(hidden, 1)
        self.head_w = nn.Linear(hidden, M)
        self.head_s = nn.Linear(hidden, M)
        nn.init.zeros_(self.head_w.weight); nn.init.zeros_(self.head_w.bias)
        nn.init.zeros_(self.head_s.weight); nn.init.constant_(self.head_s.bias, 1.0)
    def forward(self, Sh, Dh):
        h = self.enc(Sh)
        A = self.head_A(h)
        w = torch.nn.functional.softplus(self.head_w(h)) + 1e-6
        s = torch.nn.functional.softplus(self.head_s(h)) + 1e-6
        d = Dh - self.c.view(1,-1)
        bank = torch.tanh(s*d)
        return A + (w*bank).sum(dim=1, keepdim=True)

class StaticMonoNN(StaticModel):
    def __init__(self, M=6, hidden=32, epochs=200, lr=1e-3, batch=512, learn_centers=True, seed=0):
        self.M=M; self.hidden=hidden; self.epochs=epochs; self.lr=lr; self.batch=batch
        self.learn_centers=learn_centers; self.seed=seed
        self.model=None; self.state_dict=None; self.c_grid=None
    def fit(self, S, D, Y):
        t0=time.time()
        S = S.astype(np.float32); D=D.astype(np.float32); Y=Y.astype(np.float32)
        c_grid = np.linspace(np.percentile(D,5), np.percentile(D,95), self.M).astype(np.float32)
        self.c_grid = c_grid
        ds = torch.utils.data.TensorDataset(torch.tensor(S).view(-1,1), torch.tensor(D).view(-1,1), torch.tensor(Y).view(-1,1))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True, drop_last=False)
        model = MonoDeltaNN_torch(M=self.M, hidden=self.hidden, c_grid=c_grid, learn_centers=self.learn_centers)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)
        loss_fn = nn.HuberLoss(delta=3.0)
        model.train()
        for ep in range(self.epochs):
            for Sb,Db,Yb in dl:
                opt.zero_grad()
                Yh = model(Sb,Db)
                loss = loss_fn(Yh, Yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            if (ep+1)%50==0:
                print(f"      [MonoNN] epoch {ep+1}/{self.epochs}")
        self.model = model.eval()
        self.state_dict = {k:v.cpu() for k,v in model.state_dict().items()}
        print(f"    [fit MonoNN] {len(Y)} samples in {time.time()-t0:.2f}s")
        return self
    def predict(self, S, D):
        with torch.no_grad():
            S = torch.tensor(S, dtype=torch.float32).view(-1,1)
            D = torch.tensor(D, dtype=torch.float32).view(-1,1)
            Y = self.model(S,D).view(-1).cpu().numpy()
        return Y

# ----------------- ARX / NARX -----------------
def build_lag_matrix(U, Y, lag_u=3, lag_y=3, delay=0):
    N = len(Y); T0 = max(lag_y, lag_u+delay)
    rows=[]; ys=[]
    for t in range(T0, N):
        row=[]
        for k in range(1, lag_y+1): row.append(Y[t-k])
        for k in range(delay, delay+lag_u): row.extend(U[t-1-k])
        rows.append(row); ys.append(Y[t])
    X = np.array(rows, float); y = np.array(ys, float)
    return X, y, T0

def arx_fit_predict_multi(sessions, lag_y=3, lag_u=3, delay=0, alpha=1e-3, log_prefix=""):
    if not SKLEARN_OK: raise RuntimeError("sklearn needed for ARX.")
    t0=time.time()
    # concat sessions
    U_all = []; y_all = []; T0_min = None; th_all=[]
    # build per session, then concat
    for sess in sessions:
        U = np.column_stack([sess['ps'], sess['pd']])
        X, y, T0 = build_lag_matrix(U, sess['th'], lag_u=lag_u, lag_y=lag_y, delay=delay)
        if X.size==0: continue
        U_all.append((U, T0))
        y_all.append((y, T0))
        th_all.append(sess['th'])
        if T0_min is None or T0 < T0_min:
            T0_min = T0
    if not y_all:
        return float("inf"), dict(model="ARX", error="no usable samples")

    # train on concatenation of features/targets built above
    # (we rebuild X_all to align with a single ridge fit)
    X_cat=[]; y_cat=[]
    for sess in sessions:
        U = np.column_stack([sess['ps'], sess['pd']])
        X, y, _ = build_lag_matrix(U, sess['th'], lag_u=lag_u, lag_y=lag_y, delay=delay)
        if X.size==0: continue
        X_cat.append(X); y_cat.append(y)
    X_cat = np.vstack(X_cat); y_cat = np.hstack(y_cat)
    mdl = Ridge(alpha=alpha, fit_intercept=True).fit(X_cat, y_cat)

    # rollout RMSE per session then average
    rmse_list=[]
    for sess in sessions:
        th = sess['th']; U = np.column_stack([sess['ps'], sess['pd']])
        N = len(th); yhat = th.copy()
        T0 = max(lag_y, lag_u+delay)
        for t in range(T0, N):
            feats=[]
            for k in range(1, lag_y+1): feats.append(yhat[t-k])
            for k in range(delay, delay+lag_u): feats.extend(U[t-1-k])
            yhat[t] = mdl.predict(np.array(feats,float).reshape(1,-1))[0]
        rmse = float(np.sqrt(np.mean((yhat[T0:]-th[T0:])**2))) if N>T0 else float("inf")
        rmse_list.append(rmse)
    rmse_mean = float(np.mean(rmse_list))
    print(f"{log_prefix}  [ARX] delay={delay}: RMSE_mean={rmse_mean:.3f} in {time.time()-t0:.2f}s")
    return rmse_mean, dict(model="ARX", coef=mdl.coef_.tolist(), intercept=float(mdl.intercept_),
                           lag_y=lag_y, lag_u=lag_u, delay=delay)

def narx_fit_predict_multi(sessions, lag_y=2, lag_u=2, delay=0, degree=2, alpha=1e-3, log_prefix=""):
    if not SKLEARN_OK: raise RuntimeError("sklearn needed for NARX.")
    t0=time.time()
    # build & concat
    X_cat=[]; y_cat=[]; T0 = max(lag_y, lag_u+delay)
    for sess in sessions:
        U = np.column_stack([sess['ps'], sess['pd']])
        X, y, _ = build_lag_matrix(U, sess['th'], lag_u=lag_u, lag_y=lag_y, delay=delay)
        if X.size==0: continue
        X_cat.append(X); y_cat.append(y)
    if not X_cat:
        return float("inf"), dict(model="NARX", error="no usable samples")
    X_cat = np.vstack(X_cat); y_cat = np.hstack(y_cat)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X_cat)
    mdl = Ridge(alpha=alpha, fit_intercept=False).fit(Xp, y_cat)

    # rollout per session
    rmse_list=[]
    for sess in sessions:
        th = sess['th']; U = np.column_stack([sess['ps'], sess['pd']])
        N = len(th); yhat = th.copy()
        for t in range(T0, N):
            feats=[]
            for k in range(1, lag_y+1): feats.append(yhat[t-k])
            for k in range(delay, delay+lag_u): feats.extend(U[t-1-k])
            xp = poly.transform(np.array(feats,float).reshape(1,-1))
            yhat[t] = mdl.predict(xp)[0]
        rmse = float(np.sqrt(np.mean((yhat[T0:]-th[T0:])**2))) if N>T0 else float("inf")
        rmse_list.append(rmse)
    rmse_mean = float(np.mean(rmse_list))

    try:
        poly_feature_names = [str(s) for s in poly.get_feature_names_out()]
    except Exception:
        poly_feature_names = None

    meta = dict(model="NARX", degree=degree, coef=mdl.coef_.tolist(),
                lag_y=lag_y, lag_u=lag_u, delay=delay, T0=T0,
                poly_feature_names=poly_feature_names)
    print(f"{log_prefix}  [NARX] delay={delay}: RMSE_mean={rmse_mean:.3f} in {time.time()-t0:.2f}s")
    return rmse_mean, meta

# ----------------- NLSS (GRU) -----------------
class GRU_NLSS(nn.Module):
    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, u, h0=None):
        y, h = self.gru(u, h0)
        y = self.head(y)
        return y.squeeze(-1), h

def nlss_gru_fit_rollout_multi(sessions, epochs=60, hidden=32, lr=1e-3, seed=0, log_prefix=""):
    torch.manual_seed(seed)
    t0=time.time()
    # concatenate along time (simple approach)
    u = np.concatenate([np.stack([s['ps'], s['pd']], axis=1) for s in sessions], axis=0).astype(np.float32)
    y = np.concatenate([s['th'] for s in sessions], axis=0).astype(np.float32)
    model = GRU_NLSS(in_dim=2, hidden=hidden)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ut = torch.tensor(u[None,...])  # [1,T,2]
    yt = torch.tensor(y[None,...])  # [1,T]
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        yhat, _ = model(ut)
        loss = loss_fn(yhat, yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (ep+1)%10==0:
            print(f"{log_prefix}  [NLSS_GRU] epoch {ep+1}/{epochs}, loss={float(loss):.4f}")
    model.eval()
    with torch.no_grad():
        yhat, _ = model(ut)
        yhat = yhat.numpy().reshape(-1)
    T0 = 5
    rmse = float(np.sqrt(np.mean((yhat[T0:]-y[T0:])**2)))
    print(f"{log_prefix}  [NLSS_GRU] done in {time.time()-t0:.2f}s, RMSE_mean={rmse:.3f}")
    return rmse, dict(model="NLSS_GRU", hidden=hidden, epochs=epochs)

# ----------------- Koopman / EDMD -----------------
def poly_dict(XU, degree=2):
    if not SKLEARN_OK:
        x = XU
        feats = [np.ones((len(x),1)), x]
        if degree>=2:
            q=[]
            for i in range(x.shape[1]):
                for j in range(i, x.shape[1]):
                    q.append((x[:,i:i+1]*x[:,j:j+1]))
            feats.append(np.hstack(q))
        return np.hstack(feats)
    else:
        pf = PolynomialFeatures(degree=degree, include_bias=True)
        return pf.fit_transform(XU)

def koopman_edmd_fit_rollout_multi(sessions, degree=2, log_prefix=""):
    t0=time.time()
    XU = np.concatenate([np.column_stack([s['th'], s['ps'], s['pd']]) for s in sessions], axis=0)
    Z = poly_dict(XU, degree=degree)
    Zp = Z[1:]; Zk = Z[:-1]
    K = LA.lstsq(Zk, Zp, rcond=None)[0]
    W = LA.lstsq(Z, XU[:,0], rcond=None)[0]
    z = Z[0].copy(); yhat = np.zeros(XU.shape[0]); yhat[0] = XU[0,0]
    for k in range(1, len(yhat)):
        z = z @ K
        yhat[k] = float(z @ W)
    T0=5
    rmse = float(np.sqrt(np.mean((yhat[T0:]-XU[T0:,0])**2)))
    print(f"{log_prefix}  [EDMD] degree={degree}: RMSE_mean={rmse:.3f} in {time.time()-t0:.2f}s")
    return rmse, dict(model="Koopman_EDMD", degree=degree)

# ----------------- wrapper: static + shared dynamics -----------------
def eval_static_plus_dyn_multi(name, static_model, sessions, sg_win, sg_poly, delay_grid,
                               muS, sdS, muD, sdD, log_prefix=""):
    # vectorized closure: accepts arrays
    def theta_stat_vec(ps_array, pd_array):
        Sh = (np.asarray(ps_array) - muS)/sdS
        Dh = (np.asarray(pd_array) - muD)/sdD
        return static_model.predict(Sh, Dh)

    best_dyn, dyn_logs = _fit_dyn_with_delay_multi(theta_stat_vec, sessions,
                                                   sg_win, sg_poly, delay_grid,
                                                   l2_dyn=5e-3, kdelta_min=-2.0, log_prefix=log_prefix)
    return dict(model=name, dynamics=best_dyn, logs=dyn_logs)

# ----------------- NARX専用: 共通npz保存 -----------------
def save_narx_common_meta_npz(out_path_npz, *, dt, narx_meta, z_coef, z_feat_names, z_metrics=None, theta_unit="deg"):
    z_rmse_va = np.array(z_metrics["rmse_va"]) if (z_metrics and "rmse_va" in z_metrics) else np.array(np.nan)
    z_mae_va  = np.array(z_metrics["mae_va"])  if (z_metrics and "mae_va"  in z_metrics) else np.array(np.nan)
    z_r2_va   = np.array(z_metrics["r2_va"])   if (z_metrics and "r2_va"   in z_metrics) else np.array(np.nan)

    poly_feature_names = narx_meta.get("poly_feature_names", None)
    if poly_feature_names is None:
        poly_feature_names = []

    np.savez(out_path_npz,
             dt=np.array(dt, dtype=np.float64),
             model=np.array("NARX"),
             degree=np.array(narx_meta["degree"], dtype=np.int64),
             lag_y=np.array(narx_meta["lag_y"], dtype=np.int64),
             lag_u=np.array(narx_meta["lag_u"], dtype=np.int64),
             delay=np.array(narx_meta["delay"], dtype=np.int64),
             coef=np.array(narx_meta["coef"], dtype=np.float64),
             theta_unit=np.array(theta_unit),
             poly_feature_names=np.array(poly_feature_names),
             z_coef=np.array(z_coef, dtype=np.float64),
             z_feat_names=np.array(z_feat_names),
             z_rmse_va=z_rmse_va,
             z_mae_va=z_mae_va,
             z_r2_va=z_r2_va)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    # original single-csv
    ap.add_argument("--csv", default=None, help="(legacy) single CSV path")
    # new: multi-session
    ap.add_argument("--dyn_csvs", nargs="*", default=[], help="Dynamic session CSVs")
    ap.add_argument("--stat_csvs", nargs="*", default=[], help="Static session CSVs (optional)")

    ap.add_argument("--col-sum", default="p_sum[MPa]")
    ap.add_argument("--col-diff", default="p_diff[MPa]")
    ap.add_argument("--col-theta", default="theta[deg]")
    ap.add_argument("--col-time", default="time[s]")
    ap.add_argument("--dt", type=float, default=None)  # will be used only for legacy single-csv path override

    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sg-win", type=int, default=17)
    ap.add_argument("--sg-poly", type=int, default=3)
    ap.add_argument("--delay-grid", type=str, default="0,1,2,3,4,5")

    ap.add_argument("--enable-all", action="store_true")
    ap.add_argument("--enable-hw-poly", action="store_true")
    ap.add_argument("--enable-hw-rbf", action="store_true")
    ap.add_argument("--enable-pwa", action="store_true")
    ap.add_argument("--enable-gp", action="store_true")
    ap.add_argument("--enable-mono-nn", action="store_true")
    ap.add_argument("--enable-arx", action="store_true")
    ap.add_argument("--enable-narx", action="store_true")
    ap.add_argument("--enable-nlss", action="store_true")
    ap.add_argument("--enable-koopman", action="store_true")

    ap.add_argument("--poly-degree", type=int, default=3)
    ap.add_argument("--rbf-centers", type=int, default=40)
    ap.add_argument("--rbf-length", type=float, default=0.6)
    ap.add_argument("--pwa-bins", type=int, default=8)
    ap.add_argument("--ridge-alpha", type=float, default=1e-3)
    ap.add_argument("--gp-maxn", type=int, default=8000)

    ap.add_argument("--mono-nn-M", type=int, default=6)
    ap.add_argument("--mono-nn-hidden", type=int, default=32)
    ap.add_argument("--mono-nn-epochs", type=int, default=200)
    ap.add_argument("--mono-nn-lr", type=float, default=1e-3)
    ap.add_argument("--mono-nn-batch", type=int, default=512)
    ap.add_argument("--mono-nn-learn-centers", action="store_true")

    ap.add_argument("--arx-lag", type=int, default=3)
    ap.add_argument("--narx-lag", type=int, default=2)
    ap.add_argument("--narx-degree", type=int, default=2)

    ap.add_argument("--nlss-epochs", type=int, default=60)
    ap.add_argument("--nlss-hidden", type=int, default=32)
    ap.add_argument("--koopman-degree", type=int, default=2)

    # ----- z(ps,pd) 学習/設計オプション -----
    ap.add_argument("--z-mode", choices=["design","learn"], default="design",
                    help="design: 係数を設計 / learn: CSVの z列を回帰")
    ap.add_argument("--z-col", type=str, default=None, help="learn時に使う z の列名")
    ap.add_argument("--z-feat", choices=["quad","lin"], default="quad",
                    help="z特徴: quad=[1,ps,pd,ps*pd,ps^2,pd^2], lin=[1,ps,pd,ps*pd]")
    ap.add_argument("--z-w-sigma2", type=float, default=1.0, help="design用 w_{Sigma^2}")
    ap.add_argument("--z-w-delta2", type=float, default=1.0, help="design用 w_{Delta^2}")
    ap.add_argument("--z-w-cross", type=float, default=0.0, help="design用 w_{ps*pd}")
    ap.add_argument("--z-source", choices=["dyn","stat","all"], default="dyn",
                    help="learn時: zデータを動的/静的/両方から収集")
    ap.add_argument("--z-qstatic-from-stat", action="store_true",
                    help="--z-source=stat/all のとき、静的CSVから準静的サンプル(微分閾値)のみ学習に使用")
    ap.add_argument("--z-ss-theta-eps", type=float, default=1.0*np.pi/180.0,
                    help="準静的抽出: |dθ/dt| [rad/s]")
    ap.add_argument("--z-ss-dz-eps", type=float, default=2e-3,
                    help="準静的抽出: |d(dz)/dt| [m/s]")
    ap.add_argument("--z-dz-col", type=str, default="dz[m]", help="準静的抽出で使うdz列名")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        # ---- load data (legacy single-csv or multi-csv) ----
        if args.csv and (args.dyn_csvs or args.stat_csvs):
            raise RuntimeError("Use either --csv or (--dyn_csvs [--stat_csvs]), not both.")
        if not args.csv and not args.dyn_csvs:
            raise RuntimeError("Provide --csv or at least one --dyn_csvs.")

        if args.csv:
            # legacy path
            print("[load] CSV:", args.csv)
            df, dt_single = _load_single_csv(args.csv, args.col_sum, args.col_diff, args.col_theta, args.col_time)
            ps = df[args.col_sum].to_numpy(float)
            pdv= df[args.col_diff].to_numpy(float)
            th = df[args.col_theta].to_numpy(float)
            dt = float(args.dt) if args.dt is not None else dt_single
            sessions = [dict(ps=ps, pd=pdv, th=th, dt=dt, name=os.path.basename(args.csv))]
            print(f"[info] legacy single-csv: N={len(ps)}, dt≈{dt:.5f}s, val_ratio={args.val_ratio}")
        else:
            # multi-session path
            print("[load] Dynamic CSVs:", len(args.dyn_csvs))
            sessions, ps_all, pd_all, th_all, dt_rep = load_dynamic_csvs(
                args.dyn_csvs, args.col_sum, args.col_diff, args.col_theta, args.col_time
            )
            ps = ps_all; pdv = pd_all; th = th_all; dt = dt_rep
            print(f"[info] dyn sessions={len(sessions)}, concat N={len(ps)}, dt_rep≈{dt:.5f}s, val_ratio={args.val_ratio}")
            if args.stat_csvs:
                print("[load] Static CSVs:", len(args.stat_csvs))

        # ---- z 観測を準備（learn時） ----
        z_obs = None; z_ps=None; z_pd=None
        if args.z_mode == "learn":
            if args.z_col is None:
                raise RuntimeError("--z-mode learn には --z-col=<列名> が必要です")
            # from dynamic
            if args.csv or (args.z_source in ["dyn","all"]):
                if args.csv:
                    z_d = df[args.z_col].to_numpy(float) if args.z_col in df.columns else None
                    if z_d is not None:
                        z_ps = ps if z_ps is None else np.concatenate([z_ps, ps])
                        z_pd = pdv if z_pd is None else np.concatenate([z_pd, pdv])
                        z_obs= z_d if z_obs is None else np.concatenate([z_obs, z_d])
                else:
                    # multi: gather from each dynamic csv
                    import pandas as pd
                    for p in args.dyn_csvs:
                        dfi = pd.read_csv(p, comment="#").dropna()
                        if args.z_col in dfi.columns:
                            z_ps = np.concatenate([z_ps, dfi[args.col_sum].to_numpy(float)]) if z_ps is not None else dfi[args.col_sum].to_numpy(float)
                            z_pd = np.concatenate([z_pd, dfi[args.col_diff].to_numpy(float)]) if z_pd is not None else dfi[args.col_diff].to_numpy(float)
                            z_obs= np.concatenate([z_obs, dfi[args.z_col].to_numpy(float)]) if z_obs is not None else dfi[args.z_col].to_numpy(float)
            # from static
            if (args.z_source in ["stat","all"]) and args.stat_csvs:
                s_ps, s_pd, s_z = load_static_csvs(args.stat_csvs, args.z_col,
                                                   args.col_sum, args.col_diff, args.col_theta, args.col_time,
                                                   use_quasi_static=args.z_qstatic_from_stat,
                                                   ss_theta_eps=args.z_ss_theta_eps,
                                                   ss_z_eps=args.z_ss_dz_eps,
                                                   dz_col=args.z_dz_col)
                if s_ps is not None:
                    z_ps = np.concatenate([z_ps, s_ps]) if z_ps is not None else s_ps
                    z_pd = np.concatenate([z_pd, s_pd]) if z_pd is not None else s_pd
                    z_obs= np.concatenate([z_obs, s_z]) if z_obs is not None else s_z
            if z_obs is None:
                raise RuntimeError("z-mode=learn だが、zデータが見つかりませんでした。--z-col と --z-source を確認してください。")

        # ---- standardize (from dynamic data) & split ----
        muS, sdS, muD, sdD, Sh, Dh = standardize_cols(ps, pdv)
        N = len(ps)
        tr, va = train_val_split(N, args.val_ratio)
        Sh_tr, Dh_tr, th_tr = Sh[tr], Dh[tr], th[tr]

        leaderboard=[]; results={}

        def add_result(name, rmse, extra=None):
            item = dict(model=name, val_rmse_deg=float(rmse))
            if extra: item.update(extra)
            leaderboard.append(item); results[name]=item

        if args.enable_all:
            flags = dict(hw_poly=True, hw_rbf=True, pwa=True, gp=True, mono_nn=True,
                         arx=True, narx=True, nlss=True, koopman=True)
        else:
            flags = dict(hw_poly=args.enable_hw_poly, hw_rbf=args.enable_hw_rbf, pwa=args.enable_pwa,
                         gp=args.enable_gp, mono_nn=args.enable_mono_nn, arx=args.enable_arx,
                         narx=args.enable_narx, nlss=args.enable_nlss, koopman=args.enable_koopman)

        # helper: finalize static model (save z & common meta)
        def finalize_static_model(tag_name, pack_dict):
            # z 係数 + メトリクス
            if args.z_mode == "learn":
                z_coef, z_feat_names, z_metrics = fit_or_design_z(
                    z_ps, z_pd, z_obs=z_obs, feat_mode=args.z_feat,
                    w_sigma2=args.z_w_sigma2, w_delta2=args.z_w_delta2, w_cross=args.z_w_cross,
                    lam=1e-6, val_ratio=args.val_ratio
                )
            else:
                z_coef, z_feat_names, z_metrics = fit_or_design_z(
                    ps, pdv, z_obs=None, feat_mode=args.z_feat,
                    w_sigma2=args.z_w_sigma2, w_delta2=args.z_w_delta2, w_cross=args.z_w_cross,
                    lam=1e-6, val_ratio=args.val_ratio
                )
            if z_metrics is not None:
                print(f"[{tag_name}] z-learn metrics: RMSE_va={z_metrics['rmse_va']:.4f}, "
                      f"MAE_va={z_metrics['mae_va']:.4f}, R2_va={z_metrics['r2_va']:.3f}")

            # JSON
            meta_json = dict(pack_dict)
            meta_json["z_coef"] = np.asarray(z_coef, float).tolist()
            meta_json["z_feat_names"] = z_feat_names
            meta_json["z_mode"] = args.z_mode
            meta_json["z_feat"] = args.z_feat
            if z_metrics is not None:
                meta_json["z_metrics"] = z_metrics
            save_json(os.path.join(args.out_dir, f"{tag_name}_meta.json"), meta_json)

            # 共通 NPZ（dtは代表値：singleならdt、multiならdt_rep）
            out_npz = os.path.join(args.out_dir, f"{tag_name}_common_meta.npz")
            save_common_meta_npz(out_npz, dt, muS, sdS, muD, sdD,
                                 pack_dict["dynamics"], z_coef, z_feat_names, z_metrics=z_metrics)
            print(f"[{tag_name}] saved z + common meta -> {out_npz}")

        # ---- Static + shared dynamics ----
        # static fit uses concatenated dynamic data (Sh_tr, Dh_tr, th_tr)
        if flags["hw_poly"] and SKLEARN_OK:
            print("\n[HW_Poly] fit static...")
            mdl = StaticPoly(deg=args.poly_degree, alpha=args.ridge_alpha).fit(Sh_tr, Dh_tr, th_tr)
            print("[HW_Poly] fit dynamics + delay search (multi-session)...")
            pack = eval_static_plus_dyn_multi("HW_Poly", mdl, sessions,
                                              args.sg_win, args.sg_poly, args.delay_grid,
                                              muS, sdS, muD, sdD, log_prefix="[HW_Poly]")
            add_result("HW_Poly", pack["dynamics"]["rmse"], extra=pack)
            finalize_static_model("hw_poly", pack)

        if flags["hw_rbf"] and SKLEARN_OK:
            print("\n[HW_RBF] fit static...")
            mdl = StaticRBF(n_centers=args.rbf_centers, length=args.rbf_length, alpha=args.ridge_alpha).fit(Sh_tr, Dh_tr, th_tr)
            print("[HW_RBF] fit dynamics + delay search (multi-session)...")
            pack = eval_static_plus_dyn_multi("HW_RBF", mdl, sessions,
                                              args.sg_win, args.sg_poly, args.delay_grid,
                                              muS, sdS, muD, sdD, log_prefix="[HW_RBF]")
            add_result("HW_RBF", pack["dynamics"]["rmse"], extra=pack)
            finalize_static_model("hw_rbf", pack)

        if flags["pwa"]:
            print("\n[PWA] fit static...")
            mdl = StaticPWA(n_bins=args.pwa_bins, alpha=args.ridge_alpha).fit(Sh_tr, Dh_tr, th_tr)
            print("[PWA] fit dynamics + delay search (multi-session)...")
            pack = eval_static_plus_dyn_multi("PWA", mdl, sessions,
                                              args.sg_win, args.sg_poly, args.delay_grid,
                                              muS, sdS, muD, sdD, log_prefix="[PWA]")
            add_result("PWA", pack["dynamics"]["rmse"], extra=pack)
            finalize_static_model("pwa", pack)

        if flags["gp"] and SKLEARN_GP_OK:
            print("\n[GP] fit static...")
            try:
                mdl = StaticGP(gp_maxn=args.gp_maxn).fit(Sh_tr, Dh_tr, th_tr)
                print("[GP] fit dynamics + delay search (multi-session)...")
                pack = eval_static_plus_dyn_multi("GP", mdl, sessions,
                                                  args.sg_win, args.sg_poly, args.delay_grid,
                                                  muS, sdS, muD, sdD, log_prefix="[GP]")
                add_result("GP", pack["dynamics"]["rmse"], extra=pack)
                # kernel情報をJSONにも残す
                pack_with_kernel = dict(pack); pack_with_kernel["kernel"] = str(mdl.gp.kernel_)
                save_json(os.path.join(args.out_dir, "gp_meta.json"), pack_with_kernel)
                finalize_static_model("gp", pack)
            except Exception as e:
                print("[GP] ERROR:", e)
                add_result("GP", float("inf"), extra=dict(error=str(e)))

        if flags["mono_nn"]:
            print("\n[MonoNN] fit static (torch)...")
            mdl = StaticMonoNN(M=args.mono_nn_M, hidden=args.mono_nn_hidden, epochs=args.mono_nn_epochs,
                               lr=args.mono_nn_lr, batch=args.mono_nn_batch,
                               learn_centers=args.mono_nn_learn_centers).fit(Sh_tr, Dh_tr, th_tr)
            print("[MonoNN] fit dynamics + delay search (multi-session)...")
            pack = eval_static_plus_dyn_multi("MonoNN", mdl, sessions,
                                              args.sg_win, args.sg_poly, args.delay_grid,
                                              muS, sdS, muD, sdD, log_prefix="[MonoNN]")
            add_result("MonoNN", pack["dynamics"]["rmse"], extra=pack)
            torch.save(mdl.state_dict, os.path.join(args.out_dir, "mononn_state.pt"))
            meta_json = dict(M=args.mono_nn_M, hidden=args.mono_nn_hidden,
                             learn_centers=args.mono_nn_learn_centers, dyn=pack["dynamics"])
            save_json(os.path.join(args.out_dir, "mononn_meta.json"), meta_json)
            finalize_static_model("mononn", pack)

        # ---- Sequence models (multi-session) ----
        if flags["arx"] and SKLEARN_OK:
            print("\n[ARX] grid over delays (multi-session)...")
            best=None
            for d in [int(x) for x in args.delay_grid.split(",") if x.strip()!=""]:
                rmse, meta = arx_fit_predict_multi(sessions, lag_y=args.arx_lag, lag_u=args.arx_lag,
                                                   delay=d, alpha=args.ridge_alpha, log_prefix="[ARX]")
                if best is None or rmse < best[0]: best=(rmse, meta)
            add_result("ARX", best[0], extra=best[1])
            save_json(os.path.join(args.out_dir, "arx_meta.json"), best[1])

        if flags["narx"] and SKLEARN_OK:
            print("\n[NARX] grid over delays (multi-session)...")
            best=None; best_meta=None
            for d in [int(x) for x in args.delay_grid.split(",") if x.strip()!=""]:
                rmse, meta = narx_fit_predict_multi(sessions,
                                                    lag_y=args.narx_lag, lag_u=args.narx_lag,
                                                    delay=d, degree=args.narx_degree,
                                                    alpha=args.ridge_alpha, log_prefix="[NARX]")
                if best is None or rmse < best:
                    best = rmse; best_meta = meta
            add_result("NARX", best, extra=best_meta)
            save_json(os.path.join(args.out_dir, "narx_meta.json"), dict(best_meta))

            # z 同梱 npz（θ=NARX + z の係数）
            if args.z_mode == "learn":
                z_coef, z_feat_names, z_metrics = fit_or_design_z(
                    z_ps, z_pd, z_obs=z_obs, feat_mode=args.z_feat,
                    w_sigma2=args.z_w_sigma2, w_delta2=args.z_w_delta2, w_cross=args.z_w_cross,
                    lam=1e-6, val_ratio=args.val_ratio
                )
            else:
                z_coef, z_feat_names, z_metrics = fit_or_design_z(
                    ps, pdv, z_obs=None, feat_mode=args.z_feat,
                    w_sigma2=args.z_w_sigma2, w_delta2=args.z_w_delta2, w_cross=args.z_w_cross,
                    lam=1e-6, val_ratio=args.val_ratio
                )
            if z_metrics is not None:
                print(f"[NARX] z-learn metrics: RMSE_va={z_metrics['rmse_va']:.4f}, "
                      f"MAE_va={z_metrics['mae_va']:.4f}, R2_va={z_metrics['r2_va']:.3f}")

            narx_npz = os.path.join(args.out_dir, "narx_common_meta.npz")
            theta_unit = "deg"  # 列は deg 前提
            save_narx_common_meta_npz(narx_npz, dt=dt, narx_meta=best_meta,
                                      z_coef=z_coef, z_feat_names=z_feat_names,
                                      z_metrics=z_metrics, theta_unit=theta_unit)
            print(f"[NARX] saved common meta (θ NARX + z) -> {narx_npz}")

        if flags["nlss"]:
            print("\n[NLSS_GRU] train (multi-session)...")
            rmse, meta = nlss_gru_fit_rollout_multi(sessions, epochs=args.nlss_epochs, hidden=args.nlss_hidden,
                                                    log_prefix="[NLSS]")
            add_result("NLSS_GRU", rmse, extra=meta)
            save_json(os.path.join(args.out_dir, "nlss_meta.json"), meta)

        if flags["koopman"]:
            print("\n[EDMD] fit & rollout (multi-session)...")
            rmse, meta = koopman_edmd_fit_rollout_multi(sessions, degree=args.koopman_degree, log_prefix="[EDMD]")
            add_result("Koopman_EDMD", rmse, extra=meta)
            save_json(os.path.join(args.out_dir, "koopman_meta.json"), meta)

        # ---- summary ----
        leaderboard_sorted = sorted(leaderboard, key=lambda x: x["val_rmse_deg"])
        summary = dict(results=leaderboard_sorted,
                       N=len(ps), dt=dt, val_ratio=args.val_ratio,
                       n_sessions=len(sessions),
                       dyn_csvs=args.dyn_csvs if args.dyn_csvs else ([args.csv] if args.csv else []),
                       stat_csvs=args.stat_csvs)
        save_json(os.path.join(args.out_dir, "summary.json"), summary)
        print("\n=== Leaderboard (val RMSE [deg], mean over sessions where applicable) ===")
        for it in leaderboard_sorted:
            print(f"{it['model']:>14s} : {it['val_rmse_deg']:.3f}")
        print(f"[OK] saved to {args.out_dir}/summary.json")

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Writing partial summary...")
        try:
            leaderboard_sorted = sorted(leaderboard, key=lambda x: x["val_rmse_deg"])
            summary = dict(results=leaderboard_sorted)
            save_json(os.path.join(args.out_dir, "summary_partial.json"), summary)
            print("[OK] partial saved.")
        except Exception:
            pass

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
