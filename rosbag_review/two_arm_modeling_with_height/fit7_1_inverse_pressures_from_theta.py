#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse mapping benchmark: (p_sum, p_diff) = f(theta[, z])

- Static: HW_Poly / HW_RBF / PWA / GP / MonoNN
- Sequence: ARX / NARX / NLSS(GRU) / Koopman(EDMD)
- Feasibility projection: ensure p_sum >= |p_diff| and 0<=p1,p2<=pmax

Inputs CSV (columns default):
  theta[deg], p_sum[MPa], p_diff[MPa], (optional) z[m], time[s]

Outputs:
  out-dir/summary.json
  out-dir/{poly|rbf|pwa|gp|mononn|arx|narx|nlss|edmd}_meta.json
  out-dir/{poly|rbf|pwa|gp|mononn|arx|narx|nlss|edmd}_common_meta.npz
  python fit7_inverse_pressures_from_theta.py \
  --csv out/dynamic_prbs_data.csv \
  --col-theta "theta[deg]" \
  --col-sum "p_sum[MPa]" \
  --col-diff "p_diff[MPa]" \
  --col-z "z[m]" \
  --out-dir out_inv \
  --pmax 0.70 \
  --enable-all \
  --poly-degree 3 --rbf-centers 40 --rbf-length 0.6 \
  --arx-lag 3 --arx-degree 1 \
  --narx-lag 2 --narx-degree 2 \
  --nlss-epochs 80 --koopman-degree 2 \
  --mononn-epochs 250 --mononn-lam-feas 0.1
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
def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def train_val_split(N, val_ratio):
    val_n = max(1, int(N*val_ratio))
    tr = np.arange(0, N-val_n)
    va = np.arange(N-val_n, N)
    return tr, va

def standardize_cols(*cols):
    mus=[]; sds=[]; outs=[]
    for c in cols:
        mu=float(np.mean(c)); sd=float(np.std(c)+1e-8)
        mus.append(mu); sds.append(sd)
        outs.append((c-mu)/sd)
    return mus, sds, outs

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

def to_p1p2(ps, pd):
    p1 = 0.5*(ps + pd)
    p2 = 0.5*(ps - pd)
    return p1, p2

def from_p1p2(p1, p2):
    ps = p1 + p2
    pd = p1 - p2
    return ps, pd

def project_feasible(ps, pd, pmax):
    """
    Enforce p_sum >= |p_diff| and 0<=p1,p2<=pmax by minimal projection.
    """
    ps = np.asarray(ps, float).copy()
    pd = np.asarray(pd, float).copy()

    # ensure p_sum >= |p_diff|
    mask = (np.abs(pd) > ps)
    if np.any(mask):
        ps[mask] = np.abs(pd[mask])

    # map to p1,p2 and clip
    p1, p2 = to_p1p2(ps, pd)
    p1 = np.clip(p1, 0.0, pmax)
    p2 = np.clip(p2, 0.0, pmax)
    # back to (ps,pd) after clipping
    ps2, pd2 = from_p1p2(p1, p2)
    return ps2, pd2, p1, p2

def reg_metrics_2d(y_true, y_pred, names=("ps","pd")):
    m={}
    for i,k in enumerate(names):
        err = y_pred[:,i]-y_true[:,i]
        m[f"rmse_{k}"] = float(np.sqrt(np.mean(err**2)))
        m[f"mae_{k}"]  = float(np.mean(np.abs(err)))
    # p1,p2 side
    t_p1, t_p2 = to_p1p2(y_true[:,0], y_true[:,1])
    h_p1, h_p2 = to_p1p2(y_pred[:,0], y_pred[:,1])
    m["rmse_p1"] = float(np.sqrt(np.mean((h_p1-t_p1)**2)))
    m["rmse_p2"] = float(np.sqrt(np.mean((h_p2-t_p2)**2)))
    return m


# ----------------- Static inverse models -----------------
class InvModel:
    def fit(self, X, Y): ...
    def predict(self, X): ...

class InvPoly(InvModel):
    def __init__(self, deg=3, alpha=1e-3, include_z=True):
        if not SKLEARN_OK: raise RuntimeError("sklearn needed for InvPoly.")
        self.deg=deg; self.alpha=alpha; self.include_z=include_z
        self.poly = PolynomialFeatures(degree=deg, include_bias=True)
        self.ridge = Ridge(alpha=alpha, fit_intercept=False)
    def fit(self, X, Y):
        t0=time.time()
        XP = self.poly.fit_transform(X)
        self.ridge.fit(XP, Y)
        print(f"    [fit InvPoly] {len(Y)} samples in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        XP = self.poly.transform(X)
        return self.ridge.predict(XP)

class InvRBF(InvModel):
    def __init__(self, n_centers=40, length=0.6, alpha=1e-3, kmeans=True, seed=0):
        if not SKLEARN_OK: raise RuntimeError("sklearn needed for InvRBF.")
        self.n_centers=n_centers; self.length=float(length); self.alpha=float(alpha)
        self.kmeans=bool(kmeans); self.seed=int(seed)
    def _features(self, X, C):
        diff = X[:,None,:] - C[None,:,:]
        r2 = np.sum(diff*diff, axis=2)
        Phi = np.exp(-0.5*r2/(self.length**2))
        Phi = np.hstack([np.ones((Phi.shape[0],1)), Phi])
        return Phi
    def fit(self, X, Y):
        t0=time.time()
        if self.kmeans:
            km = KMeans(n_clusters=self.n_centers, random_state=self.seed).fit(X)
            C = km.cluster_centers_
        else:
            # grid on theta,z
            th = X[:,0]; z = X[:,1] if X.shape[1]>1 else np.zeros_like(th)
            gs = int(np.sqrt(self.n_centers))
            ths = np.linspace(np.quantile(th,0.02), np.quantile(th,0.98), gs)
            zs  = np.linspace(np.quantile(z, 0.02), np.quantile(z, 0.98), gs)
            C = np.array([(a,b) for a in ths for b in zs], float)[:self.n_centers]
        Phi = self._features(X, C)
        # multioutput ridge
        A = Phi.T@Phi + self.alpha*np.eye(Phi.shape[1])
        self.W = LA.solve(A, Phi.T@Y)
        self.C = C
        print(f"    [fit InvRBF] {len(Y)} samples, centers={len(C)} in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        Phi = self._features(X, self.C)
        return Phi@self.W

class InvPWA(InvModel):
    def __init__(self, n_bins=10, alpha=1e-3):
        self.n_bins=int(n_bins); self.alpha=float(alpha)
    def fit(self, X, Y):
        # bin on theta (col0)
        t0=time.time()
        th = X[:,0]
        edges = np.quantile(th, np.linspace(0,1,self.n_bins+1))
        self.edges = edges; self.W=[]
        for bi in range(self.n_bins):
            lo, hi = edges[bi], edges[bi+1]
            idx = np.where((th>=lo)&(th<=hi))[0]
            if len(idx) < 10:
                self.W.append(np.zeros((X.shape[1]+1,2))); continue
            Xb = np.column_stack([np.ones(len(idx)), X[idx]])
            Yb = Y[idx]
            A = Xb.T@Xb + self.alpha*np.eye(Xb.shape[1]); B = Xb.T@Yb
            self.W.append(LA.solve(A,B))
        print(f"    [fit InvPWA] {len(Y)} samples, bins={self.n_bins} in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        th = X[:,0]; Y=np.zeros((len(X),2))
        for i in range(len(X)):
            bi = min(self.n_bins-1, max(0, int(np.searchsorted(self.edges, th[i])-1)))
            W = self.W[bi]
            Y[i] = (np.hstack([1.0, X[i]]) @ W).ravel()
        return Y

class InvGP(InvModel):
    def __init__(self, gp_maxn=8000, seed=0):
        if not SKLEARN_GP_OK: raise RuntimeError("sklearn GP needed for InvGP.")
        self.gp_maxn=int(gp_maxn); self.seed=int(seed)
    def fit(self, X, Y):
        t0=time.time()
        if len(X) > self.gp_maxn:
            rs = np.random.RandomState(self.seed)
            idx = rs.choice(len(X), self.gp_maxn, replace=False)
            X, Y = X[idx], Y[idx]
            print(f"    [GP] subsampled to {len(X)}")
        kernel = ConstantKernel(1.0, (1e-2,1e2)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2,1e2)) \
                 + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6,1e-1))
        self.gp_ps = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=2, normalize_y=True, random_state=self.seed)
        self.gp_pd = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=2, normalize_y=True, random_state=self.seed+1)
        self.gp_ps.fit(X, Y[:,0]); self.gp_pd.fit(X, Y[:,1])
        print(f"    [fit InvGP] {len(X)} samples in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        ps = self.gp_ps.predict(X)
        pd = self.gp_pd.predict(X)
        return np.column_stack([ps,pd])

# --- MonoNN: single-input monotone-ish head for pd (optional), general for ps ---
class MonoHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.out = nn.Linear(hidden, 2)
    def forward(self, x):
        return self.out(self.enc(x))

class InvMonoNN(InvModel):
    def __init__(self, epochs=200, lr=1e-3, batch=512, seed=0, lam_feas=0.0, pmax=0.7):
        self.epochs=epochs; self.lr=lr; self.batch=batch; self.seed=seed
        self.lam_feas=float(lam_feas); self.pmax=float(pmax)
        self.model=None
    def fit(self, X, Y):
        t0=time.time()
        torch.manual_seed(self.seed)
        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                            torch.tensor(Y, dtype=torch.float32))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True, drop_last=False)
        model = MonoHead(hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)
        loss_fn = nn.HuberLoss(delta=3.0)
        model.train()
        for ep in range(self.epochs):
            for xb,yb in dl:
                opt.zero_grad()
                yh = model(xb)
                loss = loss_fn(yh, yb)
                if self.lam_feas>0:
                    # soft feasibility penalty on batch
                    ps, pd = yh[:,0], yh[:,1]
                    p1 = 0.5*(ps+pd); p2 = 0.5*(ps-pd)
                    feas = torch.relu(torch.abs(pd)-ps) \
                         + torch.relu(-p1) + torch.relu(-p2) \
                         + torch.relu(p1-self.pmax) + torch.relu(p2-self.pmax)
                    loss = loss + self.lam_feas*feas.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            if (ep+1)%50==0:
                print(f"      [MonoNN] epoch {ep+1}/{self.epochs}")
        self.model=model.eval()
        print(f"    [fit InvMonoNN] {len(Y)} samples in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            Y = self.model(X).cpu().numpy()
        return Y


# ----------------- Sequence (inverse) -----------------
def build_lag_matrix_inverse(theta, z=None, lag_x=3, degree=1, alpha=1e-3):
    """
    Build features from past theta (and z) only, to predict current [ps,pd].
    """
    N=len(theta); T0=lag_x
    rows=[]; ys=[]
    for t in range(T0, N):
        feats=[]
        for k in range(1, lag_x+1): feats.append(theta[t-k])
        if z is not None:
            for k in range(1, lag_x+1): feats.append(z[t-k])
        rows.append(feats); ys.append([0.0,0.0])  # placeholder, y will be filled outside
    X=np.array(rows,float); y=np.array(ys,float); return X,y,T0

def arx_inverse_fit_predict(theta, ps, pd, z=None, lag=3, alpha=1e-3, degree=1, log_prefix=""):
    if not SKLEARN_OK: raise RuntimeError("sklearn needed for ARX inverse.")
    t0=time.time()
    X,y,T0 = build_lag_matrix_inverse(theta, z=z, lag_x=lag, degree=degree, alpha=alpha)
    Y = np.column_stack([ps,pd])[T0:]
    if degree>1:
        poly = PolynomialFeatures(degree=degree, include_bias=True); XP = poly.fit_transform(X)
    else:
        poly=None; XP = X
    mdl = Ridge(alpha=alpha, fit_intercept=True).fit(XP, Y)
    # rollout = single pass (pure exogenous inputs)
    if poly is not None: XP2=poly.transform(X)
    else: XP2=XP
    Yhat = mdl.predict(XP2)
    rm = reg_metrics_2d(Y, Yhat)
    print(f"{log_prefix}  [Inv-ARX] lag={lag}, deg={degree}: RMSE_ps={rm['rmse_ps']:.3f}, RMSE_pd={rm['rmse_pd']:.3f} in {time.time()-t0:.2f}s")
    meta = dict(model="Inv_ARX", lag=lag, degree=degree, coef=mdl.coef_.tolist(), intercept=mdl.intercept_.tolist())
    return rm, meta, T0

def narx_inverse_fit_predict(theta, ps, pd, z=None, lag=2, degree=2, alpha=1e-3, log_prefix=""):
    # identical to ARX above but always polynomial-expanded
    return arx_inverse_fit_predict(theta, ps, pd, z=z, lag=lag, degree=degree, alpha=alpha, log_prefix=log_prefix)

class GRU_Inv(nn.Module):
    def __init__(self, in_dim=1, hidden=32, out_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, u, h0=None):
        y, h = self.gru(u, h0)
        y = self.head(y)
        return y, h

def nlss_gru_inverse_fit_rollout(theta, ps, pd, z=None, epochs=60, hidden=32, lr=1e-3, seed=0, log_prefix=""):
    torch.manual_seed(seed)
    t0=time.time()
    if z is None:
        u = theta.astype(np.float32)[:,None]  # [T,1]
    else:
        u = np.stack([theta, z], axis=1).astype(np.float32)
    y = np.stack([ps,pd], axis=1).astype(np.float32)
    model = GRU_Inv(in_dim=u.shape[1], hidden=hidden, out_dim=2)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ut = torch.tensor(u[None,...])     # [1,T,in]
    yt = torch.tensor(y[None,...])     # [1,T,2]
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        yhat,_ = model(ut)
        loss = loss_fn(yhat, yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (ep+1)%10==0:
            print(f"{log_prefix}  [Inv-NLSS_GRU] epoch {ep+1}/{epochs}, loss={float(loss):.4f}")
    model.eval()
    with torch.no_grad():
        yhat,_ = model(ut)
        YH = yhat.squeeze(0).numpy()
    rm = reg_metrics_2d(y, YH)
    print(f"{log_prefix}  [Inv-NLSS_GRU] done in {time.time()-t0:.2f}s, RMSE_ps={rm['rmse_ps']:.3f}, RMSE_pd={rm['rmse_pd']:.3f}")
    return rm, dict(model="Inv_NLSS_GRU", hidden=hidden, epochs=epochs)

def koopman_edmd_inverse_fit_rollout(theta, ps, pd, degree=2, z=None, log_prefix=""):
    t0=time.time()
    if z is None:
        XU = theta[:,None]
    else:
        XU = np.column_stack([theta, z])
    # build poly dict
    if SKLEARN_OK:
        pf = PolynomialFeatures(degree=degree, include_bias=True)
        Z = pf.fit_transform(XU)
    else:
        Z = np.hstack([np.ones((len(XU),1)), XU])
        if degree>=2:
            q=[]
            for i in range(XU.shape[1]):
                for j in range(i, XU.shape[1]):
                    q.append((XU[:,i:i+1]*XU[:,j:j+1]))
            Z = np.hstack([Z]+q)
    Zp = Z[1:]; Zk = Z[:-1]
    K = LA.lstsq(Zk, Zp, rcond=None)[0]
    W = LA.lstsq(Z, np.column_stack([ps,pd]), rcond=None)[0]  # decoder
    z = Z[0].copy(); Yhat=np.zeros((len(theta),2)); Yhat[0]=[ps[0],pd[0]]
    for k in range(1, len(theta)):
        z = z @ K
        Yhat[k] = (z @ W)
    T0=5
    rm = reg_metrics_2d(np.column_stack([ps,pd])[T0:], Yhat[T0:])
    print(f"{log_prefix}  [Inv-EDMD] degree={degree}: RMSE_ps={rm['rmse_ps']:.3f}, RMSE_pd={rm['rmse_pd']:.3f} in {time.time()-t0:.2f}s")
    return rm, dict(model="Inv_Koopman_EDMD", degree=degree)


# ----------------- common meta save -----------------
def save_common_meta_npz(out_npz, *, dt, mu_th, sd_th, mu_z, sd_z, pmax,
                         model_tag, extra_meta=None):
    np.savez(out_npz,
             dt=np.array(dt, dtype=np.float64),
             mu_theta=np.array(mu_th if mu_th is not None else np.nan, dtype=np.float64),
             sd_theta=np.array(sd_th if sd_th is not None else np.nan, dtype=np.float64),
             mu_z=np.array(mu_z if mu_z is not None else np.nan, dtype=np.float64),
             sd_z=np.array(sd_z if sd_z is not None else np.nan, dtype=np.float64),
             pmax=np.array(pmax, dtype=np.float64),
             model_tag=np.array(model_tag),
             extra_meta=json.dumps(extra_meta if extra_meta else {}))


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col-theta", default="theta[deg]")
    ap.add_argument("--col-sum",   default="p_sum[MPa]")
    ap.add_argument("--col-diff",  default="p_diff[MPa]")
    ap.add_argument("--col-z",     default=None)
    ap.add_argument("--col-time",  default="time[s]")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--out-dir", required=True)

    # feasibility/limits
    ap.add_argument("--pmax", type=float, default=0.7, help="MPa")

    # smoothing
    ap.add_argument("--sg-win", type=int, default=17)
    ap.add_argument("--sg-poly", type=int, default=3)

    # enable flags
    ap.add_argument("--enable-all", action="store_true")
    ap.add_argument("--enable-poly", action="store_true")
    ap.add_argument("--enable-rbf", action="store_true")
    ap.add_argument("--enable-pwa", action="store_true")
    ap.add_argument("--enable-gp", action="store_true")
    ap.add_argument("--enable-mononn", action="store_true")
    ap.add_argument("--enable-arx", action="store_true")
    ap.add_argument("--enable-narx", action="store_true")
    ap.add_argument("--enable-nlss", action="store_true")
    ap.add_argument("--enable-koopman", action="store_true")

    # static hyperparams
    ap.add_argument("--poly-degree", type=int, default=3)
    ap.add_argument("--rbf-centers", type=int, default=40)
    ap.add_argument("--rbf-length", type=float, default=0.6)
    ap.add_argument("--pwa-bins", type=int, default=10)
    ap.add_argument("--ridge-alpha", type=float, default=1e-3)
    ap.add_argument("--gp-maxn", type=int, default=8000)

    # mononn
    ap.add_argument("--mononn-epochs", type=int, default=200)
    ap.add_argument("--mononn-lr", type=float, default=1e-3)
    ap.add_argument("--mononn-batch", type=int, default=512)
    ap.add_argument("--mononn-lam-feas", type=float, default=0.0)

    # sequence
    ap.add_argument("--arx-lag", type=int, default=3)
    ap.add_argument("--arx-degree", type=int, default=1)
    ap.add_argument("--narx-lag", type=int, default=2)
    ap.add_argument("--narx-degree", type=int, default=2)
    ap.add_argument("--nlss-epochs", type=int, default=60)
    ap.add_argument("--nlss-hidden", type=int, default=32)
    ap.add_argument("--koopman-degree", type=int, default=2)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load data ----
    import pandas as pd
    df = pd.read_csv(args.csv, comment="#")
    for c in [args.col_theta, args.col_sum, args.col_diff]:
        if c not in df.columns:
            raise RuntimeError(f"missing column: {c}")
    theta = df[args.col_theta].to_numpy(float)
    ps    = df[args.col_sum].to_numpy(float)
    pdv   = df[args.col_diff].to_numpy(float)
    z = None
    if args.col_z is not None:
        if args.col_z not in df.columns:
            raise RuntimeError(f"missing column: {args.col_z}")
        z = df[args.col_z].to_numpy(float)

    if args.dt is not None:
        dt = float(args.dt)
    elif args.col_time in df.columns:
        t = df[args.col_time].to_numpy(float)
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = 0.01
    print(f"[load] N={len(theta)}, dt≈{dt:.5f}s, val_ratio={args.val_ratio}")

    # optional smoothing just for metrics view (not mandatory)
    theta_s, _ = smooth_and_deriv(theta, dt, win=args.sg_win, poly=args.sg_poly)

    # standardize inputs
    if z is None:
        (mu_th,), (sd_th,), (Th,) = standardize_cols(theta_s)
        mu_z=sd_z=None
        X_full = Th[:,None]                # [N,1]
    else:
        (mu_th, mu_z), (sd_th, sd_z), (Th, Zh) = standardize_cols(theta_s, z)
        X_full = np.column_stack([Th, Zh]) # [N,2]

    Y_full = np.column_stack([ps, pdv])    # [N,2]
    N=len(theta); tr, va = train_val_split(N, args.val_ratio)
    X_tr, Y_tr = X_full[tr], Y_full[tr]
    X_va, Y_va = X_full[va], Y_full[va]

    leaderboard=[]
    def push_result(tag, metrics, extra=None):
        row = dict(model=tag, **metrics)
        if extra: row.update(extra)
        leaderboard.append(row)
        save_json(os.path.join(args.out_dir, f"{tag}_meta.json"), row)

        # also save a compact common npz for controllers
        save_common_meta_npz(
            os.path.join(args.out_dir, f"{tag}_common_meta.npz"),
            dt=dt, mu_th=mu_th, sd_th=sd_th, mu_z=mu_z, sd_z=sd_z, pmax=args.pmax,
            model_tag=tag, extra_meta={k: v for k,v in (extra or {}).items() if k not in metrics}
        )

    # flags
    flags = dict(poly=args.enable_poly, rbf=args.enable_rbf, pwa=args.enable_pwa,
                 gp=args.enable_gp, mononn=args.enable_mononn, arx=args.enable_arx,
                 narx=args.enable_narx, nlss=args.enable_nlss, koopman=args.enable_koopman)
    if args.enable_all:
        for k in flags: flags[k]=True

    # -------------- Static --------------
    if flags["poly"] and SKLEARN_OK:
        print("\n[Inv-Poly] fit ...")
        mdl = InvPoly(deg=args.poly_degree, alpha=args.ridge_alpha).fit(X_tr, Y_tr)
        Yv = mdl.predict(X_va)
        # feasibility projection (eval time)
        psf,pdf,_,_ = project_feasible(Yv[:,0], Yv[:,1], args.pmax)
        rm = reg_metrics_2d(Y_va, np.column_stack([psf,pdf]))
        print(f"  RMSE_ps={rm['rmse_ps']:.3f}, RMSE_pd={rm['rmse_pd']:.3f}, RMSE_p1={rm['rmse_p1']:.3f}, RMSE_p2={rm['rmse_p2']:.3f}")
        push_result("inv_poly", rm, extra=dict(degree=args.poly_degree, alpha=args.ridge_alpha))

    if flags["rbf"] and SKLEARN_OK:
        print("\n[Inv-RBF] fit ...")
        mdl = InvRBF(n_centers=args.rbf_centers, length=args.rbf_length, alpha=args.ridge_alpha).fit(X_tr, Y_tr)
        Yv = mdl.predict(X_va)
        psf,pdf,_,_ = project_feasible(Yv[:,0], Yv[:,1], args.pmax)
        rm = reg_metrics_2d(Y_va, np.column_stack([psf,pdf]))
        print(f"  RMSE_ps={rm['rmse_ps']:.3f}, RMSE_pd={rm['rmse_pd']:.3f}")
        push_result("inv_rbf", rm, extra=dict(n_centers=args.rbf_centers, length=args.rbf_length))

    if flags["pwa"]:
        print("\n[Inv-PWA] fit ...")
        mdl = InvPWA(n_bins=args.pwa_bins, alpha=args.ridge_alpha).fit(X_tr, Y_tr)
        Yv = mdl.predict(X_va)
        psf,pdf,_,_ = project_feasible(Yv[:,0], Yv[:,1], args.pmax)
        rm = reg_metrics_2d(Y_va, np.column_stack([psf,pdf]))
        print(f"  RMSE_ps={rm['rmse_ps']:.3f}, RMSE_pd={rm['rmse_pd']:.3f}")
        push_result("inv_pwa", rm, extra=dict(n_bins=args.pwa_bins))

    if flags["gp"] and SKLEARN_GP_OK:
        print("\n[Inv-GP] fit ...")
        try:
            mdl = InvGP(gp_maxn=args.gp_maxn).fit(X_tr, Y_tr)
            Yv = mdl.predict(X_va)
            psf,pdf,_,_ = project_feasible(Yv[:,0], Yv[:,1], args.pmax)
            rm = reg_metrics_2d(Y_va, np.column_stack([psf,pdf]))
            print(f"  RMSE_ps={rm['rmse_ps']:.3f}, RMSE_pd={rm['rmse_pd']:.3f}")
            push_result("inv_gp", rm, extra=dict(gp_maxn=args.gp_maxn))
        except Exception as e:
            print("[Inv-GP] ERROR:", e)
            push_result("inv_gp", dict(rmse_ps=float("inf"), rmse_pd=float("inf"),
                                       rmse_p1=float("inf"), rmse_p2=float("inf")),
                        extra=dict(error=str(e)))

    if flags["mononn"]:
        print("\n[Inv-MonoNN] fit ...")
        mdl = InvMonoNN(epochs=args.mononn_epochs, lr=args.mononn_lr,
                        batch=args.mononn_batch, lam_feas=args.mononn_lam_feas, pmax=args.pmax).fit(X_tr, Y_tr)
        Yv = mdl.predict(X_va)
        psf,pdf,_,_ = project_feasible(Yv[:,0], Yv[:,1], args.pmax)
        rm = reg_metrics_2d(Y_va, np.column_stack([psf,pdf]))
        print(f"  RMSE_ps={rm['rmse_ps']:.3f}, RMSE_pd={rm['rmse_pd']:.3f}")
        # save torch state
        torch.save(mdl.model.state_dict(), os.path.join(args.out_dir, "inv_mononn_state.pt"))
        push_result("inv_mononn", rm, extra=dict(epochs=args.mononn_epochs, lam_feas=args.mononn_lam_feas))

    # -------------- Sequence --------------
    if flags["arx"] and SKLEARN_OK:
        print("\n[Inv-ARX] fit ...")
        rm, meta, T0 = arx_inverse_fit_predict(theta, ps, pdv, z=z, lag=args.arx_lag, degree=args.arx_degree, alpha=args.ridge_alpha, log_prefix="[Inv-ARX]")
        push_result("inv_arx", rm, extra=meta)

    if flags["narx"] and SKLEARN_OK:
        print("\n[Inv-NARX] fit ...")
        rm, meta, T0 = narx_inverse_fit_predict(theta, ps, pdv, z=z, lag=args.narx_lag, degree=args.narx_degree, alpha=args.ridge_alpha, log_prefix="[Inv-NARX]")
        push_result("inv_narx", rm, extra=meta)

    if flags["nlss"]:
        print("\n[Inv-NLSS_GRU] fit ...")
        rm, meta = nlss_gru_inverse_fit_rollout(theta, ps, pdv, z=z, epochs=args.nlss_epochs, hidden=args.nlss_hidden, log_prefix="[Inv-NLSS]")
        push_result("inv_nlss_gru", rm, extra=meta)

    if flags["koopman"]:
        print("\n[Inv-EDMD] fit ...")
        rm, meta = koopman_edmd_inverse_fit_rollout(theta, ps, pdv, degree=args.koopman_degree, z=z, log_prefix="[Inv-EDMD]")
        push_result("inv_koopman_edmd", rm, extra=meta)

    # ---- summary ----
    # sort by sum of RMSEs on ps & pd
    def keysum(x): 
        return float(x.get("rmse_ps", np.inf)) + float(x.get("rmse_pd", np.inf))
    leaderboard_sorted = sorted(leaderboard, key=keysum)
    save_json(os.path.join(args.out_dir, "summary.json"),
              dict(results=leaderboard_sorted, N=N, dt=dt, val_ratio=args.val_ratio,
                   note="metrics include feasibility-projected predictions"))
    print("\n=== Leaderboard (RMSE_ps + RMSE_pd) ===")
    for it in leaderboard_sorted:
        s = keysum(it)
        print(f"{it['model']:>16s} : {s:.3f}  (ps={it.get('rmse_ps',np.nan):.3f}, pd={it.get('rmse_pd',np.nan):.3f})")
    print(f"[OK] saved to {args.out_dir}/summary.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
