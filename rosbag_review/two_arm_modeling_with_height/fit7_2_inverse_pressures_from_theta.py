#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse mapping benchmark with multi-session CSVs:
  Learn (p_sum, p_diff) = f(theta[, z]) from multiple CSV sessions.

- Static: Inv-Poly / Inv-RBF / Inv-PWA / Inv-GP / Inv-MonoNN
- Sequence: Inv-ARX / Inv-NARX / Inv-NLSS(GRU) / Inv-Koopman(EDMD)
- Session split (default):
    train = dyn_csvs[:-2], val = dyn_csvs[-2], test = dyn_csvs[-1]
- Feasibility projection:
    enforce p_sum >= |p_diff| and 0<=p1,p2<=pmax (then map back to ps,pd)

Inputs CSV columns (defaults):
  theta[deg], p_sum[MPa], p_diff[MPa], (optional) z[m], time[s]

Outputs:
  out-dir/summary.json
  out-dir/{inv_poly|inv_rbf|inv_pwa|inv_gp|inv_mononn|inv_arx|inv_narx|inv_nlss_gru|inv_koopman_edmd}_meta.json
  out-dir/{...}_common_meta.npz     (controller-friendly stats)
  
python fit7_2_inverse_pressures_from_theta.py \
  --dyn_csvs \
    out/dynamic_prbs_data.csv out/dynamic_multi_data.csv out/dynamic_cyrip_data.csv \
  --stat_csvs out/static1_data.csv out/static2_data.csv \
  --out-dir out_inv_multi \
  --col-theta "theta[deg]" --col-sum "p_sum[MPa]" --col-diff "p_diff[MPa]" --col-z "z[m]" \
  --pmax 0.7 \
  --enable-all \
  --poly-degree 3 --rbf-centers 40 --rbf-length 0.6 --pwa-bins 12 \
  --arx-lag 3 --arx-degree 1 \
  --narx-lag 2 --narx-degree 2 \
  --nlss-epochs 80 --nlss-hidden 32 --koopman-degree 2 \
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

def train_val_split_by_sessions(dyn_csvs):
    if len(dyn_csvs) == 1:
        return dyn_csvs, dyn_csvs, dyn_csvs
    elif len(dyn_csvs) == 2:
        return [dyn_csvs[0]], [dyn_csvs[1]], [dyn_csvs[1]]
    else:
        return dyn_csvs[:-2], [dyn_csvs[-2]], [dyn_csvs[-1]]

def load_csv(path, col_theta, col_sum, col_diff, col_z=None, col_time=None):
    import pandas as pd
    df = pd.read_csv(path, comment="#").dropna().reset_index(drop=True)
    for c in [col_theta, col_sum, col_diff]:
        if c not in df.columns:
            raise RuntimeError(f"{path}: missing column {c}")
    if col_z is not None and (col_z not in df.columns):
        raise RuntimeError(f"{path}: missing column {col_z}")
    dt = None
    if col_time and (col_time in df.columns):
        t = df[col_time].to_numpy(float)
        if len(t) > 1:
            dt = float(np.nanmedian(np.diff(t)))
    if dt is None or not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    return df, dt

def smooth_and_deriv(x, dt, win=17, poly=3):
    x = np.asarray(x, float)
    if not SCIPY_OK or len(x) < max(7, poly+2):
        xs = x
        dx = np.gradient(xs, dt)
        return xs, dx
    win = min(int(win)|1, max(5, (len(x)//2)*2-1))  # odd
    from scipy.signal import savgol_filter
    xs = savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
    dx = savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=dt, mode="interp")
    return xs, dx

def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def to_p1p2(ps, pd):
    p1 = 0.5*(ps + pd); p2 = 0.5*(ps - pd)
    return p1, p2

def from_p1p2(p1, p2):
    return p1 + p2, p1 - p2

def project_feasible(ps, pd, pmax):
    ps = np.asarray(ps, float).copy()
    pd = np.asarray(pd, float).copy()
    mask = (np.abs(pd) > ps)
    if np.any(mask):
        ps[mask] = np.abs(pd[mask])
    p1, p2 = to_p1p2(ps, pd)
    p1 = np.clip(p1, 0.0, pmax)
    p2 = np.clip(p2, 0.0, pmax)
    return from_p1p2(p1, p2)  # (ps, pd) after clipping

def reg_metrics_2d(y_true, y_pred):
    m={}
    err_ps = y_pred[:,0] - y_true[:,0]
    err_pd = y_pred[:,1] - y_true[:,1]
    m["rmse_ps"] = float(np.sqrt(np.mean(err_ps**2)))
    m["rmse_pd"] = float(np.sqrt(np.mean(err_pd**2)))
    t_p1, t_p2 = to_p1p2(y_true[:,0], y_true[:,1])
    h_p1, h_p2 = to_p1p2(y_pred[:,0], y_pred[:,1])
    m["rmse_p1"] = float(np.sqrt(np.mean((h_p1 - t_p1)**2)))
    m["rmse_p2"] = float(np.sqrt(np.mean((h_p2 - t_p2)**2)))
    return m

def stack_sessions_for_inverse(csvs, *, col_theta, col_sum, col_diff, col_z=None, col_time=None,
                               sg_win=17, sg_poly=3, use_smooth_theta=True):
    """
    Build X=[theta(,z)] and Y=[ps,pd] by concatenating sessions (no lags).
    Return X, Y, (mu,sd), per-session ranges, and an example dt.
    """
    Xs, Ys = [], []
    dt_ref = 0.01
    th_min, th_max = +np.inf, -np.inf
    for p in csvs:
        df, dt = load_csv(p, col_theta, col_sum, col_diff, col_z, col_time)
        dt_ref = dt
        th = df[col_theta].to_numpy(float)
        if use_smooth_theta:
            th, _ = smooth_and_deriv(th, dt, win=sg_win, poly=sg_poly)
        th_min = min(th_min, float(np.nanmin(th)))
        th_max = max(th_max, float(np.nanmax(th)))
        if col_z is None:
            X = th.reshape(-1,1)
        else:
            z = df[col_z].to_numpy(float)
            X = np.column_stack([th, z])
        Y = np.column_stack([df[col_sum].to_numpy(float), df[col_diff].to_numpy(float)])
        ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
        if np.count_nonzero(ok) > 0:
            Xs.append(X[ok]); Ys.append(Y[ok])
    if not Xs:
        raise RuntimeError("No usable samples from given CSVs.")
    X_all = np.vstack(Xs); Y_all = np.vstack(Ys)
    mu, sd = standardize_fit(X_all)
    return X_all, Y_all, mu, sd, (th_min, th_max), dt_ref

# ---------- Models ----------
class InvModel:  # interface
    def fit(self, X, Y): ...
    def predict(self, X): ...

class InvPoly(InvModel):
    def __init__(self, deg=3, alpha=1e-3):
        if not SKLEARN_OK: raise RuntimeError("sklearn needed for InvPoly.")
        self.poly = PolynomialFeatures(degree=deg, include_bias=True)
        self.ridge = Ridge(alpha=alpha, fit_intercept=False)
        self.meta = dict(degree=deg, alpha=alpha)
    def fit(self, X, Y):
        t0=time.time()
        XP = self.poly.fit_transform(X)
        self.ridge.fit(XP, Y)
        print(f"    [Inv-Poly] {len(Y)} samples in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        XP = self.poly.transform(X)
        return self.ridge.predict(XP)

class InvRBF(InvModel):
    def __init__(self, n_centers=40, length=0.6, alpha=1e-3, seed=0):
        if not SKLEARN_OK: raise RuntimeError("sklearn needed for InvRBF.")
        self.n_centers=n_centers; self.length=float(length); self.alpha=float(alpha); self.seed=int(seed)
        self.C=None; self.W=None
        self.meta = dict(n_centers=n_centers, length=length, alpha=alpha)
    def _features(self, X):
        diff = X[:,None,:] - self.C[None,:,:]
        r2 = np.sum(diff*diff, axis=2)
        Phi = np.exp(-0.5*r2/(self.length**2))
        Phi = np.hstack([np.ones((Phi.shape[0],1)), Phi])
        return Phi
    def fit(self, X, Y):
        t0=time.time()
        km = KMeans(n_clusters=self.n_centers, random_state=self.seed).fit(X)
        self.C = km.cluster_centers_
        Phi = self._features(X)
        A = Phi.T@Phi + self.alpha*np.eye(Phi.shape[1])
        self.W = LA.solve(A, Phi.T@Y)
        print(f"    [Inv-RBF] {len(Y)} samples, centers={len(self.C)} in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        return self._features(X) @ self.W

class InvPWA(InvModel):
    def __init__(self, n_bins=12, alpha=1e-3):
        self.n_bins=int(n_bins); self.alpha=float(alpha)
        self.edges=None; self.W=[]
        self.meta = dict(n_bins=n_bins, alpha=alpha)
    def fit(self, X, Y):
        t0=time.time()
        th = X[:,0]
        self.edges = np.quantile(th, np.linspace(0,1,self.n_bins+1))
        self.W=[]
        for bi in range(self.n_bins):
            lo, hi = self.edges[bi], self.edges[bi+1]
            idx = np.where((th>=lo) & (th<=hi))[0]
            if len(idx) < 10:
                self.W.append(np.zeros((X.shape[1]+1,2))); continue
            Xb = np.column_stack([np.ones(len(idx)), X[idx]])
            Yb = Y[idx]
            A = Xb.T@Xb + self.alpha*np.eye(Xb.shape[1]); B = Xb.T@Yb
            self.W.append(LA.solve(A,B))
        print(f"    [Inv-PWA] {len(Y)} samples, bins={self.n_bins} in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        th = X[:,0]; Y = np.zeros((len(X),2))
        for i in range(len(X)):
            bi = min(self.n_bins-1, max(0, int(np.searchsorted(self.edges, th[i])-1)))
            W = self.W[bi]; Y[i] = (np.hstack([1.0, X[i]]) @ W).ravel()
        return Y

class InvGP(InvModel):
    def __init__(self, gp_maxn=10000, seed=0):
        if not SKLEARN_GP_OK: raise RuntimeError("sklearn GP needed for InvGP.")
        self.gp_maxn=int(gp_maxn); self.seed=int(seed)
        self.gp_ps=None; self.gp_pd=None
        self.meta = dict(gp_maxn=gp_maxn)
    def fit(self, X, Y):
        t0=time.time()
        if len(X) > self.gp_maxn:
            rs = np.random.RandomState(self.seed)
            idx = rs.choice(len(X), self.gp_maxn, replace=False)
            X = X[idx]; Y = Y[idx]
            print(f"    [Inv-GP] subsampled to {len(X)}")
        kernel = ConstantKernel(1.0, (1e-2,1e2)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2,1e2)) \
                 + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6,1e-1))
        self.gp_ps = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=2, normalize_y=True, random_state=self.seed)
        self.gp_pd = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=2, normalize_y=True, random_state=self.seed+1)
        self.gp_ps.fit(X, Y[:,0]); self.gp_pd.fit(X, Y[:,1])
        print(f"    [Inv-GP] {len(X)} samples in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        ps = self.gp_ps.predict(X); pd = self.gp_pd.predict(X)
        return np.column_stack([ps,pd])

# --- MonoNN (torch) ---
class MonoHead(nn.Module):
    def __init__(self, hidden=64, out_dim=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x):
        return self.out(self.enc(x))

class InvMonoNN(InvModel):
    def __init__(self, epochs=250, lr=1e-3, batch=1024, seed=0, lam_feas=0.0, pmax=0.7):
        self.epochs=epochs; self.lr=lr; self.batch=batch; self.seed=seed
        self.lam_feas=float(lam_feas); self.pmax=float(pmax)
        self.model=None
        self.meta = dict(epochs=epochs, lr=lr, batch=batch, lam_feas=lam_feas)
    def fit(self, X, Y):
        t0=time.time()
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                            torch.tensor(Y, dtype=torch.float32))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True, drop_last=False)
        model = MonoHead(hidden=64, out_dim=2)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)
        loss_fn = nn.HuberLoss(delta=3.0)
        model.train()
        for ep in range(self.epochs):
            tot=0.0
            for xb,yb in dl:
                opt.zero_grad()
                yh = model(xb)
                loss = loss_fn(yh, yb)
                if self.lam_feas > 0:
                    ps, pd = yh[:,0], yh[:,1]
                    p1 = 0.5*(ps+pd); p2 = 0.5*(ps-pd)
                    feas = torch.relu(torch.abs(pd)-ps) \
                         + torch.relu(-p1) + torch.relu(-p2) \
                         + torch.relu(p1-self.pmax) + torch.relu(p2-self.pmax)
                    loss = loss + self.lam_feas*feas.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); tot += float(loss)
            if (ep+1)%50==0:
                print(f"      [Inv-MonoNN] epoch {ep+1}/{self.epochs} loss≈{tot/max(1,len(dl)):.4f}")
        self.model = model.eval()
        print(f"    [Inv-MonoNN] {len(Y)} samples in {time.time()-t0:.2f}s")
        return self
    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            Y = self.model(X).cpu().numpy()
        return Y

# --------- Sequence inverse ----------
def build_lag_matrix_inverse(theta, z=None, lag=3):
    N=len(theta); T0=lag
    rows=[]
    for t in range(T0, N):
        feats=[]
        for k in range(1, lag+1): feats.append(theta[t-k])
        if z is not None:
            for k in range(1, lag+1): feats.append(z[t-k])
        rows.append(feats)
    X=np.array(rows,float); return X, T0

def arx_inverse_fit_predict(csvs, col_theta, col_sum, col_diff, col_z, col_time, lag, degree, alpha, pmax):
    # Fit on all train-val rows concatenated; test metrics per-session (last one)
    from sklearn.linear_model import Ridge
    Xs=[]; Ys=[]
    last_df=None; last_dt=0.01
    for p in csvs:
        df, dt = load_csv(p, col_theta, col_sum, col_diff, col_z, col_time)
        th = df[col_theta].to_numpy(float)
        z = df[col_z].to_numpy(float) if col_z else None
        X, T0 = build_lag_matrix_inverse(th, z=z, lag=lag)
        Y = np.column_stack([df[col_sum].to_numpy(float), df[col_diff].to_numpy(float)])[T0:]
        if len(X)>0:
            Xs.append(X); Ys.append(Y)
            last_df, last_dt = df, dt
    Xall = np.vstack(Xs); Yall=np.vstack(Ys)
    if degree>1:
        pf = PolynomialFeatures(degree=degree, include_bias=True); XP = pf.fit_transform(Xall)
    else:
        pf=None; XP=Xall
    mdl = Ridge(alpha=alpha, fit_intercept=True).fit(XP, Yall)
    # Evaluate on last session only
    th = last_df[col_theta].to_numpy(float)
    z  = last_df[col_z].to_numpy(float) if col_z else None
    Xte, T0 = build_lag_matrix_inverse(th, z=z, lag=lag)
    Yte = np.column_stack([last_df[col_sum].to_numpy(float), last_df[col_diff].to_numpy(float)])[T0:]
    XPte = pf.transform(Xte) if pf is not None else Xte
    Yhat = mdl.predict(XPte)
    psf,pdf = project_feasible(Yhat[:,0], Yhat[:,1], pmax)
    rm = reg_metrics_2d(Yte, np.column_stack([psf,pdf]))
    meta = dict(model="Inv_ARX", lag=lag, degree=degree, alpha=alpha)
    return rm, meta, last_dt

def nlss_gru_inverse_fit_predict(csvs, col_theta, col_sum, col_diff, col_z, col_time, epochs, hidden, lr, pmax, seed=0):
    torch.manual_seed(seed)
    u_list=[]; y_list=[]
    last_df=None; last_dt=0.01
    for p in csvs:
        df, dt = load_csv(p, col_theta, col_sum, col_diff, col_z, col_time)
        th = df[col_theta].to_numpy(float)
        if col_z:
            z = df[col_z].to_numpy(float); u = np.stack([th,z],axis=1).astype(np.float32)
        else:
            u = th[:,None].astype(np.float32)
        y = np.stack([df[col_sum].to_numpy(float), df[col_diff].to_numpy(float)],axis=1).astype(np.float32)
        u_list.append(u[None,...]); y_list.append(y[None,...])
        last_df, last_dt = df, dt
    ut = torch.tensor(np.concatenate(u_list,axis=0))  # [S,T,in]
    yt = torch.tensor(np.concatenate(y_list,axis=0))  # [S,T,2]

    class GRUInv(nn.Module):
        def __init__(self,in_dim,hidden=32):
            super().__init__()
            self.gru=nn.GRU(input_size=in_dim,hidden_size=hidden,batch_first=True)
            self.head=nn.Linear(hidden,2)
        def forward(self,u):
            y,h=self.gru(u); return self.head(y)

    mdl = GRUInv(ut.shape[-1], hidden)
    opt = optim.Adam(mdl.parameters(), lr=lr)
    crit = nn.MSELoss()
    mdl.train()
    for ep in range(epochs):
        opt.zero_grad()
        yhat = mdl(ut)
        loss = crit(yhat, yt)
        loss.backward()
        nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
        opt.step()
        if (ep+1)%10==0:
            print(f"    [Inv-NLSS_GRU] epoch {ep+1}/{epochs}, loss={float(loss):.5f}")
    mdl.eval()
    with torch.no_grad():
        yhat = mdl(torch.tensor(u_list[-1])).numpy().squeeze(0)  # last session
    Yte = np.stack([last_df[col_sum].to_numpy(float), last_df[col_diff].to_numpy(float)],axis=1)
    psf,pdf = project_feasible(yhat[:,0], yhat[:,1], pmax)
    rm = reg_metrics_2d(Yte, np.column_stack([psf,pdf]))
    meta = dict(model="Inv_NLSS_GRU", hidden=hidden, epochs=epochs, lr=lr)
    return rm, meta, last_dt

def koopman_edmd_inverse_fit_predict(csvs, col_theta, col_sum, col_diff, col_z, col_time, degree, pmax):
    Xs=[]; Ys=[]; last_df=None; last_dt=0.01
    for p in csvs:
        df, dt = load_csv(p, col_theta, col_sum, col_diff, col_z, col_time)
        if col_z:
            XU = np.column_stack([df[col_theta].to_numpy(float), df[col_z].to_numpy(float)])
        else:
            XU = df[col_theta].to_numpy(float)[:,None]
        Y = np.column_stack([df[col_sum].to_numpy(float), df[col_diff].to_numpy(float)])
        Xs.append(XU); Ys.append(Y); last_df, last_dt=df, dt
    X = np.vstack(Xs); Y = np.vstack(Ys)
    if SKLEARN_OK:
        pf = PolynomialFeatures(degree=degree, include_bias=True); Z = pf.fit_transform(X)
    else:
        Z = np.hstack([np.ones((len(X),1)), X])
    Zp = Z[1:]; Zk = Z[:-1]
    K = LA.lstsq(Zk, Zp, rcond=None)[0]
    W = LA.lstsq(Z, Y, rcond=None)[0]
    z = Z[0].copy()
    Yhat=np.zeros_like(Y); Yhat[0]=Y[0]
    for k in range(1,len(Y)):
        z = z @ K; Yhat[k] = z @ W
    # evaluate on last session
    Ylast = np.column_stack([last_df[col_sum].to_numpy(float), last_df[col_diff].to_numpy(float)])
    Yhlast = Yhat[-len(Ylast):]
    psf,pdf = project_feasible(Yhlast[:,0], Yhlast[:,1], pmax)
    rm = reg_metrics_2d(Ylast, np.column_stack([psf,pdf]))
    meta = dict(model="Inv_Koopman_EDMD", degree=degree)
    return rm, meta, last_dt

# --------- common meta NPZ ----------
def save_common_meta_npz(out_npz, *, dt, mu_theta, sd_theta, mu_z, sd_z, pmax, model_tag, extra_meta=None):
    np.savez(out_npz,
             dt=np.array(dt, dtype=np.float64),
             mu_theta=np.array(mu_theta if mu_theta is not None else np.nan, dtype=np.float64),
             sd_theta=np.array(sd_theta if sd_theta is not None else np.nan, dtype=np.float64),
             mu_z=np.array(mu_z if mu_z is not None else np.nan, dtype=np.float64),
             sd_z=np.array(sd_z if sd_z is not None else np.nan, dtype=np.float64),
             pmax=np.array(pmax, dtype=np.float64),
             model_tag=np.array(model_tag),
             extra_meta=json.dumps(extra_meta if extra_meta else {}))

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn_csvs", nargs="+", required=True, help="Dynamic session CSVs (>=1, typically 5)")
    ap.add_argument("--stat_csvs", nargs="*", default=[], help="Static CSVs (optional, not required here)")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--col-theta", default="theta[deg]")
    ap.add_argument("--col-sum",   default="p_sum[MPa]")
    ap.add_argument("--col-diff",  default="p_diff[MPa]")
    ap.add_argument("--col-z",     default=None)
    ap.add_argument("--col-time",  default="time[s]")

    ap.add_argument("--pmax", type=float, default=0.7)

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
    ap.add_argument("--enable-narx", action="store_true")  # alias of ARX with degree>1
    ap.add_argument("--enable-nlss", action="store_true")
    ap.add_argument("--enable-koopman", action="store_true")

    # static hyperparams
    ap.add_argument("--poly-degree", type=int, default=3)
    ap.add_argument("--rbf-centers", type=int, default=40)
    ap.add_argument("--rbf-length", type=float, default=0.6)
    ap.add_argument("--pwa-bins", type=int, default=12)
    ap.add_argument("--ridge-alpha", type=float, default=1e-3)
    ap.add_argument("--gp-maxn", type=int, default=10000)

    # mononn
    ap.add_argument("--mononn-epochs", type=int, default=250)
    ap.add_argument("--mononn-lr", type=float, default=1e-3)
    ap.add_argument("--mononn-batch", type=int, default=1024)
    ap.add_argument("--mononn-lam-feas", type=float, default=0.1)

    # sequence
    ap.add_argument("--arx-lag", type=int, default=3)
    ap.add_argument("--arx-degree", type=int, default=1)
    ap.add_argument("--narx-lag", type=int, default=2)
    ap.add_argument("--narx-degree", type=int, default=2)
    ap.add_argument("--nlss-epochs", type=int, default=80)
    ap.add_argument("--nlss-hidden", type=int, default=32)
    ap.add_argument("--koopman-degree", type=int, default=2)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # session split
    train_csvs, val_csvs, test_csvs = train_val_split_by_sessions(args.dyn_csvs)
    print("[sessions] train:", len(train_csvs), "val:", len(val_csvs), "test:", len(test_csvs))

    # build standardized X (theta[,z]) / Y (ps,pd) per split
    def build_split(csvs):
        X, Y, mu, sd, th_range, dt = stack_sessions_for_inverse(
            csvs,
            col_theta=args.col_theta, col_sum=args.col_sum, col_diff=args.col_diff,
            col_z=args.col_z, col_time=args.col_time,
            sg_win=args.sg_win, sg_poly=args.sg_poly, use_smooth_theta=True
        )
        return X, Y, mu, sd, th_range, dt

    Xtr, Ytr, mu_tr, sd_tr, th_rng_tr, dt_tr = build_split(train_csvs)
    Xva, Yva, _,      _,       _,        _   = build_split(val_csvs)
    Xte, Yte, _,      _,       _,       dt_te= build_split(test_csvs)

    # normalize by train stats
    Xtr_s = standardize_apply(Xtr, mu_tr, sd_tr)
    Xva_s = standardize_apply(Xva, mu_tr, sd_tr)
    Xte_s = standardize_apply(Xte, mu_tr, sd_tr)

    leaderboard=[]
    def push_result(tag, metrics, extra=None, dt_hint=None):
        row = dict(model=tag, **metrics)
        if extra: row.update(extra)
        leaderboard.append(row)
        save_json(os.path.join(args.out_dir, f"{tag}_meta.json"), row)
        # compact NPZ
        mu_th = float(mu_tr[0]); sd_th=float(sd_tr[0])
        mu_z  = float(mu_tr[1]) if (args.col_z is not None and len(mu_tr)>1) else None
        sd_z  = float(sd_tr[1]) if (args.col_z is not None and len(sd_tr)>1) else None
        save_common_meta_npz(
            os.path.join(args.out_dir, f"{tag}_common_meta.npz"),
            dt=(float(dt_hint) if dt_hint is not None else float(dt_te)),
            mu_theta=mu_th, sd_theta=sd_th, mu_z=mu_z, sd_z=sd_z,
            pmax=args.pmax, model_tag=tag, extra_meta={k:v for k,v in (extra or {}).items() if k not in metrics}
        )

    # -------------- Static models --------------
    flags = dict(poly=args.enable_poly, rbf=args.enable_rbf, pwa=args.enable_pwa,
                 gp=args.enable_gp, mononn=args.enable_mononn,
                 arx=args.enable_arx, narx=args.enable_narx,
                 nlss=args.enable_nlss, koopman=args.enable_koopman)
    if args.enable_all:
        for k in flags: flags[k]=True

    if flags["poly"] and SKLEARN_OK:
        print("\n[Inv-Poly] train/val/test ...")
        mdl = InvPoly(deg=args.poly_degree, alpha=args.ridge_alpha).fit(Xtr_s, Ytr)
        Yv = mdl.predict(Xva_s); psf,pdf = project_feasible(Yv[:,0], Yv[:,1], args.pmax)
        rm_va = reg_metrics_2d(Yva, np.column_stack([psf,pdf]))
        Yh = mdl.predict(Xte_s); psf,pdf = project_feasible(Yh[:,0], Yh[:,1], args.pmax)
        rm_te = reg_metrics_2d(Yte, np.column_stack([psf,pdf]))
        print(f"  [VA] RMSE_ps={rm_va['rmse_ps']:.3f}, RMSE_pd={rm_va['rmse_pd']:.3f}")
        print(f"  [TE] RMSE_ps={rm_te['rmse_ps']:.3f}, RMSE_pd={rm_te['rmse_pd']:.3f}")
        push_result("inv_poly", {**{f"va_{k}":v for k,v in rm_va.items()},
                                 **{f"te_{k}":v for k,v in rm_te.items()}},
                    extra=mdl.meta, dt_hint=dt_te)

    if flags["rbf"] and SKLEARN_OK:
        print("\n[Inv-RBF] train/val/test ...")
        mdl = InvRBF(n_centers=args.rbf_centers, length=args.rbf_length, alpha=args.ridge_alpha).fit(Xtr_s, Ytr)
        rm_va = reg_metrics_2d(Yva, np.column_stack(project_feasible(*mdl.predict(Xva_s).T, args.pmax)))
        rm_te = reg_metrics_2d(Yte, np.column_stack(project_feasible(*mdl.predict(Xte_s).T, args.pmax)))
        print(f"  [VA] RMSE_ps={rm_va['rmse_ps']:.3f}, RMSE_pd={rm_va['rmse_pd']:.3f}")
        print(f"  [TE] RMSE_ps={rm_te['rmse_ps']:.3f}, RMSE_pd={rm_te['rmse_pd']:.3f}")
        push_result("inv_rbf", {**{f"va_{k}":v for k,v in rm_va.items()},
                                **{f"te_{k}":v for k,v in rm_te.items()}},
                    extra=mdl.meta, dt_hint=dt_te)

    if flags["pwa"]:
        print("\n[Inv-PWA] train/val/test ...")
        mdl = InvPWA(n_bins=args.pwa_bins, alpha=args.ridge_alpha).fit(Xtr_s, Ytr)
        rm_va = reg_metrics_2d(Yva, np.column_stack(project_feasible(*mdl.predict(Xva_s).T, args.pmax)))
        rm_te = reg_metrics_2d(Yte, np.column_stack(project_feasible(*mdl.predict(Xte_s).T, args.pmax)))
        print(f"  [VA] RMSE_ps={rm_va['rmse_ps']:.3f}, RMSE_pd={rm_va['rmse_pd']:.3f}")
        print(f"  [TE] RMSE_ps={rm_te['rmse_ps']:.3f}, RMSE_pd={rm_te['rmse_pd']:.3f}")
        push_result("inv_pwa", {**{f"va_{k}":v for k,v in rm_va.items()},
                                **{f"te_{k}":v for k,v in rm_te.items()}},
                    extra=mdl.meta, dt_hint=dt_te)

    if flags["gp"] and SKLEARN_GP_OK:
        print("\n[Inv-GP] train/val/test ...")
        try:
            mdl = InvGP(gp_maxn=args.gp_maxn).fit(Xtr_s, Ytr)
            rm_va = reg_metrics_2d(Yva, np.column_stack(project_feasible(*mdl.predict(Xva_s).T, args.pmax)))
            rm_te = reg_metrics_2d(Yte, np.column_stack(project_feasible(*mdl.predict(Xte_s).T, args.pmax)))
            print(f"  [VA] RMSE_ps={rm_va['rmse_ps']:.3f}, RMSE_pd={rm_va['rmse_pd']:.3f}")
            print(f"  [TE] RMSE_ps={rm_te['rmse_ps']:.3f}, RMSE_pd={rm_te['rmse_pd']:.3f}")
            push_result("inv_gp", {**{f"va_{k}":v for k,v in rm_va.items()},
                                   **{f"te_{k}":v for k,v in rm_te.items()}},
                        extra=mdl.meta, dt_hint=dt_te)
        except Exception as e:
            print("[Inv-GP] ERROR:", e)
            push_result("inv_gp", dict(va_rmse_ps=float("inf"), va_rmse_pd=float("inf"),
                                       va_rmse_p1=float("inf"), va_rmse_p2=float("inf"),
                                       te_rmse_ps=float("inf"), te_rmse_pd=float("inf"),
                                       te_rmse_p1=float("inf"), te_rmse_p2=float("inf")),
                        extra=dict(error=str(e)), dt_hint=dt_te)

    if flags["mononn"]:
        print("\n[Inv-MonoNN] train/val/test ...")
        mdl = InvMonoNN(epochs=args.mononn_epochs, lr=args.mononn_lr, batch=args.mononn_batch,
                        lam_feas=args.mononn_lam_feas, pmax=args.pmax).fit(Xtr_s, Ytr)
        Yv = mdl.predict(Xva_s); rm_va = reg_metrics_2d(Yva, np.column_stack(project_feasible(*Yv.T, args.pmax)))
        Yh = mdl.predict(Xte_s); rm_te = reg_metrics_2d(Yte, np.column_stack(project_feasible(*Yh.T, args.pmax)))
        print(f"  [VA] RMSE_ps={rm_va['rmse_ps']:.3f}, RMSE_pd={rm_va['rmse_pd']:.3f}")
        print(f"  [TE] RMSE_ps={rm_te['rmse_ps']:.3f}, RMSE_pd={rm_te['rmse_pd']:.3f}")
        torch.save(mdl.model.state_dict(), os.path.join(args.out_dir, "inv_mononn_state.pt"))
        push_result("inv_mononn", {**{f"va_{k}":v for k,v in rm_va.items()},
                                   **{f"te_{k}":v for k,v in rm_te.items()}},
                    extra=mdl.meta, dt_hint=dt_te)

    # -------------- Sequence models --------------
    if flags["arx"] and SKLEARN_OK:
        print("\n[Inv-ARX] train/test (last session) ...")
        rm, meta, dt_hint = arx_inverse_fit_predict(train_csvs+val_csvs, args.col_theta, args.col_sum, args.col_diff,
                                                    args.col_z, args.col_time, lag=args.arx_lag,
                                                    degree=args.arx_degree, alpha=args.ridge_alpha, pmax=args.pmax)
        push_result("inv_arx",
                    {f"te_{k}":v for k,v in rm.items()},
                    extra=meta, dt_hint=dt_hint)

    if flags["narx"] and SKLEARN_OK:
        print("\n[Inv-NARX] (ARX with polynomial features) ...")
        rm, meta, dt_hint = arx_inverse_fit_predict(train_csvs+val_csvs, args.col_theta, args.col_sum, args.col_diff,
                                                    args.col_z, args.col_time, lag=args.narx_lag,
                                                    degree=args.narx_degree, alpha=args.ridge_alpha, pmax=args.pmax)
        meta["model"]="Inv_NARX"
        push_result("inv_narx",
                    {f"te_{k}":v for k,v in rm.items()},
                    extra=meta, dt_hint=dt_hint)

    if flags["nlss"]:
        print("\n[Inv-NLSS_GRU] train/test (last session) ...")
        rm, meta, dt_hint = nlss_gru_inverse_fit_predict(train_csvs+val_csvs, args.col_theta, args.col_sum, args.col_diff,
                                                         args.col_z, args.col_time, epochs=args.nlss_epochs,
                                                         hidden=args.nlss_hidden, lr=1e-3, pmax=args.pmax, seed=0)
        push_result("inv_nlss_gru",
                    {f"te_{k}":v for k,v in rm.items()},
                    extra=meta, dt_hint=dt_hint)

    if flags["koopman"]:
        print("\n[Inv-EDMD] fit/test (last session) ...")
        rm, meta, dt_hint = koopman_edmd_inverse_fit_predict(train_csvs+val_csvs, args.col_theta, args.col_sum, args.col_diff,
                                                             args.col_z, args.col_time, degree=args.koopman_degree, pmax=args.pmax)
        push_result("inv_koopman_edmd",
                    {f"te_{k}":v for k,v in rm.items()},
                    extra=meta, dt_hint=dt_hint)

    # ---- summary ----
    def keysum(it):
        # prefer TE metrics ifある、無ければVA
        te = it.get("te_rmse_ps"), it.get("te_rmse_pd")
        va = it.get("va_rmse_ps"), it.get("va_rmse_pd")
        if all(v is not None for v in te):
            return float(te[0])+float(te[1])
        return float(va[0])+float(va[1])
    leaderboard_sorted = sorted(leaderboard, key=keysum)
    save_json(os.path.join(args.out_dir, "summary.json"),
              dict(results=leaderboard_sorted,
                   train_csvs=train_csvs, val_csvs=val_csvs, test_csvs=test_csvs))
    print("\n=== Leaderboard (best = smaller RMSE_ps+RMSE_pd) ===")
    for it in leaderboard_sorted:
        te_s = it.get("te_rmse_ps", np.nan) + it.get("te_rmse_pd", np.nan) if "te_rmse_ps" in it else np.nan
        va_s = it.get("va_rmse_ps", np.nan) + it.get("va_rmse_pd", np.nan) if "va_rmse_ps" in it else np.nan
        print(f"{it['model']:>18s} : TE={te_s:.3f}  VA={va_s:.3f}")
    print(f"[OK] saved → {os.path.join(args.out_dir,'summary.json')}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
