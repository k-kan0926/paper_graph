#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-1: Model zoo benchmark (with progress logs & vectorized static eval)

- Static + shared dynamics: HW_Poly / HW_RBF / PWA / GP / MonoNN
- Sequence: ARX / NARX / NLSS(GRU) / Koopman(EDMD)

Outputs:
  out-dir/summary.json (or summary_partial.json on Ctrl+C)
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
    """theta_stat_vec must accept arrays and return array."""
    N = len(ps)
    start = d + 1
    th_pred = th.copy()
    for k in range(start, N-1):
        ku = k - d
        kup = k - d - 1
        # vectorized but here 1 sample; keep signature consistent
        ths = float(theta_stat_vec(ps[ku:ku+1], pd[ku:ku+1])[0])
        dS  = (ps[ku] - ps[kup]) / dt
        dD  = (pd[ku] - pd[kup]) / dt
        rhs = alpha*(ths - th_pred[k]) + kS*dS + kD*dD
        th_pred[k+1] = th_pred[k] + dt*rhs
    err = th_pred[start:] - th[start:]
    return float(np.sqrt(np.mean(err**2)))

def fit_dyn_with_delay(theta_stat_vec, ps, pd, th, dt, sg_win, sg_poly, delay_grid, l2_dyn=5e-3, kdelta_min=-2.0, log_prefix=""):
    """theta_stat_vec must accept arrays."""
    if not SCIPY_OK:
        raise RuntimeError("scipy is required for fit_dyn_with_delay().")
    from scipy.optimize import lsq_linear

    t0 = time.time()
    ps_s, _ = smooth_and_deriv(ps, dt, win=sg_win, poly=sg_poly)
    pd_s, _ = smooth_and_deriv(pd, dt, win=sg_win, poly=sg_poly)
    th_s, dth= smooth_and_deriv(th, dt, win=sg_win, poly=sg_poly)
    delays = [int(x) for x in delay_grid.split(",") if x.strip()!=""]
    best = None; logs = []
    N = len(ps)

    print(f"{log_prefix}  [dyn] start grid-search over delays: {delays}")
    for d in delays:
        start = d+1
        idx = np.arange(start, N)
        ku = idx - d
        kup = idx - d - 1

        # ---- vectorized static eval here ----
        ths = theta_stat_vec(ps_s[ku], pd_s[ku])  # array
        phi1= ths - th_s[idx]
        dS  = (ps_s[ku] - ps_s[kup]) / dt
        dD  = (pd_s[ku] - pd_s[kup]) / dt
        y   = dth[idx]
        X   = np.column_stack([phi1, dS, dD])

        scales = np.std(X, axis=0, ddof=1); scales[scales<1e-8]=1.0
        Xs = X / scales
        lam = float(l2_dyn)
        X_aug = np.vstack([Xs, math.sqrt(lam)*np.eye(3)])
        y_aug = np.hstack([y, np.zeros(3)])

        lb = [0.0, -np.inf, float(kdelta_min)]
        ub = [np.inf, np.inf, np.inf]
        sol = lsq_linear(X_aug, y_aug, bounds=(lb,ub), method="trf", lsmr_tol='auto', verbose=0)
        alpha, kS, kD = (sol.x / scales).tolist()

        rmse = rollout_rmse_firstorder(ps_s, pd_s, th_s, dt, d, alpha, kS, kD, theta_stat_vec)
        logs.append(dict(d=d, alpha=alpha, kS=kS, kD=kD, rmse=rmse))
        print(f"{log_prefix}    delay={d}: RMSE={rmse:.3f}  (alpha={alpha:.3f}, kS={kS:.3f}, kD={kD:.3f})")
        if (best is None) or (rmse < best["rmse"]):
            best = dict(d=d, alpha=alpha, kS=kS, kD=kD, rmse=rmse)

    print(f"{log_prefix}  [dyn] done in {time.time()-t0:.2f}s -> best: d={best['d']}, RMSE={best['rmse']:.3f}")
    return best, logs

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
        opt = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)
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

def arx_fit_predict(ps, pd, th, lag_y=3, lag_u=3, delay=0, alpha=1e-3, log_prefix=""):
    if not SKLEARN_OK: raise RuntimeError("sklearn needed for ARX.")
    t0=time.time()
    U = np.column_stack([ps, pd])
    X, y, T0 = build_lag_matrix(U, th, lag_u=lag_u, lag_y=lag_y, delay=delay)
    mdl = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
    N = len(th); yhat = th.copy()
    for t in range(T0, N):
        feats=[]
        for k in range(1, lag_y+1): feats.append(yhat[t-k])
        for k in range(delay, delay+lag_u): feats.extend(U[t-1-k])
        yhat[t] = mdl.predict(np.array(feats,float).reshape(1,-1))[0]
    rmse = float(np.sqrt(np.mean((yhat[T0:]-th[T0:])**2)))
    print(f"{log_prefix}  [ARX] delay={delay}: RMSE={rmse:.3f} in {time.time()-t0:.2f}s")
    return rmse, dict(model="ARX", coef=mdl.coef_.tolist(), intercept=float(mdl.intercept_), lag_y=lag_y, lag_u=lag_u, delay=delay)

def narx_fit_predict(ps, pd, th, lag_y=2, lag_u=2, delay=0, degree=2, alpha=1e-3, log_prefix=""):
    if not SKLEARN_OK: raise RuntimeError("sklearn needed for NARX.")
    t0=time.time()
    U = np.column_stack([ps, pd])
    X, y, T0 = build_lag_matrix(U, th, lag_u=lag_u, lag_y=lag_y, delay=delay)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X)
    mdl = Ridge(alpha=alpha, fit_intercept=False).fit(Xp, y)
    N = len(th); yhat = th.copy()
    for t in range(T0, N):
        feats=[]
        for k in range(1, lag_y+1): feats.append(yhat[t-k])
        for k in range(delay, delay+lag_u): feats.extend(U[t-1-k])
        xp = poly.transform(np.array(feats,float).reshape(1,-1))
        yhat[t] = mdl.predict(xp)[0]
    rmse = float(np.sqrt(np.mean((yhat[T0:]-th[T0:])**2)))
    print(f"{log_prefix}  [NARX] delay={delay}: RMSE={rmse:.3f} in {time.time()-t0:.2f}s")
    return rmse, dict(model="NARX", degree=degree, coef=mdl.coef_.tolist(), lag_y=lag_y, lag_u=lag_u, delay=delay)

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

def nlss_gru_fit_rollout(ps, pd, th, epochs=60, hidden=32, lr=1e-3, seed=0, log_prefix=""):
    torch.manual_seed(seed)
    t0=time.time()
    u = np.stack([ps, pd], axis=1).astype(np.float32)
    y = th.astype(np.float32)
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
    print(f"{log_prefix}  [NLSS_GRU] done in {time.time()-t0:.2f}s, RMSE={rmse:.3f}")
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

def koopman_edmd_fit_rollout(ps, pd, th, degree=2, log_prefix=""):
    t0=time.time()
    XU = np.column_stack([th, ps, pd])
    Z = poly_dict(XU, degree=degree)
    Zp = Z[1:]; Zk = Z[:-1]
    K = LA.lstsq(Zk, Zp, rcond=None)[0]
    W = LA.lstsq(Z, th, rcond=None)[0]
    z = Z[0].copy(); yhat = np.zeros_like(th); yhat[0] = th[0]
    for k in range(1, len(th)):
        z = z @ K
        yhat[k] = float(z @ W)
    T0=5
    rmse = float(np.sqrt(np.mean((yhat[T0:]-th[T0:])**2)))
    print(f"{log_prefix}  [EDMD] degree={degree}: RMSE={rmse:.3f} in {time.time()-t0:.2f}s")
    return rmse, dict(model="Koopman_EDMD", degree=degree)

# ----------------- wrapper: static + shared dynamics -----------------
def eval_static_plus_dyn(name, static_model, ps, pd, th, dt, sg_win, sg_poly, delay_grid,
                         muS, sdS, muD, sdD, log_prefix=""):
    # vectorized closure: accepts arrays
    def theta_stat_vec(ps_array, pd_array):
        Sh = (np.asarray(ps_array) - muS)/sdS
        Dh = (np.asarray(pd_array) - muD)/sdD
        return static_model.predict(Sh, Dh)

    best_dyn, dyn_logs = fit_dyn_with_delay(theta_stat_vec, ps, pd, th, dt,
                                            sg_win, sg_poly, delay_grid,
                                            l2_dyn=5e-3, kdelta_min=-2.0, log_prefix=log_prefix)
    return dict(model=name, dynamics=best_dyn, logs=dyn_logs)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col-sum", default="p_sum[MPa]")
    ap.add_argument("--col-diff", default="p_diff[MPa]")
    ap.add_argument("--col-theta", default="theta[deg]")
    ap.add_argument("--col-time", default="time[s]")
    ap.add_argument("--dt", type=float, default=None)
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

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        # ---- load data ----
        print("[load] CSV:", args.csv)
        import pandas as pd
        df = pd.read_csv(args.csv, comment="#")
        for c in [args.col_sum, args.col_diff, args.col_theta]:
            if c not in df.columns:
                raise RuntimeError(f"missing column: {c}")
        ps = df[args.col_sum].to_numpy(float)
        pdv= df[args.col_diff].to_numpy(float)
        th = df[args.col_theta].to_numpy(float)

        if args.dt is not None:
            dt = float(args.dt)
        elif args.col_time in df.columns:
            t = df[args.col_time].to_numpy(float)
            dt = float(np.nanmedian(np.diff(t)))
        else:
            dt = 0.01
        print(f"[info] N={len(ps)}, dt≈{dt:.5f}s, val_ratio={args.val_ratio}")

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

        # ---- Static + shared dynamics ----
        if flags["hw_poly"] and SKLEARN_OK:
            print("\n[HW_Poly] fit static...")
            mdl = StaticPoly(deg=args.poly_degree, alpha=args.ridge_alpha).fit(Sh_tr, Dh_tr, th_tr)
            print("[HW_Poly] fit dynamics + delay search...")
            pack = eval_static_plus_dyn("HW_Poly", mdl, ps, pdv, th, dt, args.sg_win, args.sg_poly,
                                        args.delay_grid, muS, sdS, muD, sdD, log_prefix="[HW_Poly]")
            add_result("HW_Poly", pack["dynamics"]["rmse"], extra=pack)
            save_json(os.path.join(args.out_dir, "hw_poly_meta.json"), pack)

        if flags["hw_rbf"] and SKLEARN_OK:
            print("\n[HW_RBF] fit static...")
            mdl = StaticRBF(n_centers=args.rbf_centers, length=args.rbf_length, alpha=args.ridge_alpha).fit(Sh_tr, Dh_tr, th_tr)
            print("[HW_RBF] fit dynamics + delay search...")
            pack = eval_static_plus_dyn("HW_RBF", mdl, ps, pdv, th, dt, args.sg_win, args.sg_poly,
                                        args.delay_grid, muS, sdS, muD, sdD, log_prefix="[HW_RBF]")
            add_result("HW_RBF", pack["dynamics"]["rmse"], extra=pack)
            save_json(os.path.join(args.out_dir, "hw_rbf_meta.json"), pack)

        if flags["pwa"]:
            print("\n[PWA] fit static...")
            mdl = StaticPWA(n_bins=args.pwa_bins, alpha=args.ridge_alpha).fit(Sh_tr, Dh_tr, th_tr)
            print("[PWA] fit dynamics + delay search...")
            pack = eval_static_plus_dyn("PWA", mdl, ps, pdv, th, dt, args.sg_win, args.sg_poly,
                                        args.delay_grid, muS, sdS, muD, sdD, log_prefix="[PWA]")
            add_result("PWA", pack["dynamics"]["rmse"], extra=pack)
            save_json(os.path.join(args.out_dir, "pwa_meta.json"), pack)

        if flags["gp"] and SKLEARN_GP_OK:
            print("\n[GP] fit static...")
            try:
                mdl = StaticGP(gp_maxn=args.gp_maxn).fit(Sh_tr, Dh_tr, th_tr)
                print("[GP] fit dynamics + delay search...")
                pack = eval_static_plus_dyn("GP", mdl, ps, pdv, th, dt, args.sg_win, args.sg_poly,
                                            args.delay_grid, muS, sdS, muD, sdD, log_prefix="[GP]")
                add_result("GP", pack["dynamics"]["rmse"], extra=pack)
                save_json(os.path.join(args.out_dir, "gp_meta.json"),
                          dict(kernel=str(mdl.gp.kernel_), dyn=pack["dynamics"]))
            except Exception as e:
                print("[GP] ERROR:", e)
                add_result("GP", float("inf"), extra=dict(error=str(e)))

        if flags["mono_nn"]:
            print("\n[MonoNN] fit static (torch)...")
            mdl = StaticMonoNN(M=args.mono_nn_M, hidden=args.mono_nn_hidden, epochs=args.mono_nn_epochs,
                               lr=args.mono_nn_lr, batch=args.mono_nn_batch,
                               learn_centers=args.mono_nn_learn_centers).fit(Sh_tr, Dh_tr, th_tr)
            print("[MonoNN] fit dynamics + delay search...")
            pack = eval_static_plus_dyn("MonoNN", mdl, ps, pdv, th, dt, args.sg_win, args.sg_poly,
                                        args.delay_grid, muS, sdS, muD, sdD, log_prefix="[MonoNN]")
            add_result("MonoNN", pack["dynamics"]["rmse"], extra=pack)
            torch.save(mdl.state_dict, os.path.join(args.out_dir, "mononn_state.pt"))
            save_json(os.path.join(args.out_dir, "mononn_meta.json"),
                      dict(M=args.mono_nn_M, hidden=args.mono_nn_hidden,
                           learn_centers=args.mono_nn_learn_centers, dyn=pack["dynamics"]))

        # ---- Sequence models ----
        if flags["arx"] and SKLEARN_OK:
            print("\n[ARX] grid over delays...")
            best=None
            for d in [int(x) for x in args.delay_grid.split(",") if x.strip()!=""]:
                rmse, meta = arx_fit_predict(ps, pdv, th, lag_y=args.arx_lag, lag_u=args.arx_lag, delay=d,
                                             alpha=args.ridge_alpha, log_prefix="[ARX]")
                if best is None or rmse < best[0]: best=(rmse, meta)
            add_result("ARX", best[0], extra=best[1])
            save_json(os.path.join(args.out_dir, "arx_meta.json"), best[1])

        if flags["narx"] and SKLEARN_OK:
            print("\n[NARX] grid over delays...")
            best=None
            for d in [int(x) for x in args.delay_grid.split(",") if x.strip()!=""]:
                rmse, meta = narx_fit_predict(ps, pdv, th, lag_y=args.narx_lag, lag_u=args.narx_lag, delay=d,
                                              degree=args.narx_degree, alpha=args.ridge_alpha, log_prefix="[NARX]")
                if best is None or rmse < best[0]: best=(rmse, meta)
            add_result("NARX", best[0], extra=best[1])
            save_json(os.path.join(args.out_dir, "narx_meta.json"), best[1])

        if flags["nlss"]:
            print("\n[NLSS_GRU] train...")
            rmse, meta = nlss_gru_fit_rollout(ps, pdv, th, epochs=args.nlss_epochs, hidden=args.nlss_hidden,
                                              log_prefix="[NLSS]")
            add_result("NLSS_GRU", rmse, extra=meta)
            save_json(os.path.join(args.out_dir, "nlss_meta.json"), meta)

        if flags["koopman"]:
            print("\n[EDMD] fit & rollout...")
            rmse, meta = koopman_edmd_fit_rollout(ps, pdv, th, degree=args.koopman_degree, log_prefix="[EDMD]")
            add_result("Koopman_EDMD", rmse, extra=meta)
            save_json(os.path.join(args.out_dir, "koopman_meta.json"), meta)

        # ---- summary ----
        leaderboard_sorted = sorted(leaderboard, key=lambda x: x["val_rmse_deg"])
        summary = dict(results=leaderboard_sorted, N=len(ps), dt=dt, val_ratio=args.val_ratio)
        save_json(os.path.join(args.out_dir, "summary.json"), summary)
        print("\n=== Leaderboard (val RMSE [deg]) ===")
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
