#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive inverse MPC using Neural Hammerstein static kernel with delay d.

Example:
  python inverse4_predictive_neural_mpc.py --model-pt out/model_k4/model_k4_nh.pt --model-meta out/model_k4/model_k4_nh_meta.npz --theta-deg 20 --theta0 12.3 --H 10 --wz 0.3 --w-sigma 0.02 --w-rate 0.3 --w-block 10.0 --rate-sigma 0.12 --rate-delta 0.12
"""
import argparse, numpy as np, torch, torch.nn as nn

# ---- NN (must match training) ----
class MonoDeltaNN(nn.Module):
    def __init__(self, M, hidden, c_grid):
        super().__init__()
        self.M = M
        self.c = nn.Parameter(torch.tensor(c_grid, dtype=torch.float32), requires_grad=False)
        self.enc = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU()
        )
        self.head_A = nn.Linear(hidden, 1)
        self.head_w = nn.Linear(hidden, M)
        self.head_s = nn.Linear(hidden, M)
        nn.init.zeros_(self.head_w.weight); nn.init.zeros_(self.head_w.bias)
        nn.init.zeros_(self.head_s.weight); nn.init.constant_(self.head_s.bias, 1.0)

    def forward(self, Sigma_hat, Delta_hat):
        h = self.enc(Sigma_hat)
        A = self.head_A(h)
        w = torch.nn.functional.softplus(self.head_w(h)) + 1e-6
        s = torch.nn.functional.softplus(self.head_s(h)) + 1e-6
        d = Delta_hat - self.c.view(1,-1)
        bank = torch.tanh(s * d)
        return A + (w*bank).sum(dim=1, keepdim=True)

def theta_stat_fn_factory(meta, state_dict):
    M = int(meta["M"]); hidden = int(meta["hidden"])
    c_grid = meta["c_grid"].tolist() if hasattr(meta["c_grid"], "tolist") else list(meta["c_grid"])
    muS, sdS = float(meta["muS"]), float(meta["sdS"])
    muD, sdD = float(meta["muD"]), float(meta["sdD"])

    model = MonoDeltaNN(M, hidden, c_grid)
    model.load_state_dict(state_dict)
    model.eval()

    @torch.no_grad()
    def f(ps, pd):
        S = torch.tensor([(ps-muS)/sdS], dtype=torch.float32).view(1,1)
        D = torch.tensor([(pd-muD)/sdD], dtype=torch.float32).view(1,1)
        return float(model(S,D).item())
    return f

def z_pred(ps, pd, zc):
    if zc is None or zc.size < 4: return 0.0
    return float(zc[0] + zc[1]*ps + zc[2]*pd + zc[3]*ps*pd)

def clamp_box(p1, p2, pmax):
    return np.clip(p1,0,pmax), np.clip(p2,0,pmax)

def invert_delta_closed(ps, theta_star, f_theta_stat):
    # 1D solve for Δ via monotone property (bisection in Δ)
    lo, hi = -2.0, 2.0  # MPa range in Δ (wider than box; will be projected)
    for _ in range(50):
        mid = 0.5*(lo+hi)
        val = f_theta_stat(ps, mid) - theta_star
        if val==0.0: break
        # need sign at bounds
        vlo = f_theta_stat(ps, lo) - theta_star
        if np.sign(vlo)*np.sign(val) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pt", required=True)
    ap.add_argument("--model-meta", required=True)
    ap.add_argument("--theta-deg", type=float, required=True)
    ap.add_argument("--theta0", type=float, required=True)
    ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--delta0", type=float, default=None)
    ap.add_argument("--hist-sigma", type=str, default=None)
    ap.add_argument("--hist-delta", type=str, default=None)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--w-stage", type=float, default=1.0)
    ap.add_argument("--w-term", type=float, default=5.0)
    ap.add_argument("--wz", type=float, default=0.2)
    ap.add_argument("--w-sigma", type=float, default=0.02)
    ap.add_argument("--w-rate", type=float, default=0.3)
    ap.add_argument("--w-block", type=float, default=10.0)
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--rate-sigma", type=float, default=0.12)
    ap.add_argument("--rate-delta", type=float, default=0.12)
    ap.add_argument("--max-nfev", type=int, default=600)
    args = ap.parse_args()

    meta = np.load(args.model_meta, allow_pickle=True)
    state = torch.load(args.model_pt, map_location="cpu")
    theta_stat = theta_stat_fn_factory(meta, state)

    alpha = float(meta["alpha"]); kS = float(meta["kSigma"]); kD = float(meta["kDelta"])
    dt = float(meta["dt"]); delay = int(meta["delay"])
    pmax = float(meta["pmax"]); sigma_ref = float(meta["sigma_ref"])
    zc = meta["z_coef"] if "z_coef" in meta.files else np.array([])

    theta_star = float(args.theta_deg)
    theta0 = float(args.theta0)
    sigma0 = float(args.sigma0) if args.sigma0 is not None else sigma_ref
    if args.delta0 is not None:
        delta0 = float(args.delta0)
    else:
        delta0 = invert_delta_closed(sigma0, theta0, theta_stat)
        p1 = 0.5*(sigma0+delta0); p2=0.5*(sigma0-delta0)
        p1,p2 = clamp_box(p1,p2,pmax); sigma0 = p1+p2; delta0 = p1-p2

    # build history
    if delay > 0:
        if args.hist_sigma: hist_sigma = [float(x) for x in args.hist_sigma.split(",")]
        else:               hist_sigma = [sigma0]*delay
        if args.hist_delta: hist_delta = [float(x) for x in args.hist_delta.split(",")]
        else:               hist_delta = [delta0]*delay
        if len(hist_sigma)!=delay or len(hist_delta)!=delay:
            raise ValueError("hist lengths must equal delay")
    else:
        hist_sigma, hist_delta = [], []

    H = int(args.H)
    u = np.zeros(2*H, float)
    # static inverse warm start
    for k in range(H):
        ps = sigma_ref
        pd = invert_delta_closed(ps, theta_star, theta_stat)
        p1 = 0.5*(ps+pd); p2 = 0.5*(ps-pd); p1,p2 = clamp_box(p1,p2,pmax)
        u[2*k+0] = p1+p2; u[2*k+1] = p1-p2

    rateS = float(args.rate_sigma); rateD = float(args.rate_delta)
    block = max(1, int(args.block))

    def simulate(u_vec):
        sig_appl = np.r_[hist_sigma, u_vec[0::2]]
        del_appl = np.r_[hist_delta, u_vec[1::2]]

        theta = theta0
        cost_terms = []
        for k in range(H):
            # applied to plant
            ps_eff = sig_appl[k]; pd_eff = del_appl[k]
            if k == 0:
                ps_prev = (hist_sigma[-1] if delay>0 else sigma0)
                pd_prev = (hist_delta[-1] if delay>0 else delta0)
            else:
                ps_prev = sig_appl[k-1]; pd_prev = del_appl[k-1]
            dS_eff = (ps_eff-ps_prev)/dt; dD_eff = (pd_eff-pd_prev)/dt

            ths = theta_stat(ps_eff, pd_eff)
            theta = theta + dt*( alpha*(ths - theta) + kS*dS_eff + kD*dD_eff )

            # command penalties
            ps_cmd = u_vec[2*k+0]; pd_cmd = u_vec[2*k+1]
            p1 = 0.5*(ps_cmd+pd_cmd); p2 = 0.5*(ps_cmd-pd_cmd)
            box_violation = 0.0
            if p1<0: box_violation += -p1
            if p2<0: box_violation += -p2
            if p1>pmax: box_violation += (p1-pmax)
            if p2>pmax: box_violation += (p2-pmax)

            if k==0:
                dS_cmd = ps_cmd - sigma0; dD_cmd = pd_cmd - delta0
            else:
                dS_cmd = ps_cmd - u_vec[2*(k-1)+0]
                dD_cmd = pd_cmd - u_vec[2*(k-1)+1]
            rate_pen = max(0.0, abs(dS_cmd)-rateS) + max(0.0, abs(dD_cmd)-rateD)

            block_pen = 0.0
            if block>1 and (k%block!=0):
                ps_prev_cmd = u_vec[2*(k-1)+0]; pd_prev_cmd = u_vec[2*(k-1)+1]
                block_pen = (ps_cmd-ps_prev_cmd)**2 + (pd_cmd-pd_prev_cmd)**2

            c = args.w_stage*(theta-theta_star)**2 \
              + args.wz*(z_pred(ps_cmd,pd_cmd,zc)**2) \
              + args.w_sigma*((ps_cmd - sigma_ref)**2) \
              + args.w_rate*(dS_cmd**2 + dD_cmd**2) \
              + args.w_block*block_pen \
              + 1e6*box_violation + 1e3*rate_pen
            cost_terms.append(c)

        c_term = args.w_term*(theta-theta_star)**2
        return np.array(cost_terms + [c_term], float)

    try:
        from scipy.optimize import least_squares
        res = least_squares(lambda x: simulate(x), u, method="trf",
                            ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=int(args.max_nfev))
        u_opt = res.x
    except Exception:
        # fallback
        u_opt = u.copy()
        for _ in range(80):
            for i in range(len(u_opt)):
                h=1e-3
                base = simulate(u_opt).sum()
                u_opt[i]+=h; c_plus = simulate(u_opt).sum()
                u_opt[i]-=2*h; c_minus = simulate(u_opt).sum()
                u_opt[i]+=h; g=(c_plus-c_minus)/(2*h)
                u_opt[i]-=0.1*g

    ps0 = float(u_opt[0]); pd0=float(u_opt[1])
    p1 = 0.5*(ps0+pd0); p2=0.5*(ps0-pd0)
    p1,p2 = clamp_box(p1,p2,pmax)

    print("=== predictive inverse (Neural Hammerstein + delay) ===")
    print(f" delay d : {delay} steps (dt={dt:.4f}s -> {delay*dt*1e3:.0f} ms)")
    print(f" theta*  : {float(args.theta_deg):.3f} deg,  θ0: {float(args.theta0):.3f} deg")
    print(f" Sigma0  : {ps0:.4f} MPa,  Delta0: {pd0:.4f} MPa")
    print(f" -> p1   : {p1:.4f} MPa,  p2: {p2:.4f} MPa  (box [0,{pmax}])")

if __name__ == "__main__":
    main()
