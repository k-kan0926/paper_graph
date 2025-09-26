#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short-horizon predictive inverse (MPC) for Hammerstein with input delay d.

Dynamics (Euler, with delay):
  applied inputs at stage k are u_applied[k] = history[k]  (k<d)
                                      or   = decisions[k-d] (k>=d)
  θ_{k+1} = θ_k + dt*[ α*(θ_stat(applied_k) - θ_k)
                     + kΣ*(ΔΣ_applied_k/dt) + kΔ*(ΔΔ_applied_k/dt) ]

Cost:
  sum_k w_stage*(θ_k - θ*)^2 + w_z*z(Σ_k,Δ_k)^2 + w_sigma*(Σ_k - sigma_ref)^2
        + w_rate*(ΔΣ_cmd_k^2 + ΔΔ_cmd_k^2) + w_block*block_penalty
  + w_term*(θ_H - θ*)^2
Hard constraints via penalties:
  commands (p1,p2 from Σ_k,Δ_k) are softly penalized if outside [0, pmax]
  and soft rate limits |ΔΣ_cmd|,|ΔΔ_cmd| <= rate limits.

History:
  If --hist-sigma/--hist-delta are not given, history of length d
  is filled with (sigma0, delta0).

Example:
  python inverse3c_predictive_hammerstein_delay_mpc.py --model out/model_k3c/model_k3c_hamm_model.npz --theta-deg 20 --theta0 12.3 --H 10 --wz 0.3 --w-sigma 0.02 --w-rate 0.3 --w-block 10.0 --rate-sigma 0.12 --rate-delta 0.12
"""
import argparse, numpy as np

def poly_eval(ps, coef):
    ps = np.asarray(ps, float)
    out = np.zeros_like(ps, float); p = np.ones_like(ps, float)
    for c in coef:
        out += c*p; p *= ps
    return out

def theta_stat(ps, pd, a,b,c):
    A = poly_eval(ps, a)
    B = np.exp(poly_eval(ps, b))
    C = np.exp(poly_eval(ps, c))
    return A + B*np.tanh(C*pd)

def z_pred(ps, pd, zc):
    if zc is None or zc.size < 4:
        return np.zeros_like(np.asarray(ps, float), float)
    return zc[0] + zc[1]*ps + zc[2]*pd + zc[3]*ps*pd

def invert_delta_closed(ps, theta_star, a,b,c):
    A = poly_eval(ps, a)
    B = np.maximum(np.exp(poly_eval(ps, b)), 1e-8)
    C = np.maximum(np.exp(poly_eval(ps, c)), 1e-8)
    x = np.clip((theta_star - A)/B, -0.999, 0.999)
    pd = np.arctanh(x)/C
    return float(pd)

def clamp_box(p1, p2, pmax):
    return np.clip(p1,0,pmax), np.clip(p2,0,pmax)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="*_hamm_model.npz")
    ap.add_argument("--theta-deg", type=float, required=True)
    ap.add_argument("--theta0", type=float, required=True, help="current θ [deg]")
    ap.add_argument("--sigma0", type=float, default=None, help="previous Σ; default sigma_ref")
    ap.add_argument("--delta0", type=float, default=None, help="previous Δ; default from static inverse at θ0")
    ap.add_argument("--hist-sigma", type=str, default=None, help="comma-separated Σ history (length d)")
    ap.add_argument("--hist-delta", type=str, default=None, help="comma-separated Δ history (length d)")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--w-stage", type=float, default=1.0)
    ap.add_argument("--w-term", type=float, default=5.0)
    ap.add_argument("--wz", type=float, default=0.2)
    ap.add_argument("--w-sigma", type=float, default=0.02)
    ap.add_argument("--w-rate", type=float, default=0.3)
    ap.add_argument("--w-block", type=float, default=10.0, help="move-blocking: encourage u_k == u_{k-1} every block step")
    ap.add_argument("--block", type=int, default=2, help="block size (>=1). If 1, no blocking penalty is added.")
    ap.add_argument("--rate-sigma", type=float, default=0.12, help="|ΔΣ| max per step (MPa/step), soft")
    ap.add_argument("--rate-delta", type=float, default=0.12, help="|ΔΔ| max per step (MPa/step), soft")
    ap.add_argument("--max-nfev", type=int, default=600)
    args = ap.parse_args()

    D = np.load(args.model, allow_pickle=True)
    a,b,c = D["a_coef"], D["b_coef"], D["c_coef"]
    zc = D["z_coef"]
    alpha = float(D["alpha"]); kS = float(D["kSigma"]); kD = float(D["kDelta"])
    dt = float(D["dt"]); pmax = float(D["pmax"]); sigma_ref = float(D["sigma_ref"])
    delay = int(D["delay"]) if "delay" in D.files else 0

    theta_star = float(args.theta_deg)
    theta0 = float(args.theta0)
    sigma0 = float(args.sigma0) if args.sigma0 is not None else sigma_ref
    if args.delta0 is not None:
        delta0 = float(args.delta0)
    else:
        delta0 = invert_delta_closed(sigma0, theta0, a,b,c)
        # box projection
        p1 = 0.5*(sigma0+delta0); p2=0.5*(sigma0-delta0)
        p1,p2 = clamp_box(p1,p2,pmax); sigma0 = p1+p2; delta0 = p1-p2

    # build history (length = delay)
    if delay > 0:
        if args.hist_sigma:
            hist_sigma = [float(x) for x in args.hist_sigma.split(",")]
        else:
            hist_sigma = [sigma0]*delay
        if args.hist_delta:
            hist_delta = [float(x) for x in args.hist_delta.split(",")]
        else:
            hist_delta = [delta0]*delay
        if len(hist_sigma)!=delay or len(hist_delta)!=delay:
            raise ValueError("hist-sigma / hist-delta must have length == delay")
    else:
        hist_sigma, hist_delta = [], []

    # decision variables: u = [Σ_0,Δ_0, Σ_1,Δ_1, ..., Σ_{H-1},Δ_{H-1}]
    H = int(args.H)
    u = np.zeros(2*H, float)
    # initialize with static inverse of θ* and Σ=sigma_ref
    for k in range(H):
        ps = sigma_ref
        pd = invert_delta_closed(ps, theta_star, a,b,c)
        p1 = 0.5*(ps+pd); p2=0.5*(ps-pd); p1,p2 = clamp_box(p1,p2,pmax)
        u[2*k+0] = p1+p2
        u[2*k+1] = p1-p2

    rateS = float(args.rate_sigma)
    rateD = float(args.rate_delta)
    block = max(1, int(args.block))

    def simulate(u_vec):
        # applied sequence seen by plant (length delay+H)
        sig_cmd = np.r_[sigma0, u_vec[0::2]]  # for command-rate penalty ref
        del_cmd = np.r_[delta0, u_vec[1::2]]

        sig_applied = np.r_[hist_sigma, u_vec[0::2]]  # length delay+H
        del_applied = np.r_[hist_delta, u_vec[1::2]]

        theta = theta0
        cost_terms = []

        for k in range(H):
            # applied (felt) at this stage
            ps_eff = sig_applied[k]        # history if k<delay else u_{k-delay}
            pd_eff = del_applied[k]

            # dynamics rates on applied signals
            if k == 0:
                ps_eff_prev = (hist_sigma[-1] if delay>0 else sigma0)
                pd_eff_prev = (hist_delta[-1] if delay>0 else delta0)
            else:
                ps_eff_prev = sig_applied[k-1]
                pd_eff_prev = del_applied[k-1]

            dS_eff = (ps_eff - ps_eff_prev) / dt
            dD_eff = (pd_eff - pd_eff_prev) / dt

            # Euler step
            ths = theta_stat(ps_eff, pd_eff, a,b,c)
            theta = theta + dt*( alpha*(ths - theta) + kS*dS_eff + kD*dD_eff )

            # command at this stage (for penalties/box/z)
            ps_cmd = u_vec[2*k+0]; pd_cmd = u_vec[2*k+1]
            p1 = 0.5*(ps_cmd+pd_cmd); p2 = 0.5*(ps_cmd-pd_cmd)
            box_violation = 0.0
            if p1<0: box_violation += -p1
            if p2<0: box_violation += -p2
            if p1>pmax: box_violation += (p1-pmax)
            if p2>pmax: box_violation += (p2-pmax)

            # command-rate (soft) limits
            if k == 0:
                dS_cmd = ps_cmd - sigma0
                dD_cmd = pd_cmd - delta0
            else:
                dS_cmd = ps_cmd - u_vec[2*(k-1)+0]
                dD_cmd = pd_cmd - u_vec[2*(k-1)+1]
            rate_pen = max(0.0, abs(dS_cmd)-rateS) + max(0.0, abs(dD_cmd)-rateD)

            # move-blocking penalty (encourage piecewise-constant inputs)
            block_pen = 0.0
            if block > 1 and (k % block != 0):
                ps_prev = u_vec[2*(k-1)+0]
                pd_prev = u_vec[2*(k-1)+1]
                block_pen = (ps_cmd-ps_prev)**2 + (pd_cmd-pd_prev)**2

            # stage costs
            c_stage = args.w_stage*(theta - theta_star)**2 \
                    + args.wz*(z_pred(ps_cmd,pd_cmd,zc)**2) \
                    + args.w_sigma*((ps_cmd - sigma_ref)**2) \
                    + args.w_rate*(dS_cmd**2 + dD_cmd**2) \
                    + args.w_block*block_pen \
                    + 1e6*box_violation + 1e3*rate_pen
            cost_terms.append(c_stage)

        # terminal
        c_term = args.w_term*(theta - theta_star)**2
        return np.array(cost_terms + [c_term], float)

    try:
        from scipy.optimize import least_squares
        res = least_squares(lambda x: simulate(x), u, method="trf",
                            ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=int(args.max_nfev))
        u_opt = res.x
    except Exception:
        # simple coordinate-descent fallback
        u_opt = u.copy()
        for _ in range(80):
            base = simulate(u_opt).sum()
            for i in range(len(u_opt)):
                h = 1e-3
                u_opt[i] += h
                c_plus = simulate(u_opt).sum()
                u_opt[i] -= 2*h
                c_minus = simulate(u_opt).sum()
                u_opt[i] += h
                g = (c_plus - c_minus)/(2*h)
                u_opt[i] -= 0.1*g

    # first-step command
    ps0 = float(u_opt[0]); pd0 = float(u_opt[1])
    p1 = 0.5*(ps0+pd0); p2 = 0.5*(ps0-pd0)
    p1,p2 = clamp_box(p1,p2,pmax)

    print("=== predictive inverse (Hammerstein + delay) ===")
    print(f" delay d : {delay} steps (dt={dt:.4f}s -> {delay*dt*1e3:.0f} ms)")
    print(f" theta*  : {theta_star:.3f} deg,  θ0: {theta0:.3f} deg")
    print(f" Sigma0  : {ps0:.4f} MPa,  Delta0: {pd0:.4f} MPa")
    print(f" -> p1   : {p1:.4f} MPa,  p2: {p2:.4f} MPa  (box [0,{pmax}])")

if __name__ == "__main__":
    main()
