#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short-horizon predictive inverse for Hammerstein model with box/rate constraints.

Dynamics (Euler):
  θ_{k+1} = θ_k + dt*[ α*(θ_stat(Σ_k,Δ_k) - θ_k) + kΣ*(Σ_k-Σ_{k-1})/dt + kΔ*(Δ_k-Δ_{k-1})/dt ]

Cost:
  sum_k w_stage*(θ_k - θ*)^2 + w_z*z(Σ_k,Δ_k)^2 + w_sigma*(Σ_k - sigma_ref)^2
  + w_rate*(ΔΣ_k^2 + ΔΔ_k^2) + w_term*(θ_H - θ*)^2
Hard constraints via penalties:
  p1,p2 in [0,pmax], |ΔΣ|,|ΔΔ| <= rate limits (softly enforced)

Outputs first-step command (p1,p2).
python inverse3_predictive_hammerstein.py --model out/model_k3/model_k3_hamm_model.npz --theta-deg 20 --theta0 12.3 --H 5 --wz 0.3 --w-sigma 0.02 --w-rate 0.1

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
    if zc is None or zc.size==0: return np.zeros_like(np.asarray(ps), float)
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
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--w-stage", type=float, default=1.0)
    ap.add_argument("--w-term", type=float, default=5.0)
    ap.add_argument("--wz", type=float, default=0.2)
    ap.add_argument("--w-sigma", type=float, default=0.02)
    ap.add_argument("--w-rate", type=float, default=0.1)
    ap.add_argument("--rate-sigma", type=float, default=None, help="|ΔΣ| max per step (MPa/step), soft")
    ap.add_argument("--rate-delta", type=float, default=None, help="|ΔΔ| max per step (MPa/step), soft")
    args = ap.parse_args()

    D = np.load(args.model, allow_pickle=True)
    a,b,c = D["a_coef"], D["b_coef"], D["c_coef"]
    zc = D["z_coef"]
    alpha = float(D["alpha"]); kS = float(D["kSigma"]); kD = float(D["kDelta"])
    dt = float(D["dt"]); pmax = float(D["pmax"]); sigma_ref = float(D["sigma_ref"])

    theta_star = float(args.theta_deg)
    theta0 = float(args.theta0)
    sigma0 = float(args.sigma0) if args.sigma0 is not None else sigma_ref
    if args.delta0 is not None:
        delta0 = float(args.delta0)
    else:
        # static inverse at current θ0 as a starting point
        delta0 = invert_delta_closed(sigma0, theta0, a,b,c)
        # project into box via p1,p2
        p1 = 0.5*(sigma0+delta0); p2=0.5*(sigma0-delta0)
        p1,p2 = clamp_box(p1,p2,pmax); sigma0 = p1+p2; delta0 = p1-p2

    # decision variables: u = [Σ_0,Δ_0, Σ_1,Δ_1, ..., Σ_{H-1},Δ_{H-1}]
    H = int(args.H)
    u = np.zeros(2*H, float)
    # initialize with static inverse of θ* and Σ=sigma_ref
    for k in range(H):
        ps = sigma_ref
        pd = invert_delta_closed(ps, theta_star, a,b,c)
        # box projection
        p1 = 0.5*(ps+pd); p2=0.5*(ps-pd); p1,p2 = clamp_box(p1,p2,pmax)
        u[2*k+0] = p1+p2
        u[2*k+1] = p1-p2

    rateS = args.rate_sigma if args.rate_sigma is not None else 0.15   # MPa/step (example)
    rateD = args.rate_delta if args.rate_delta is not None else 0.15

    def simulate(u_vec):
        theta = theta0
        sig_prev, del_prev = sigma0, delta0
        cost_terms = []
        for k in range(H):
            ps = u_vec[2*k+0]; pd = u_vec[2*k+1]
            # penalties for box
            p1 = 0.5*(ps+pd); p2 = 0.5*(ps-pd)
            box_violation = 0.0
            if p1<0: box_violation += -p1
            if p2<0: box_violation += -p2
            if p1>pmax: box_violation += (p1-pmax)
            if p2>pmax: box_violation += (p2-pmax)

            # rates
            dS = ps - (sig_prev if k==0 else u_vec[2*(k-1)+0])
            dD = pd - (del_prev if k==0 else u_vec[2*(k-1)+1])
            rate_pen = max(0.0, abs(dS)-rateS) + max(0.0, abs(dD)-rateD)

            # dynamics step
            ths = theta_stat(ps, pd, a,b,c)
            theta = theta + dt*( alpha*(ths - theta) + kS*(dS/dt) + kD*(dD/dt) )

            # stage costs
            c_stage = args.w_stage*(theta - theta_star)**2 \
                    + args.wz*(z_pred(ps,pd,zc)**2) \
                    + args.w_sigma*((ps - sigma_ref)**2) \
                    + args.w_rate*(dS**2 + dD**2) \
                    + 1e6*box_violation + 1e3*rate_pen
            cost_terms.append(c_stage)

        # terminal
        c_term = args.w_term*(theta - theta_star)**2
        return np.array(cost_terms + [c_term], float)

    try:
        from scipy.optimize import least_squares
        res = least_squares(lambda x: simulate(x), u, method="trf", ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=400)
        u_opt = res.x
    except Exception:
        # simple local improvement (coordinate descent-ish)
        u_opt = u.copy()
        for _ in range(60):
            grads = []
            base = simulate(u_opt).sum()
            for i in range(len(u_opt)):
                h = 1e-3
                u_opt[i] += h
                c_plus = simulate(u_opt).sum()
                u_opt[i] -= 2*h
                c_minus = simulate(u_opt).sum()
                u_opt[i] += h
                g = (c_plus - c_minus)/(2*h)
                grads.append(g)
            u_opt -= 0.1*np.array(grads)

    # first-step command
    ps0 = float(u_opt[0]); pd0 = float(u_opt[1])
    p1 = 0.5*(ps0+pd0); p2 = 0.5*(ps0-pd0)
    p1,p2 = clamp_box(p1,p2,pmax)

    print("=== predictive inverse (Hammerstein) ===")
    print(f" theta* : {theta_star:.3f} deg,  θ0: {theta0:.3f} deg")
    print(f" Sigma0 : {ps0:.4f} MPa,  Delta0: {pd0:.4f} MPa")
    print(f" -> p1  : {p1:.4f} MPa,  p2: {p2:.4f} MPa  (box [0,{pmax}])")

if __name__ == "__main__":
    main()
