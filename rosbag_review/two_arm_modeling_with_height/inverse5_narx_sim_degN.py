
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NARX共通NPZ（任意次数の PolynomialFeatures に対応）を使った簡易逆写像シミュレーション

重要: poly_feature_names に三次以上の項（例: "x0^2 x1" や "x0 x1 x2"）が含まれてもOK。
フィーチャ名を字句分解し、積を評価します。

使い方例:
  python narx_inverse_sim_degN.py \\
    --npz /home/keiichiro/documents/paper_graph/rosbag_review/two_arm_modeling_with_height/out/bench_phase3_narx_z/lag2_deg3/narx_common_meta.npz \\
    --target 20 --T 300 --w-theta 1.0 --w-z 0.2 --w-rate 0.05 --p-lo 0.0 --p-hi 1.0
"""
import argparse, numpy as np
import matplotlib.pyplot as plt

def _eval_feature_from_tokens(name: str, xmap: dict) -> float:
    if name == '1':
        return 1.0
    prod = 1.0
    for tok in name.split(' '):
        tok = tok.strip()
        if not tok:
            continue
        if '^' in tok:
            base, p = tok.split('^', 1)
            prod *= xmap[base] ** int(p)
        else:
            prod *= xmap[tok]
    return float(prod)

def build_poly_from_vector(x_vec, names):
    xmap = {f'x{i}': x_vec[i] for i in range(len(x_vec))}
    return np.asarray([_eval_feature_from_tokens(nm, xmap) for nm in names], dtype=float)

def narx_predict_next(theta_hist, ps_hist, pd_hist, meta):
    lag_y, lag_u, delay = int(meta["lag_y"]), int(meta["lag_u"]), int(meta["delay"])
    coef, names = meta["coef"], meta["poly_feature_names"]
    x_vec = []
    for k in range(1, lag_y+1):
        x_vec.append(theta_hist[-k])
    for k in range(delay, delay+lag_u):
        x_vec.extend([ps_hist[-1-k], pd_hist[-1-k]])
    return float(np.dot(coef, build_poly_from_vector(np.array(x_vec, float), names)))

def z_cost(ps, pd, z_coef, z_names):
    vals = {"1":1.0, "ps":ps, "pd":pd, "ps*pd":ps*pd, "ps^2":ps*ps, "pd^2":pd*pd}
    return float(sum(z_coef[i]*vals[z_names[i]] for i in range(len(z_names))))

def receding_inverse_step(theta_hist, ps_hist, pd_hist, meta, target_deg, w_theta=1.0, w_z=0.0, w_rate=0.0, p_bounds=(0.0,1.0)):
    lag_u, delay = int(meta["lag_u"]), int(meta["delay"])
    p_lo, p_hi = p_bounds
    ps_prev, pd_prev = ps_hist[-1], pd_hist[-1]
    best = None
    for n in (25, 13):
        cand = np.linspace(p_lo, p_hi, n)
        for ps in cand:
            for pd in cand:
                th_tmp = list(theta_hist); ps_tmp = list(ps_hist); pd_tmp = list(pd_hist)
                for _ in range(delay+1):
                    ps_tmp.append(ps); pd_tmp.append(pd)
                    th_tmp.append(narx_predict_next(th_tmp, ps_tmp, pd_tmp, meta))
                y_pred = th_tmp[-1]
                J = w_theta*(y_pred-target_deg)**2 + w_z*z_cost(ps,pd,meta["z_coef"],meta["z_feat_names"]) + w_rate*((ps-ps_prev)**2+(pd-pd_prev)**2)
                if (best is None) or (J < best[0]):
                    best = (J, ps, pd, y_pred)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--target", type=float, default=20.0)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--w-theta", type=float, default=1.0)
    ap.add_argument("--w-z", type=float, default=0.2)
    ap.add_argument("--w-rate", type=float, default=0.05)
    ap.add_argument("--p-lo", type=float, default=0.0)
    ap.add_argument("--p-hi", type=float, default=1.0)
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    meta = {k: npz[k].tolist() if k in ["model","theta_unit"] else npz[k] for k in npz.files}
    meta["coef"] = np.asarray(meta.get("coef"), float)
    meta["poly_feature_names"] = [str(s) for s in meta.get("poly_feature_names", [])]
    meta["z_coef"] = np.asarray(meta.get("z_coef"), float)
    meta["z_feat_names"] = [str(s) for s in meta.get("z_feat_names", [])]

    dt = float(npz["dt"])
    lag_y, lag_u, delay = int(meta["lag_y"]), int(meta["lag_u"]), int(meta["delay"])

    theta_hist = [0.0]*max(lag_y,1)
    ps_hist = [0.0]*(delay + lag_u + 5)
    pd_hist = [0.0]*(delay + lag_u + 5)

    theta = [theta_hist[-1]]
    ps_log, pd_log, cost_log = [], [], []

    for t in range(args.T):
        J, ps_t, pd_t, y_pred = receding_inverse_step(theta_hist, ps_hist, pd_hist, meta, args.target,
                                                      w_theta=args.w_theta, w_z=args.w_z, w_rate=args.w_rate,
                                                      p_bounds=(args.p_lo, args.p_hi))
        ps_hist.append(ps_t); pd_hist.append(pd_t)
        theta_next = narx_predict_next(theta_hist, ps_hist, pd_hist, meta)
        theta_hist.append(theta_next); theta.append(theta_next)
        ps_log.append(ps_t); pd_log.append(pd_t); cost_log.append(J)

    time = np.arange(len(theta))*dt

    plt.figure(figsize=(8,4.5))
    plt.plot(time, theta, label="theta [deg]")
    plt.plot(time, np.ones_like(time)*args.target, linestyle="--", label="target [deg]")
    plt.xlabel("time [s]"); plt.ylabel("theta [deg]")
    plt.title("Inverse simulation with NARX (degree-agnostic)")
    plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,4.5))
    plt.plot(time[1:], ps_log, label="ps [MPa]")
    plt.plot(time[1:], pd_log, label="pd [MPa]")
    plt.xlabel("time [s]"); plt.ylabel("pressure [MPa]")
    plt.title("Commanded pressures"); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,4.5))
    plt.plot(time[1:], cost_log, label="stage cost")
    plt.xlabel("time [s]"); plt.ylabel("cost")
    plt.title("Cost per step"); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
