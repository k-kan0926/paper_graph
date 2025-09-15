#!/usr/bin/env python3
#python eval_theta_dyn_fit.py --csv out/diff_run1_data.csv --predictor out/theta_dyn/theta_dyn_predictor.py --out-prefix out/theta_dyn_eval/theta_dyn_eval
import argparse, os, importlib.util, numpy as np, pandas as pd, matplotlib.pyplot as plt

def load_predictor(py_path):
    spec = importlib.util.spec_from_file_location("pred", py_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

ap=argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--predictor", default="out/theta_dyn/theta_dyn_predictor.py")
ap.add_argument("--out-prefix", default="out/theta_dyn_eval/theta_dyn_eval")
args=ap.parse_args()
os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

df = pd.read_csv(args.csv, comment="#")
t  = df["t[s]"].to_numpy()
ps = df["p_sum[MPa]"].to_numpy()
pd = df["p_diff[MPa]"].to_numpy()
th = df["theta[rad]"].to_numpy()

pred = load_predictor(args.predictor)
dt = np.diff(t, prepend=t[0])
if hasattr(pred, "simulate"):
    th_hat = pred.simulate(th[0], ps, pd, dt)
elif hasattr(pred, "step"):
    th_hat = pred.step(th[0], ps, pd, dt)
else:
    th_hat = pred.f_static(ps, pd)  # 静的のみ
err = th - th_hat
rmse = float(np.sqrt(np.mean(err**2)))
mae  = float(np.mean(np.abs(err)))
print(f"[Eval] RMSE={rmse:.4f} rad, MAE={mae:.4f} rad")

# 図
plt.figure(figsize=(11,5))
plt.plot(t-t[0], np.degrees(th), label="theta [deg]")
plt.plot(t-t[0], np.degrees(th_hat), label="theta_hat [deg]")
plt.grid(True); plt.legend(); plt.xlabel("time [s]"); plt.tight_layout()
plt.savefig(args.out_prefix+"_timeseries.png", dpi=150)

plt.figure(figsize=(6,5))
plt.hist(np.degrees(err), bins=80)
plt.xlabel("residual [deg]"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(args.out_prefix+"_residual_hist.png", dpi=150)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.scatter(ps, np.degrees(err), s=4); plt.xlabel("p_sum [MPa]"); plt.ylabel("residual [deg]"); plt.grid(True)
plt.subplot(1,2,2); plt.scatter(pd, np.degrees(err), s=4); plt.xlabel("p_diff [MPa]"); plt.ylabel("residual [deg]"); plt.grid(True)
plt.tight_layout(); plt.savefig(args.out_prefix+"_residual_vs_p.png", dpi=150)
print("[OK] wrote eval plots")
