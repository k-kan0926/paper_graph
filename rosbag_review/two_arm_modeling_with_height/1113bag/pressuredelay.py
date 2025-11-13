#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy.signal import correlate

def estimate_delay_one(csv_path,
                       col_cmd="p2_cmd[MPa]",
                       col_meas="p2_meas[MPa]"):
    df = pd.read_csv(csv_path).dropna()
    t = df["t[s]"].to_numpy()
    dt = np.median(np.diff(t))

    x = df[col_cmd].to_numpy()
    y = df[col_meas].to_numpy()

    # DC 成分を抜いておくと相互相関が安定しやすい
    x = x - np.mean(x)
    y = y - np.mean(y)

    corr = correlate(y, x, mode="full")
    lag_idx = np.argmax(corr) - (len(x) - 1)
    delay_s = lag_idx * dt

    return lag_idx, delay_s, dt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="delay_measurement の CSV")
    args = ap.parse_args()


    lag, delay_s, dt = estimate_delay_one(args.csv)
    print(f"dt ≈ {dt*1000:.2f} ms")
    print(f"lag_samples = {lag}")
    print(f"pressure_delay_s ≈ {delay_s*1000:.1f} ms")
