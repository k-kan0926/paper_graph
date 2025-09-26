#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuro-network based dynamic model fit on top of mono (tanh) static model.
Model predicts theta at t+1 based on inputs at t-d and current state.

Model: θ(t+dt) = NN(θ(t), Σ(t-d), Δ(t-d), Σ̇(t-d), Δ̇(t-d), ...)

Usage:
  python fit4a_theta_nn.py --csv out/diff_run1_h_data.csv --out-prefix out/model_nn/model_nn
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

def smooth_and_deriv(x, dt, win=17, poly=3):
    """
    Smooths data and calculates its derivative using Savitzky-Golay filter.
    """
    x = np.asarray(x, float)
    N = len(x)
    win = min(win, (N // 2) * 2 - 1)
    if win < 5:
        return x, np.gradient(x, dt)
    try:
        from scipy.signal import savgol_filter
        xs = savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
        dx = savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=dt, mode="interp")
        return xs, dx
    except Exception:
        k = win
        w = np.ones(k) / k
        xs = np.convolve(x, w, mode="same")
        dx = np.gradient(xs, dt)
        return xs, dx

def build_dataset_for_nn(ps_s, pd_s, th_s, dps, dpd, d, dt, lookback=1):
    """
    Prepares time-series data for a neural network.
    Returns X (features) and y (target).
    """
    N = len(ps_s)
    start = d + 1
    end = N
    X, y = [], []

    for k in range(start, end):
        # Build feature vector at time k-1, using inputs at time k-1-d
        ku = k - 1 - d
        kup = k - 1 - d - 1
        
        if kup < 0:
            continue
        
        # Features: current theta, current and past inputs
        features = [
            th_s[k-1],
            ps_s[ku],
            pd_s[ku],
            (ps_s[ku] - ps_s[kup]) / dt,
            (pd_s[ku] - pd_s[kup]) / dt,
        ]
        
        # Target: next theta
        target = th_s[k]
        
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y)

def create_nn_model(input_dim):
    """
    Creates a simple neural network model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1) # Output: theta at the next step
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--sg-win", type=int, default=17)
    ap.add_argument("--sg-poly", type=int, default=3)
    ap.add_argument("--delay", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    df = pd.read_csv(args.csv, comment="#")
    ps = df["p_sum[MPa]"].to_numpy(float)
    pdiff = df["p_diff[MPa]"].to_numpy(float)
    theta = df["theta[deg]"].to_numpy(float)

    if args.dt is not None:
        dt = float(args.dt)
    elif "time[s]" in df.columns:
        t = df["time[s]"].to_numpy(float)
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = 0.01

    # Pre-process data
    ps_s, dps = smooth_and_deriv(ps, dt, args.sg_win, args.sg_poly)
    pd_s, dpd = smooth_and_deriv(pdiff, dt, args.sg_win, args.sg_poly)
    th_s, dth = smooth_and_deriv(theta, dt, args.sg_win, args.sg_poly)

    # Build dataset
    X, y = build_dataset_for_nn(ps_s, pd_s, th_s, dps, dpd, args.delay, dt)

    # Normalize data
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Create and train the model
    input_dim = X_train.shape[1]
    model = create_nn_model(input_dim)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping], 
        verbose=1
    )

    # Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Train Loss (MSE): {train_loss:.4f}")
    print(f"Test Loss (MSE): {test_loss:.4f}")

    # Save the model and scalers
    out_dir = os.path.dirname(args.out_prefix)
    model.save(os.path.join(out_dir, "nn_model.h5"))
    np.savez(os.path.join(out_dir, "nn_scalers.npz"), 
             X_min=X_scaler.data_min_, X_max=X_scaler.data_max_,
             y_min=y_scaler.data_min_, y_max=y_scaler.data_max_)
    print(f"[OK] saved model and scalers to {out_dir}")

if __name__ == "__main__":
    main()