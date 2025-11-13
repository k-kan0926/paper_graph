#!/usr/bin/env python3
"""
analyze_pressure_delay_v2.py
改良版: 診断情報を追加、閾値を自動調整
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def measure_delay_v2(csv_path, output_dir="delay_analysis_v2"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    t = df['t[s]'].values
    dt = np.median(np.diff(t))
    
    print(f"\n{'='*70}")
    print(f" 圧力遅延測定 (改良版)")
    print(f"{'='*70}")
    print(f"CSV: {csv_path}")
    print(f"Samples: {len(df)}, dt: {dt*1000:.2f} ms ({1/dt:.0f} Hz)")
    
    p1_cmd = df['p1_cmd[MPa]'].values
    p2_cmd = df['p2_cmd[MPa]'].values
    p1_meas = df['p1_meas[MPa]'].values
    p2_meas = df['p2_meas[MPa]'].values
    
    # --- データ品質チェック ---
    print(f"\n[Data Quality]")
    for name, data in [('p1_cmd', p1_cmd), ('p2_cmd', p2_cmd), 
                        ('p1_meas', p1_meas), ('p2_meas', p2_meas)]:
        print(f"  {name:12s}: range=[{data.min():.4f}, {data.max():.4f}] MPa, "
              f"std={data.std():.4f}, mean={data.mean():.4f}")
    
    # --- Method 1: 相互相関 (robust) ---
    print(f"\n[Method 1: Cross-Correlation]")
    def xcorr_delay_robust(cmd, meas, max_lag_sec=0.3):
        """ロバストな相互相関（ノイズ除去 + 最大ラグ制限）"""
        # ハイパスフィルタでDC除去
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 0.5 / (0.5/dt), btype='high')
        cmd_filt = filtfilt(b, a, cmd - cmd.mean())
        meas_filt = filtfilt(b, a, meas - meas.mean())
        
        corr = signal.correlate(meas_filt, cmd_filt, mode='full')
        lags = signal.correlation_lags(len(cmd), len(meas), mode='full')
        
        # 最大ラグ制限（±300ms）
        max_lag_samples = int(max_lag_sec / dt)
        valid_mask = np.abs(lags) <= max_lag_samples
        
        corr_valid = corr[valid_mask]
        lags_valid = lags[valid_mask]
        
        peak_idx = np.argmax(corr_valid)
        lag_samples = lags_valid[peak_idx]
        
        return lag_samples * dt, corr_valid[peak_idx] / (len(cmd) * cmd_filt.std() * meas_filt.std())
    
    delay_p1_corr, conf_p1 = xcorr_delay_robust(p1_cmd, p1_meas)
    delay_p2_corr, conf_p2 = xcorr_delay_robust(p2_cmd, p2_meas)
    
    print(f"  p1: {delay_p1_corr*1000:7.2f} ms  (confidence: {conf_p1:.3f})")
    print(f"  p2: {delay_p2_corr*1000:7.2f} ms  (confidence: {conf_p2:.3f})")
    
    # --- Method 2: ステップ応答 (adaptive threshold) ---
    print(f"\n[Method 2: Step Response (Adaptive)]")
    def step_response_delay_adaptive(cmd, meas, name=""):
        """適応的閾値でステップ検出"""
        d_cmd = np.abs(np.diff(cmd))
        
        # ステップサイズの分布を解析
        d_sorted = np.sort(d_cmd[d_cmd > 0.001])
        if len(d_sorted) < 10:
            print(f"    {name}: 警告 - ステップが少なすぎます（<10）")
            return np.nan, 0, []
        
        # 上位20%をステップとして検出（適応的）
        threshold = np.percentile(d_sorted, 80)
        step_indices = np.where(d_cmd > threshold)[0]
        
        print(f"    {name}: threshold={threshold:.4f} MPa → {len(step_indices)} steps detected")
        
        delays = []
        valid_steps = []
        for idx in step_indices[:30]:  # 最大30ステップ解析
            if idx + 1 >= len(cmd) or idx + 100 >= len(t):
                continue
            
            t_cmd = t[idx]
            val_before = cmd[idx]
            val_after = cmd[idx + 1]
            step_size = abs(val_after - val_before)
            
            if step_size < 0.02:  # 20kPa未満は無視
                continue
            
            target = 0.5 * (val_before + val_after)
            
            # 探索窓: ±300ms
            search_mask = (t > t_cmd) & (t < t_cmd + 0.3)
            if not search_mask.any():
                continue
            
            if val_after > val_before:
                cross_idx = np.where(search_mask & (meas > target))[0]
            else:
                cross_idx = np.where(search_mask & (meas < target))[0]
            
            if len(cross_idx) > 0:
                t_meas = t[cross_idx[0]]
                delay = t_meas - t_cmd
                
                # 異常値除外（0-200ms範囲外）
                if 0.0 < delay < 0.2:
                    delays.append(delay)
                    valid_steps.append({'idx': idx, 'delay': delay, 'size': step_size})
        
        if len(delays) < 3:
            print(f"    {name}: 警告 - 有効ステップ不足（{len(delays)} < 3）")
            return np.nan, len(delays), []
        
        # 外れ値除去（中央値±2σ）
        delays_arr = np.array(delays)
        median = np.median(delays_arr)
        mad = np.median(np.abs(delays_arr - median))
        robust_std = 1.4826 * mad
        
        inliers = np.abs(delays_arr - median) < 2 * robust_std
        delays_clean = delays_arr[inliers]
        
        print(f"    {name}: {len(delays_clean)}/{len(delays)} steps used (outliers removed)")
        
        return np.median(delays_clean), len(delays_clean), valid_steps
    
    delay_p1_step, n1, steps_p1 = step_response_delay_adaptive(p1_cmd, p1_meas, "p1")
    delay_p2_step, n2, steps_p2 = step_response_delay_adaptive(p2_cmd, p2_meas, "p2")
    
    print(f"  Final:")
    print(f"    p1: {delay_p1_step*1000:7.2f} ms  (n={n1})")
    print(f"    p2: {delay_p2_step*1000:7.2f} ms  (n={n2})")
    
    # --- 結果統合 (weighted average) ---
    print(f"\n[Result Integration]")
    
    # 信頼度に基づく重み付け平均
    results = []
    weights = []
    
    # 相互相関の重み: confidence × 1.0
    if conf_p1 > 0.3:
        results.append(delay_p1_corr)
        weights.append(conf_p1 * 1.0)
        print(f"  p1_corr: {delay_p1_corr*1000:.2f} ms  (weight: {conf_p1*1.0:.2f})")
    
    if conf_p2 > 0.3:
        results.append(delay_p2_corr)
        weights.append(conf_p2 * 1.0)
        print(f"  p2_corr: {delay_p2_corr*1000:.2f} ms  (weight: {conf_p2*1.0:.2f})")
    
    # ステップ応答の重み: (n_steps / 20) × 1.5（より信頼）
    if np.isfinite(delay_p1_step) and n1 >= 3:
        w = min(1.0, n1 / 20.0) * 1.5
        results.append(delay_p1_step)
        weights.append(w)
        print(f"  p1_step: {delay_p1_step*1000:.2f} ms  (weight: {w:.2f})")
    
    if np.isfinite(delay_p2_step) and n2 >= 3:
        w = min(1.0, n2 / 20.0) * 1.5
        results.append(delay_p2_step)
        weights.append(w)
        print(f"  p2_step: {delay_p2_step*1000:.2f} ms  (weight: {w:.2f})")
    
    if not results:
        raise ValueError("全ての測定手法が失敗しました")
    
    delay_weighted = np.average(results, weights=weights)
    delay_std = np.sqrt(np.average((np.array(results) - delay_weighted)**2, weights=weights))
    
    # ステップ数換算
    n_steps_200Hz = int(np.round(delay_weighted / 0.005))
    n_steps_100Hz = int(np.round(delay_weighted / 0.010))
    
    print(f"\n{'='*70}")
    print(f" 【最終推奨値】")
    print(f"{'='*70}")
    print(f"  加重平均遅延: {delay_weighted*1000:.2f} ± {delay_std*1000:.2f} ms")
    print(f"  訓練時 (200Hz): --delay {n_steps_200Hz}")
    print(f"  制御時: pressure_delay_s:={delay_weighted:.6f}")
    print(f"{'='*70}\n")
    
    # --- 詳細可視化 ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Row 1: 全体像（最初の5秒）
    mask_full = t < 5.0
    
    ax = axes[0, 0]
    ax.plot(t[mask_full], p1_cmd[mask_full], 'b-', label='p1_cmd', linewidth=1.5, alpha=0.8)
    ax.plot(t[mask_full], p1_meas[mask_full], 'r-', label='p1_meas', linewidth=1.2)
    ax.axvline(delay_weighted, color='g', linestyle='--', linewidth=2, label=f'delay={delay_weighted*1000:.1f}ms')
    ax.set_ylabel('Pressure [MPa]')
    ax.set_title('p1: Overview (first 5s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(t[mask_full], p2_cmd[mask_full], 'b-', label='p2_cmd', linewidth=1.5, alpha=0.8)
    ax.plot(t[mask_full], p2_meas[mask_full], 'r-', label='p2_meas', linewidth=1.2)
    ax.axvline(delay_weighted, color='g', linestyle='--', linewidth=2, label=f'delay={delay_weighted*1000:.1f}ms')
    ax.set_ylabel('Pressure [MPa]')
    ax.set_title('p2: Overview (first 5s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 2: 拡大図（典型的なステップ）
    if steps_p1:
        step = steps_p1[len(steps_p1)//2]  # 中央のステップ
        idx = step['idx']
        t_window = t[max(0, idx-20):min(len(t), idx+80)]
        p1_cmd_win = p1_cmd[max(0, idx-20):min(len(t), idx+80)]
        p1_meas_win = p1_meas[max(0, idx-20):min(len(t), idx+80)]
        
        ax = axes[1, 0]
        ax.plot(t_window - t[idx], p1_cmd_win, 'b-', label='p1_cmd', linewidth=2)
        ax.plot(t_window - t[idx], p1_meas_win, 'r-', label='p1_meas', linewidth=1.5)
        ax.axvline(0, color='b', linestyle=':', alpha=0.5, label='cmd step')
        ax.axvline(step['delay'], color='r', linestyle=':', alpha=0.5, label=f"meas @ {step['delay']*1000:.1f}ms")
        ax.set_xlabel('Time relative to step [s]')
        ax.set_ylabel('Pressure [MPa]')
        ax.set_title(f'p1: Typical Step (size={step["size"]:.3f} MPa)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if steps_p2:
        step = steps_p2[len(steps_p2)//2]
        idx = step['idx']
        t_window = t[max(0, idx-20):min(len(t), idx+80)]
        p2_cmd_win = p2_cmd[max(0, idx-20):min(len(t), idx+80)]
        p2_meas_win = p2_meas[max(0, idx-20):min(len(t), idx+80)]
        
        ax = axes[1, 1]
        ax.plot(t_window - t[idx], p2_cmd_win, 'b-', label='p2_cmd', linewidth=2)
        ax.plot(t_window - t[idx], p2_meas_win, 'r-', label='p2_meas', linewidth=1.5)
        ax.axvline(0, color='b', linestyle=':', alpha=0.5)
        ax.axvline(step['delay'], color='r', linestyle=':', alpha=0.5, label=f"{step['delay']*1000:.1f}ms")
        ax.set_xlabel('Time relative to step [s]')
        ax.set_ylabel('Pressure [MPa]')
        ax.set_title(f'p2: Typical Step (size={step["size"]:.3f} MPa)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 3: 遅延分布のヒストグラム
    if steps_p1:
        delays_p1 = [s['delay']*1000 for s in steps_p1]
        ax = axes[2, 0]
        ax.hist(delays_p1, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(delay_p1_step*1000 if np.isfinite(delay_p1_step) else delay_weighted*1000, 
                   color='r', linestyle='--', linewidth=2, label='median')
        ax.set_xlabel('Delay [ms]')
        ax.set_ylabel('Count')
        ax.set_title(f'p1: Delay Distribution (n={len(delays_p1)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if steps_p2:
        delays_p2 = [s['delay']*1000 for s in steps_p2]
        ax = axes[2, 1]
        ax.hist(delays_p2, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(delay_p2_step*1000 if np.isfinite(delay_p2_step) else delay_weighted*1000, 
                   color='r', linestyle='--', linewidth=2, label='median')
        ax.set_xlabel('Delay [ms]')
        ax.set_ylabel('Count')
        ax.set_title(f'p2: Delay Distribution (n={len(delays_p2)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'pressure_delay_detailed.png')
    plt.savefig(plot_path, dpi=150)
    print(f"[Saved] {plot_path}")
    
    # JSON保存
    import json
    result = {
        'delay_p1_corr': float(delay_p1_corr),
        'delay_p2_corr': float(delay_p2_corr),
        'delay_p1_step': float(delay_p1_step) if np.isfinite(delay_p1_step) else None,
        'delay_p2_step': float(delay_p2_step) if np.isfinite(delay_p2_step) else None,
        'delay_weighted': float(delay_weighted),
        'delay_std': float(delay_std),
        'n_steps_p1': int(n1),
        'n_steps_p2': int(n2),
        'n_steps_200Hz': n_steps_200Hz,
        'n_steps_100Hz': n_steps_100Hz,
        'confidence_p1_corr': float(conf_p1),
        'confidence_p2_corr': float(conf_p2),
        'dt_sampling': float(dt)
    }
    
    json_path = os.path.join(output_dir, 'delay_measurement_v2.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[Saved] {json_path}\n")
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_pressure_delay_v2.py <delay_measurement.csv>")
        sys.exit(1)
    
    measure_delay_v2(sys.argv[1])