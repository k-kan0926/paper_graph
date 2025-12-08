#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dual_mppi_evaluation_v2.py
2系統MPPI制御システムの総合評価スクリプト（改良版）

改良点:
- ミラーセンサーの補正（theta_index, theta_index_4に-1を乗算）
- 各図を個別ファイルとして出力
- 時間範囲指定対応
- 時間軸を0始まりに正規化
"""

import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import os

# ============================================
# フォント設定（論文用）
# ============================================
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "pdf.use14corefonts": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

label_font = {'fontsize': 14, 'fontweight': 'bold'}
legend_font = {'fontsize': 10}

# ============================================
# パラメータ設定
# ============================================
BAG_FILE = '/home/keiichiro/documents/20251205/mppi.bag'
OUTPUT_DIR = './figures'  # 出力ディレクトリ

# センサーインデックス（コントローラーの設定に合わせる）
# System 1: theta_index_2 (0) = 正方向, theta_index (3) = ミラー
# System 2: theta_index_3 (1) = 正方向, theta_index_4 (2) = ミラー
THETA_INDEX_SYSTEM1_1 = 0   # theta_index_2 (正)
THETA_INDEX_SYSTEM1_2 = 3   # theta_index (ミラー → -1を乗算)
THETA_INDEX_SYSTEM2_1 = 1   # theta_index_3 (正)
THETA_INDEX_SYSTEM2_2 = 2   # theta_index_4 (ミラー → -1を乗算)

# 解析区間の設定（秒単位、Noneの場合は全区間）
TIME_OFFSET_START = 80  # 開始オフセット [s] (例: 10.0)
TIME_OFFSET_END = 230    # 終了オフセット [s] (例: 60.0)

# ============================================
# 出力ディレクトリ作成
# ============================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# データ読み込み
# ============================================
print("="*60)
print("Loading ROS bag...")
print("="*60)

bag = rosbag.Bag(BAG_FILE, 'r')
bag_start = bag.get_start_time()
bag_end = bag.get_end_time()

# 解析区間の決定
if TIME_OFFSET_START is not None and TIME_OFFSET_END is not None:
    analysis_start = bag_start + TIME_OFFSET_START
    analysis_end = bag_start + TIME_OFFSET_END
elif TIME_OFFSET_START is not None:
    analysis_start = bag_start + TIME_OFFSET_START
    analysis_end = bag_end
elif TIME_OFFSET_END is not None:
    analysis_start = bag_start
    analysis_end = bag_start + TIME_OFFSET_END
else:
    analysis_start = bag_start
    analysis_end = bag_end

start_time = rospy.Time(analysis_start)
end_time = rospy.Time(analysis_end)

print(f"Bag duration: {bag_end - bag_start:.2f}s")
print(f"Analysis range: {analysis_start - bag_start:.2f}s - {analysis_end - bag_start:.2f}s")
print(f"Analysis duration: {analysis_end - analysis_start:.2f}s")
print("="*60)

# データ格納用辞書
data = {
    # 時刻（0始まりに正規化）
    'time_target': [],
    'time_target_2': [],
    'time_cmd': [],
    'time_cmd_2': [],
    'time_joint': [],
    'time_odom': [],
    'time_pressure': [],
    'time_pid': [],
    
    # System 1
    'theta_target_1': [],
    'theta_cmd_1': [],
    'theta_actual_1_sensor1': [],  # 正方向
    'theta_actual_1_sensor2': [],  # ミラー補正後
    
    # System 2
    'theta_target_2': [],
    'theta_cmd_2': [],
    'theta_actual_2_sensor1': [],  # 正方向
    'theta_actual_2_sensor2': [],  # ミラー補正後
    
    # 圧力
    'p1': [], 'p2': [], 'p3': [], 'p4': [],
    
    # 姿勢
    'roll': [], 'pitch': [], 'yaw': [],
    
    # 位置
    'pos_x': [], 'pos_y': [], 'pos_z': [],
    
    # PID誤差
    'err_x': [], 'err_y': [], 'err_z': [],
}

# 目標角度 (System 1)
for topic, msg, t in bag.read_messages(topics=['/mppi/theta_target_deg'],
                                       start_time=start_time, end_time=end_time):
    data['time_target'].append(t.to_sec() - analysis_start)
    data['theta_target_1'].append(float(msg.data))

# 目標角度 (System 2)
for topic, msg, t in bag.read_messages(topics=['/mppi/theta_target_deg_2'],
                                       start_time=start_time, end_time=end_time):
    data['time_target_2'].append(t.to_sec() - analysis_start)
    data['theta_target_2'].append(float(msg.data))

# コマンド角度 (System 1)
for topic, msg, t in bag.read_messages(topics=['/mppi/target_deg_cmd'],
                                       start_time=start_time, end_time=end_time):
    data['time_cmd'].append(t.to_sec() - analysis_start)
    data['theta_cmd_1'].append(float(msg.data))

# コマンド角度 (System 2)
for topic, msg, t in bag.read_messages(topics=['/mppi/target_deg_cmd2'],
                                       start_time=start_time, end_time=end_time):
    data['time_cmd_2'].append(t.to_sec() - analysis_start)
    data['theta_cmd_2'].append(float(msg.data))

# 実角度（joint_states）- ミラー補正適用
for topic, msg, t in bag.read_messages(topics=['/kinikun1/joint_states'],
                                       start_time=start_time, end_time=end_time):
    if len(msg.position) > max(THETA_INDEX_SYSTEM1_1, THETA_INDEX_SYSTEM1_2,
                                 THETA_INDEX_SYSTEM2_1, THETA_INDEX_SYSTEM2_2):
        t_sec = t.to_sec() - analysis_start
        data['time_joint'].append(t_sec)
        
        # System 1 (rad -> deg)
        # sensor1: 正方向
        data['theta_actual_1_sensor1'].append(np.rad2deg(-msg.position[THETA_INDEX_SYSTEM1_1]))
        # sensor2: ミラー → -1を乗算
        data['theta_actual_1_sensor2'].append(np.rad2deg(msg.position[THETA_INDEX_SYSTEM1_2]))
        
        # System 2 (rad -> deg)
        # sensor1: 正方向
        data['theta_actual_2_sensor1'].append(np.rad2deg(msg.position[THETA_INDEX_SYSTEM2_1]))
        # sensor2: ミラー → -1を乗算
        data['theta_actual_2_sensor2'].append(np.rad2deg(-msg.position[THETA_INDEX_SYSTEM2_2]))

# 圧力指令
for topic, msg, t in bag.read_messages(topics=['/mpa_cmd'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - analysis_start
    data['time_pressure'].append(t_sec)
    # DAC値 -> MPa変換
    data['p1'].append(msg.x * 0.9 / 4096.0)
    data['p2'].append(msg.y * 0.9 / 4096.0)
    data['p3'].append(msg.z * 0.9 / 4096.0)
    data['p4'].append(msg.w * 0.9 / 4096.0)

# 姿勢・位置
for topic, msg, t in bag.read_messages(topics=['/kinikun1/uav/baselink/odom'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - analysis_start
    data['time_odom'].append(t_sec)
    
    # 姿勢（オイラー角として格納されていると仮定）
    data['roll'].append(msg.pose.pose.orientation.x)
    data['pitch'].append(msg.pose.pose.orientation.y)
    data['yaw'].append(msg.pose.pose.orientation.z)
    
    # 位置
    data['pos_x'].append(msg.pose.pose.position.x)
    data['pos_y'].append(msg.pose.pose.position.y)
    data['pos_z'].append(msg.pose.pose.position.z)

# PID誤差データ
for topic, msg, t in bag.read_messages(topics=['/kinikun1/debug/pose/pid'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - analysis_start
    data['time_pid'].append(t_sec)
    data['err_x'].append(msg.x.err_p)
    data['err_y'].append(msg.y.err_p)
    data['err_z'].append(msg.z.err_p)

bag.close()

# NumPy配列に変換
for key in data:
    data[key] = np.array(data[key])

print(f"\nData loaded:")
print(f"  Joint states: {len(data['time_joint'])} samples")
print(f"  Target (Sys1): {len(data['time_target'])} samples")
print(f"  Target (Sys2): {len(data['time_target_2'])} samples")
print(f"  Pressure: {len(data['time_pressure'])} samples")
print(f"  Odom: {len(data['time_odom'])} samples")
print(f"  PID: {len(data['time_pid'])} samples")

# ============================================
# データ処理
# ============================================

# 目標角度を実角度の時刻に補間
if len(data['time_target']) > 0 and len(data['time_joint']) > 0:
    theta_target_1_interp = np.interp(data['time_joint'], data['time_target'], data['theta_target_1'])
else:
    theta_target_1_interp = np.zeros_like(data['time_joint'])

if len(data['time_target_2']) > 0 and len(data['time_joint']) > 0:
    theta_target_2_interp = np.interp(data['time_joint'], data['time_target_2'], data['theta_target_2'])
else:
    theta_target_2_interp = np.zeros_like(data['time_joint'])

# 追従誤差（各センサー個別）
error_1_sensor1 = theta_target_1_interp - data['theta_actual_1_sensor1']
error_1_sensor2 = theta_target_1_interp - data['theta_actual_1_sensor2']
error_2_sensor1 = theta_target_2_interp - data['theta_actual_2_sensor1']
error_2_sensor2 = theta_target_2_interp - data['theta_actual_2_sensor2']

# 平均誤差（参考用）
error_1_avg = (error_1_sensor1 + error_1_sensor2) / 2.0
error_2_avg = (error_2_sensor1 + error_2_sensor2) / 2.0

# ============================================
# 統計計算
# ============================================
def calc_stats(data, name=""):
    """統計量計算"""
    if len(data) == 0:
        return {'name': name, 'rmse': 0, 'mean': 0, 'std': 0, 'max': 0}
    rmse = np.sqrt(np.mean(data**2))
    mean = np.mean(data)
    std = np.std(data)
    max_val = np.max(np.abs(data))
    return {
        'name': name,
        'rmse': rmse,
        'mean': mean,
        'std': std,
        'max': max_val
    }

print("\n" + "="*60)
print("TRACKING PERFORMANCE STATISTICS")
print("="*60)

stats_1_s1 = calc_stats(error_1_sensor1, "System 1 Sensor 1")
stats_1_s2 = calc_stats(error_1_sensor2, "System 1 Sensor 2 (mirror corrected)")
stats_2_s1 = calc_stats(error_2_sensor1, "System 2 Sensor 1")
stats_2_s2 = calc_stats(error_2_sensor2, "System 2 Sensor 2 (mirror corrected)")

print(f"\n[System 1 - Sensor 1 (index {THETA_INDEX_SYSTEM1_1})]")
print(f"  RMSE:  {stats_1_s1['rmse']:.4f} deg")
print(f"  Mean:  {stats_1_s1['mean']:+.4f} deg")
print(f"  Std:   {stats_1_s1['std']:.4f} deg")
print(f"  Max:   {stats_1_s1['max']:.4f} deg")

print(f"\n[System 1 - Sensor 2 (index {THETA_INDEX_SYSTEM1_2}, mirror corrected)]")
print(f"  RMSE:  {stats_1_s2['rmse']:.4f} deg")
print(f"  Mean:  {stats_1_s2['mean']:+.4f} deg")
print(f"  Std:   {stats_1_s2['std']:.4f} deg")
print(f"  Max:   {stats_1_s2['max']:.4f} deg")

print(f"\n[System 2 - Sensor 1 (index {THETA_INDEX_SYSTEM2_1})]")
print(f"  RMSE:  {stats_2_s1['rmse']:.4f} deg")
print(f"  Mean:  {stats_2_s1['mean']:+.4f} deg")
print(f"  Std:   {stats_2_s1['std']:.4f} deg")
print(f"  Max:   {stats_2_s1['max']:.4f} deg")

print(f"\n[System 2 - Sensor 2 (index {THETA_INDEX_SYSTEM2_2}, mirror corrected)]")
print(f"  RMSE:  {stats_2_s2['rmse']:.4f} deg")
print(f"  Mean:  {stats_2_s2['mean']:+.4f} deg")
print(f"  Std:   {stats_2_s2['std']:.4f} deg")
print(f"  Max:   {stats_2_s2['max']:.4f} deg")

# 相関係数
if len(data['theta_actual_1_sensor1']) > 0:
    corr_1_s1 = np.corrcoef(theta_target_1_interp, data['theta_actual_1_sensor1'])[0, 1]
    corr_1_s2 = np.corrcoef(theta_target_1_interp, data['theta_actual_1_sensor2'])[0, 1]
    corr_2_s1 = np.corrcoef(theta_target_2_interp, data['theta_actual_2_sensor1'])[0, 1]
    corr_2_s2 = np.corrcoef(theta_target_2_interp, data['theta_actual_2_sensor2'])[0, 1]
    print(f"\n[Correlation]")
    print(f"  System 1 Sensor 1: {corr_1_s1:.4f}")
    print(f"  System 1 Sensor 2: {corr_1_s2:.4f}")
    print(f"  System 2 Sensor 1: {corr_2_s1:.4f}")
    print(f"  System 2 Sensor 2: {corr_2_s2:.4f}")

print("="*60 + "\n")

# ============================================
# 図1: System 1 目標角度追従
# ============================================
fig1, ax1 = plt.subplots(figsize=(12, 6))

if len(data['time_target']) > 0:
    ax1.plot(data['time_target'], data['theta_target_1'], 'k-', linewidth=2, 
             label='Target', alpha=0.8)
if len(data['time_cmd']) > 0:
    ax1.plot(data['time_cmd'], data['theta_cmd_1'], 'g--', linewidth=1.5, 
             label='Command', alpha=0.6)
if len(data['time_joint']) > 0:
    ax1.plot(data['time_joint'], data['theta_actual_1_sensor1'], 'b-', linewidth=1.5, 
             label=f'Actual (idx {THETA_INDEX_SYSTEM1_1}, mirror corr.)', alpha=0.8)
    ax1.plot(data['time_joint'], data['theta_actual_1_sensor2'], 'r-', linewidth=1.5, 
             label=f'Actual (idx {THETA_INDEX_SYSTEM1_2})', alpha=0.8)

ax1.set_xlabel('Time [s]', **label_font)
ax1.set_ylabel('Angle [deg]', **label_font)
ax1.set_title('System 1: Target Tracking', fontweight='bold', fontsize=14)
ax1.legend(loc='best', **legend_font)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, None)
plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, '01_system1_target_tracking.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 01_system1_target_tracking.pdf")

# ============================================
# 図2: System 2 目標角度追従
# ============================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

if len(data['time_target_2']) > 0:
    ax2.plot(data['time_target_2'], data['theta_target_2'], 'k-', linewidth=2, 
             label='Target', alpha=0.8)
if len(data['time_cmd_2']) > 0:
    ax2.plot(data['time_cmd_2'], data['theta_cmd_2'], 'g--', linewidth=1.5, 
             label='Command', alpha=0.6)
if len(data['time_joint']) > 0:
    ax2.plot(data['time_joint'], data['theta_actual_2_sensor1'], 'b-', linewidth=1.5, 
             label=f'Actual (idx {THETA_INDEX_SYSTEM2_1})', alpha=0.8)
    ax2.plot(data['time_joint'], data['theta_actual_2_sensor2'], 'r-', linewidth=1.5, 
             label=f'Actual (idx {THETA_INDEX_SYSTEM2_2}, mirror corr.)', alpha=0.8)

ax2.set_xlabel('Time [s]', **label_font)
ax2.set_ylabel('Angle [deg]', **label_font)
ax2.set_title('System 2: Target Tracking', fontweight='bold', fontsize=14)
ax2.legend(loc='best', **legend_font)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, None)
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, '02_system2_target_tracking.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 02_system2_target_tracking.pdf")

# ============================================
# 図3: System 1 追従誤差
# ============================================
fig3, ax3 = plt.subplots(figsize=(12, 5))

if len(data['time_joint']) > 0:
    ax3.plot(data['time_joint'], error_1_sensor1, 'b-', linewidth=1.5, 
             label=f'Sensor 1 (idx {THETA_INDEX_SYSTEM1_1})', alpha=0.8)
    ax3.plot(data['time_joint'], error_1_sensor2, 'r-', linewidth=1.5, 
             label=f'Sensor 2 (idx {THETA_INDEX_SYSTEM1_2}, mirror corr.)', alpha=0.8)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.5)

ax3.set_xlabel('Time [s]', **label_font)
ax3.set_ylabel('Tracking Error [deg]', **label_font)
ax3.set_title(f'System 1: Tracking Error (RMSE: S1={stats_1_s1["rmse"]:.3f}°, S2={stats_1_s2["rmse"]:.3f}°)', 
              fontweight='bold', fontsize=14)
ax3.legend(loc='best', **legend_font)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, None)
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, '03_system1_tracking_error.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 03_system1_tracking_error.pdf")

# ============================================
# 図4: System 2 追従誤差
# ============================================
fig4, ax4 = plt.subplots(figsize=(12, 5))

if len(data['time_joint']) > 0:
    ax4.plot(data['time_joint'], error_2_sensor1, 'b-', linewidth=1.5, 
             label=f'Sensor 1 (idx {THETA_INDEX_SYSTEM2_1})', alpha=0.8)
    ax4.plot(data['time_joint'], error_2_sensor2, 'r-', linewidth=1.5, 
             label=f'Sensor 2 (idx {THETA_INDEX_SYSTEM2_2}, mirror corr.)', alpha=0.8)
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.5)

ax4.set_xlabel('Time [s]', **label_font)
ax4.set_ylabel('Tracking Error [deg]', **label_font)
ax4.set_title(f'System 2: Tracking Error (RMSE: S1={stats_2_s1["rmse"]:.3f}°, S2={stats_2_s2["rmse"]:.3f}°)', 
              fontweight='bold', fontsize=14)
ax4.legend(loc='best', **legend_font)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, None)
plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, '04_system2_tracking_error.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 04_system2_tracking_error.pdf")

# ============================================
# 図5: System 1 拮抗圧力指令
# ============================================
fig5, ax5 = plt.subplots(figsize=(12, 5))

if len(data['time_pressure']) > 0:
    ax5.plot(data['time_pressure'], data['p1'], 'tab:blue', linewidth=1.5, label='P1 (Agonist)')
    ax5.plot(data['time_pressure'], data['p2'], 'tab:orange', linewidth=1.5, label='P2 (Antagonist)')

ax5.set_xlabel('Time [s]', **label_font)
ax5.set_ylabel('Pressure [MPa]', **label_font)
ax5.set_title('System 1: Antagonistic Pressure Commands', fontweight='bold', fontsize=14)
ax5.legend(loc='best', **legend_font)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, None)
plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, '05_system1_pressure_command.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 05_system1_pressure_command.pdf")

# ============================================
# 図6: System 2 拮抗圧力指令
# ============================================
fig6, ax6 = plt.subplots(figsize=(12, 5))

if len(data['time_pressure']) > 0:
    ax6.plot(data['time_pressure'], data['p3'], 'tab:blue', linewidth=1.5, label='P3 (Agonist)')
    ax6.plot(data['time_pressure'], data['p4'], 'tab:orange', linewidth=1.5, label='P4 (Antagonist)')

ax6.set_xlabel('Time [s]', **label_font)
ax6.set_ylabel('Pressure [MPa]', **label_font)
ax6.set_title('System 2: Antagonistic Pressure Commands', fontweight='bold', fontsize=14)
ax6.legend(loc='best', **legend_font)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, None)
plt.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, '06_system2_pressure_command.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 06_system2_pressure_command.pdf")

# ============================================
# 図7: 圧力差分
# ============================================
fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

if len(data['time_pressure']) > 0:
    p_diff_1 = data['p1'] - data['p2']
    p_diff_2 = data['p3'] - data['p4']
    
    ax7a.plot(data['time_pressure'], p_diff_1, 'tab:purple', linewidth=1.5)
    ax7a.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax7a.set_ylabel('ΔP (P1-P2) [MPa]', **label_font)
    ax7a.set_title('System 1: Pressure Difference', fontweight='bold')
    ax7a.grid(True, alpha=0.3)
    
    ax7b.plot(data['time_pressure'], p_diff_2, 'tab:green', linewidth=1.5)
    ax7b.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax7b.set_xlabel('Time [s]', **label_font)
    ax7b.set_ylabel('ΔP (P3-P4) [MPa]', **label_font)
    ax7b.set_title('System 2: Pressure Difference', fontweight='bold')
    ax7b.grid(True, alpha=0.3)

ax7a.set_xlim(0, None)
plt.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, '07_pressure_difference.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 07_pressure_difference.pdf")

# ============================================
# 図8: UAV位置誤差
# ============================================
fig8, ax8 = plt.subplots(figsize=(12, 5))

if len(data['time_pid']) > 0:
    ax8.plot(data['time_pid'], data['err_x'], 'r-', linewidth=1.5, label='X error', alpha=0.8)
    ax8.plot(data['time_pid'], data['err_y'], 'g-', linewidth=1.5, label='Y error', alpha=0.8)
    ax8.plot(data['time_pid'], data['err_z'], 'b-', linewidth=1.5, label='Z error', alpha=0.8)
    ax8.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # RMSE表示
    rmse_x = np.sqrt(np.mean(data['err_x']**2))
    rmse_y = np.sqrt(np.mean(data['err_y']**2))
    rmse_z = np.sqrt(np.mean(data['err_z']**2))
    ax8.set_title(f'UAV Position Error (RMSE: X={rmse_x:.4f}m, Y={rmse_y:.4f}m, Z={rmse_z:.4f}m)', 
                  fontweight='bold', fontsize=14)
elif len(data['time_odom']) > 0:
    # PIDがない場合は位置の変動を表示
    pos_x_centered = data['pos_x'] - np.mean(data['pos_x'])
    pos_y_centered = data['pos_y'] - np.mean(data['pos_y'])
    pos_z_centered = data['pos_z'] - np.mean(data['pos_z'])
    
    ax8.plot(data['time_odom'], pos_x_centered, 'r-', linewidth=1.5, label='X deviation', alpha=0.8)
    ax8.plot(data['time_odom'], pos_y_centered, 'g-', linewidth=1.5, label='Y deviation', alpha=0.8)
    ax8.plot(data['time_odom'], pos_z_centered, 'b-', linewidth=1.5, label='Z deviation', alpha=0.8)
    ax8.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax8.set_title('UAV Position Deviation from Mean', fontweight='bold', fontsize=14)

ax8.set_xlabel('Time [s]', **label_font)
ax8.set_ylabel('Position Error [m]', **label_font)
ax8.legend(loc='best', **legend_font)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, None)
plt.tight_layout()
fig8.savefig(os.path.join(OUTPUT_DIR, '08_uav_position_error.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 08_uav_position_error.pdf")

# ============================================
# 図9: UAV姿勢変化
# ============================================
fig9, ax9 = plt.subplots(figsize=(12, 5))

if len(data['time_odom']) > 0:
    ax9.plot(data['time_odom'], np.rad2deg(data['roll']), 'r-', linewidth=1.5, label='Roll', alpha=0.8)
    ax9.plot(data['time_odom'], np.rad2deg(data['pitch']), 'g-', linewidth=1.5, label='Pitch', alpha=0.8)
    ax9.plot(data['time_odom'], np.rad2deg(data['yaw']), 'b-', linewidth=1.5, label='Yaw', alpha=0.8)
    ax9.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # 標準偏差表示
    std_roll = np.rad2deg(np.std(data['roll']))
    std_pitch = np.rad2deg(np.std(data['pitch']))
    std_yaw = np.rad2deg(np.std(data['yaw']))
    ax9.set_title(f'UAV Attitude (Std: Roll={std_roll:.3f}°, Pitch={std_pitch:.3f}°, Yaw={std_yaw:.3f}°)', 
                  fontweight='bold', fontsize=14)

ax9.set_xlabel('Time [s]', **label_font)
ax9.set_ylabel('Attitude [deg]', **label_font)
ax9.legend(loc='best', ncol=3, **legend_font)
ax9.grid(True, alpha=0.3)
ax9.set_xlim(0, None)
plt.tight_layout()
fig9.savefig(os.path.join(OUTPUT_DIR, '09_uav_attitude.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 09_uav_attitude.pdf")

# ============================================
# 図10: 統計サマリーテーブル
# ============================================
fig10, ax10 = plt.subplots(figsize=(10, 8))
ax10.axis('off')

table_data = [
    ['Metric', 'Sys1 S1', 'Sys1 S2', 'Sys2 S1', 'Sys2 S2'],
    ['RMSE [deg]', f'{stats_1_s1["rmse"]:.4f}', f'{stats_1_s2["rmse"]:.4f}', 
     f'{stats_2_s1["rmse"]:.4f}', f'{stats_2_s2["rmse"]:.4f}'],
    ['Mean [deg]', f'{stats_1_s1["mean"]:+.4f}', f'{stats_1_s2["mean"]:+.4f}', 
     f'{stats_2_s1["mean"]:+.4f}', f'{stats_2_s2["mean"]:+.4f}'],
    ['Std [deg]', f'{stats_1_s1["std"]:.4f}', f'{stats_1_s2["std"]:.4f}', 
     f'{stats_2_s1["std"]:.4f}', f'{stats_2_s2["std"]:.4f}'],
    ['Max [deg]', f'{stats_1_s1["max"]:.4f}', f'{stats_1_s2["max"]:.4f}', 
     f'{stats_2_s1["max"]:.4f}', f'{stats_2_s2["max"]:.4f}'],
]

if len(data['theta_actual_1_sensor1']) > 0:
    table_data.append(['Correlation', f'{corr_1_s1:.4f}', f'{corr_1_s2:.4f}', 
                       f'{corr_2_s1:.4f}', f'{corr_2_s2:.4f}'])

# UAV安定性
if len(data['time_odom']) > 0:
    table_data.append(['', '', '', '', ''])
    table_data.append(['UAV Stability', '', '', '', ''])
    table_data.append(['Roll Std [deg]', f'{np.rad2deg(np.std(data["roll"])):.4f}', '', '', ''])
    table_data.append(['Pitch Std [deg]', f'{np.rad2deg(np.std(data["pitch"])):.4f}', '', '', ''])
    table_data.append(['Yaw Std [deg]', f'{np.rad2deg(np.std(data["yaw"])):.4f}', '', '', ''])

if len(data['time_pid']) > 0:
    table_data.append(['Pos X RMSE [m]', f'{np.sqrt(np.mean(data["err_x"]**2)):.4f}', '', '', ''])
    table_data.append(['Pos Y RMSE [m]', f'{np.sqrt(np.mean(data["err_y"]**2)):.4f}', '', '', ''])
    table_data.append(['Pos Z RMSE [m]', f'{np.sqrt(np.mean(data["err_z"]**2)):.4f}', '', '', ''])

table = ax10.table(cellText=table_data, loc='center', cellLoc='center',
                   colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# ヘッダー行装飾
for j in range(5):
    table[(0, j)].set_text_props(fontweight='bold')
    table[(0, j)].set_facecolor('#E6E6E6')

ax10.set_title('Performance Statistics Summary', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
fig10.savefig(os.path.join(OUTPUT_DIR, '10_performance_statistics.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 10_performance_statistics.pdf")

# ============================================
# 図11: 誤差スペクトル (System 1)
# ============================================
fig11, ax11 = plt.subplots(figsize=(10, 5))

if len(error_1_sensor1) > 10:
    dt_mean = np.mean(np.diff(data['time_joint']))
    fs = 1.0 / dt_mean
    n = len(error_1_sensor1)
    
    freqs = np.fft.rfftfreq(n, dt_mean)
    fft_s1 = np.abs(np.fft.rfft(error_1_sensor1))
    fft_s2 = np.abs(np.fft.rfft(error_1_sensor2))
    
    ax11.semilogy(freqs, fft_s1, 'b-', linewidth=1.5, label='Sensor 1', alpha=0.8)
    ax11.semilogy(freqs, fft_s2, 'r-', linewidth=1.5, label='Sensor 2 (mirror corr.)', alpha=0.8)
    ax11.set_xlabel('Frequency [Hz]', **label_font)
    ax11.set_ylabel('Magnitude', **label_font)
    ax11.set_title('System 1: Error Spectrum', fontweight='bold', fontsize=14)
    ax11.legend(loc='best', **legend_font)
    ax11.grid(True, alpha=0.3, which='both')
    ax11.set_xlim(0, min(50, fs/2))

plt.tight_layout()
fig11.savefig(os.path.join(OUTPUT_DIR, '11_system1_error_spectrum.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 11_system1_error_spectrum.pdf")

# ============================================
# 図12: 誤差スペクトル (System 2)
# ============================================
fig12, ax12 = plt.subplots(figsize=(10, 5))

if len(error_2_sensor1) > 10:
    dt_mean = np.mean(np.diff(data['time_joint']))
    fs = 1.0 / dt_mean
    n = len(error_2_sensor1)
    
    freqs = np.fft.rfftfreq(n, dt_mean)
    fft_s1 = np.abs(np.fft.rfft(error_2_sensor1))
    fft_s2 = np.abs(np.fft.rfft(error_2_sensor2))
    
    ax12.semilogy(freqs, fft_s1, 'b-', linewidth=1.5, label='Sensor 1', alpha=0.8)
    ax12.semilogy(freqs, fft_s2, 'r-', linewidth=1.5, label='Sensor 2 (mirror corr.)', alpha=0.8)
    ax12.set_xlabel('Frequency [Hz]', **label_font)
    ax12.set_ylabel('Magnitude', **label_font)
    ax12.set_title('System 2: Error Spectrum', fontweight='bold', fontsize=14)
    ax12.legend(loc='best', **legend_font)
    ax12.grid(True, alpha=0.3, which='both')
    ax12.set_xlim(0, min(50, fs/2))

plt.tight_layout()
fig12.savefig(os.path.join(OUTPUT_DIR, '12_system2_error_spectrum.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 12_system2_error_spectrum.pdf")

# ============================================
# 図13: 圧力-角度相関 (System 1)
# ============================================
fig13, ax13 = plt.subplots(figsize=(8, 6))

if len(data['time_pressure']) > 0 and len(data['time_joint']) > 0:
    # 時刻を合わせるために補間
    p1_interp = np.interp(data['time_joint'], data['time_pressure'], data['p1'])
    p2_interp = np.interp(data['time_joint'], data['time_pressure'], data['p2'])
    p_diff_1 = p1_interp - p2_interp
    
    scatter13 = ax13.scatter(p_diff_1, data['theta_actual_1_sensor1'], 
                             c=data['time_joint'], cmap='coolwarm', s=10, alpha=0.7,
                             label='Sensor 1')
    ax13.scatter(p_diff_1, data['theta_actual_1_sensor2'], 
                 c=data['time_joint'], cmap='coolwarm', s=10, alpha=0.3,
                 marker='x', label='Sensor 2 (mirror corr.)')
    cbar13 = plt.colorbar(scatter13, ax=ax13)
    cbar13.set_label('Time [s]')

ax13.set_xlabel('Pressure Difference (P1-P2) [MPa]', **label_font)
ax13.set_ylabel('Angle [deg]', **label_font)
ax13.set_title('System 1: Pressure-Angle Correlation', fontweight='bold', fontsize=14)
ax13.legend(loc='best', **legend_font)
ax13.grid(True, alpha=0.3)
plt.tight_layout()
fig13.savefig(os.path.join(OUTPUT_DIR, '13_system1_pressure_angle.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 13_system1_pressure_angle.pdf")

# ============================================
# 図14: 圧力-角度相関 (System 2)
# ============================================
fig14, ax14 = plt.subplots(figsize=(8, 6))

if len(data['time_pressure']) > 0 and len(data['time_joint']) > 0:
    p3_interp = np.interp(data['time_joint'], data['time_pressure'], data['p3'])
    p4_interp = np.interp(data['time_joint'], data['time_pressure'], data['p4'])
    p_diff_2 = p3_interp - p4_interp
    
    scatter14 = ax14.scatter(p_diff_2, data['theta_actual_2_sensor1'], 
                             c=data['time_joint'], cmap='coolwarm', s=10, alpha=0.7,
                             label='Sensor 1')
    ax14.scatter(p_diff_2, data['theta_actual_2_sensor2'], 
                 c=data['time_joint'], cmap='coolwarm', s=10, alpha=0.3,
                 marker='x', label='Sensor 2 (mirror corr.)')
    cbar14 = plt.colorbar(scatter14, ax=ax14)
    cbar14.set_label('Time [s]')

ax14.set_xlabel('Pressure Difference (P3-P4) [MPa]', **label_font)
ax14.set_ylabel('Angle [deg]', **label_font)
ax14.set_title('System 2: Pressure-Angle Correlation', fontweight='bold', fontsize=14)
ax14.legend(loc='best', **legend_font)
ax14.grid(True, alpha=0.3)
plt.tight_layout()
fig14.savefig(os.path.join(OUTPUT_DIR, '14_system2_pressure_angle.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 14_system2_pressure_angle.pdf")

# ============================================
# 図15: 全圧力指令
# ============================================
fig15, ax15 = plt.subplots(figsize=(14, 6))

if len(data['time_pressure']) > 0:
    ax15.plot(data['time_pressure'], data['p1'], 'tab:blue', linewidth=1.5, 
              label='P1 (Sys1)', alpha=0.8)
    ax15.plot(data['time_pressure'], data['p2'], 'tab:orange', linewidth=1.5, 
              label='P2 (Sys1)', alpha=0.8)
    ax15.plot(data['time_pressure'], data['p3'], 'tab:green', linewidth=1.5, 
              label='P3 (Sys2)', alpha=0.8)
    ax15.plot(data['time_pressure'], data['p4'], 'tab:red', linewidth=1.5, 
              label='P4 (Sys2)', alpha=0.8)

ax15.set_xlabel('Time [s]', **label_font)
ax15.set_ylabel('Pressure [MPa]', **label_font)
ax15.set_title('All Pressure Commands', fontweight='bold', fontsize=14)
ax15.legend(loc='best', ncol=4, **legend_font)
ax15.grid(True, alpha=0.3)
ax15.set_xlim(0, None)
ax15.set_ylim(0, 0.8)
plt.tight_layout()
fig15.savefig(os.path.join(OUTPUT_DIR, '15_all_pressure_commands.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: 15_all_pressure_commands.pdf")

# ============================================
# 完了メッセージ
# ============================================
print("\n" + "="*60)
print(f"All figures saved to: {OUTPUT_DIR}/")
print("="*60)
print("\nFiles generated:")
print("  01_system1_target_tracking.pdf")
print("  02_system2_target_tracking.pdf")
print("  03_system1_tracking_error.pdf")
print("  04_system2_tracking_error.pdf")
print("  05_system1_pressure_command.pdf")
print("  06_system2_pressure_command.pdf")
print("  07_pressure_difference.pdf")
print("  08_uav_position_error.pdf")
print("  09_uav_attitude.pdf")
print("  10_performance_statistics.pdf")
print("  11_system1_error_spectrum.pdf")
print("  12_system2_error_spectrum.pdf")
print("  13_system1_pressure_angle.pdf")
print("  14_system2_pressure_angle.pdf")
print("  15_all_pressure_commands.pdf")

plt.show()