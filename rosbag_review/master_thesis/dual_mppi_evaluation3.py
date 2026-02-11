#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
dual_mppi_evaluation_improved.py
2系統MPPI制御システムの総合評価スクリプト（改良版）

改良点:
- 圧力-角度の対応関係をコード2に合わせて修正
  * System 1 (θ₁): 圧力 X (正方向) と Y (負方向) が拮抗
  * System 2 (θ₂): 圧力 Z (正方向) と W (負方向) が拮抗 (ミラー配置)
- 図の結合: fig1+fig2（縦）、fig8+fig9（縦）、fig13+fig14（横）
- 単体出力: fig10 (統計表), fig15 (全圧力)
- ミラーセンサーの補正（theta_index, theta_index_4に-1を乗算）
"""

import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import os
from tf.transformations import euler_from_quaternion

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

label_font = {'fontsize': 18, 'fontweight': 'bold'}
legend_font = {'fontsize': 14}

# ============================================
# パラメータ設定
# ============================================
BAG_FILE = '/home/keiichiro/document/paper_graph/rosbag_review/master_thesis/20251215/mppi.bag'
OUTPUT_DIR = './sec7/narrow/figures_improved'

# ============================================
# 圧力-角度の対応関係（コード2に準拠）
# ============================================
# /mpa_cmd トピックの Vector4:
#   x → Pressure X
#   y → Pressure Y  
#   z → Pressure Z
#   w → Pressure W
#
# 関節角度との対応（コード2より）:
#   Joint 1: X+/Y- (圧力Xが正方向、Yが負方向)
#   Joint 2: W+/Z- (圧力Wが正方向、Zが負方向)  
#   Joint 3: Z+/W- (圧力Zが正方向、Wが負方向)
#   Joint 4: Y+/X- (圧力Yが正方向、Xが負方向)
#
# システム定義:
#   System 1 (θ₁): Joint 1 & 4 → 圧力 X と Y が拮抗
#   System 2 (θ₂): Joint 2 & 3 → 圧力 Z と W が拮抗

# センサーインデックス
THETA_INDEX_SYSTEM1_1 = 0   # theta_index_2 (正)
THETA_INDEX_SYSTEM1_2 = 3   # theta_index (ミラー → -1を乗算)
THETA_INDEX_SYSTEM2_1 = 1   # theta_index_3 (正)
THETA_INDEX_SYSTEM2_2 = 2   # theta_index_4 (ミラー → -1を乗算)

# 解析区間の設定（秒単位）
TIME_OFFSET_START = 52   # 開始オフセット [s]
TIME_OFFSET_END = 182    # 終了オフセット [s]
#mppi 52 182 bring 35 103 narrow 130 162

# 圧力変換係数
PRESSURE_COEFF = 0.9 / 4096.0

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
analysis_start = bag_start + TIME_OFFSET_START
analysis_end = bag_start + TIME_OFFSET_END

start_time = rospy.Time(analysis_start)
end_time = rospy.Time(analysis_end)

print(f"Bag duration: {bag_end - bag_start:.2f}s")
print(f"Analysis range: {TIME_OFFSET_START:.2f}s - {TIME_OFFSET_END:.2f}s")
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
    
    # System 1 (θ₁)
    'theta_target_1': [],
    'theta_cmd_1': [],
    'theta_actual_1_sensor1': [],  # 正方向
    'theta_actual_1_sensor2': [],  # ミラー補正後
    
    # System 2 (θ₂)
    'theta_target_2': [],
    'theta_cmd_2': [],
    'theta_actual_2_sensor1': [],  # 正方向
    'theta_actual_2_sensor2': [],  # ミラー補正後
    
    # 圧力（X, Y, Z, W として格納）
    'pressure_x': [], 'pressure_y': [], 'pressure_z': [], 'pressure_w': [],
    
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
        data['theta_actual_1_sensor1'].append(np.rad2deg(-msg.position[THETA_INDEX_SYSTEM1_1]))
        data['theta_actual_1_sensor2'].append(np.rad2deg(msg.position[THETA_INDEX_SYSTEM1_2]))
        
        # System 2 (rad -> deg)
        data['theta_actual_2_sensor1'].append(np.rad2deg(msg.position[THETA_INDEX_SYSTEM2_1]))
        data['theta_actual_2_sensor2'].append(np.rad2deg(-msg.position[THETA_INDEX_SYSTEM2_2]))

# 圧力指令（X, Y, Z, W として格納）
for topic, msg, t in bag.read_messages(topics=['/mpa_cmd'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - analysis_start
    data['time_pressure'].append(t_sec)
    # DAC値 -> MPa変換
    data['pressure_x'].append(msg.x * PRESSURE_COEFF)
    data['pressure_y'].append(msg.y * PRESSURE_COEFF)
    data['pressure_z'].append(msg.z * PRESSURE_COEFF)
    data['pressure_w'].append(msg.w * PRESSURE_COEFF)

# 姿勢・位置

for topic, msg, t in bag.read_messages(topics=['/kinikun1/uav/baselink/odom'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - analysis_start
    data['time_odom'].append(t_sec)
    
    # クォータニオンからオイラー角に変換
    q = msg.pose.pose.orientation
    quaternion = [q.x, q.y, q.z, q.w]
    r, p, y = euler_from_quaternion(quaternion)
    
    data['roll'].append(r)
    data['pitch'].append(p)
    data['yaw'].append(y)
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

# ============================================
# 統計計算
# ============================================
def calc_stats(data_array, name=""):
    """統計量計算"""
    if len(data_array) == 0:
        return {'name': name, 'rmse': 0, 'mean': 0, 'std': 0, 'max': 0}
    rmse = np.sqrt(np.mean(data_array**2))
    mean = np.mean(data_array)
    std = np.std(data_array)
    max_val = np.max(np.abs(data_array))
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
stats_1_s2 = calc_stats(error_1_sensor2, "System 1 Sensor 2")
stats_2_s1 = calc_stats(error_2_sensor1, "System 2 Sensor 1")
stats_2_s2 = calc_stats(error_2_sensor2, "System 2 Sensor 2")

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
# 図1: System 1 & 2 目標角度追従（縦結合）
# ============================================
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# System 1
if len(data['time_target']) > 0:
    ax1a.plot(data['time_target'], data['theta_target_1'], 'k-', linewidth=2, 
              label='Target', alpha=0.8)
if len(data['time_joint']) > 0:
    ax1a.plot(data['time_joint'], data['theta_actual_1_sensor1'], 'b-', linewidth=1.5, 
              label='Joint 1 (1+/2-)', alpha=0.8)
    ax1a.plot(data['time_joint'], data['theta_actual_1_sensor2'], 'r-', linewidth=1.5, 
              label='Joint 4 (2+/1-)', alpha=0.8)

ax1a.set_ylabel('SystemA Angle [deg]', **label_font)
ax1a.legend(loc='best', **legend_font)
ax1a.grid(True, alpha=0.3)
ax1a.set_ylim(-40, 40)
ax1a.tick_params(axis='both', which='major', labelsize=16)


# System 2
if len(data['time_target_2']) > 0:
    ax1b.plot(data['time_target_2'], data['theta_target_2'], 'k-', linewidth=2, 
              label='Target', alpha=0.8)
if len(data['time_joint']) > 0:
    ax1b.plot(data['time_joint'], data['theta_actual_2_sensor1'], 'b-', linewidth=1.5, 
              label='Joint 2 (4+/3-)', alpha=0.8)
    ax1b.plot(data['time_joint'], data['theta_actual_2_sensor2'], 'r-', linewidth=1.5, 
              label='Joint 3 (3+/4-)', alpha=0.8)

ax1b.set_xlabel('Time [s]', **label_font)
ax1b.set_ylabel('SystemB Angle [deg]', **label_font)
ax1b.legend(loc='best', **legend_font)
ax1b.grid(True, alpha=0.3)
ax1b.set_xlim(0, None)
ax1b.set_ylim(-40, 40)
ax1b.tick_params(axis='both', which='major', labelsize=16)


plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'fig01_both_systems_tracking.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig01_both_systems_tracking.pdf")

# ============================================
# 図2: System 1 & 2 追従誤差（縦結合）
# ============================================
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# System 1 誤差
if len(data['time_joint']) > 0:
    ax2a.plot(data['time_joint'], error_1_sensor1, 'b-', linewidth=1.5, 
              label='Joint 1 (X+/Y-)', alpha=0.8)
    ax2a.plot(data['time_joint'], error_1_sensor2, 'r-', linewidth=1.5, 
              label='Joint 4 (Y+/X-)', alpha=0.8)
    ax2a.axhline(0, color='gray', linestyle='--', linewidth=0.5)

ax2a.set_ylabel('System1 Tracking Error [deg]', **label_font)
ax2a.legend(loc='best', **legend_font)
ax2a.grid(True, alpha=0.3)

# System 2 誤差
if len(data['time_joint']) > 0:
    ax2b.plot(data['time_joint'], error_2_sensor1, 'b-', linewidth=1.5, 
              label='Joint 2 (W+/Z-)', alpha=0.8)
    ax2b.plot(data['time_joint'], error_2_sensor2, 'r-', linewidth=1.5, 
              label='Joint 3 (Z+/W-)', alpha=0.8)
    ax2b.axhline(0, color='gray', linestyle='--', linewidth=0.5)

ax2b.set_xlabel('Time [s]', **label_font)
ax2b.set_ylabel('System2 Tracking Error [deg]', **label_font)
ax2b.legend(loc='best', **legend_font)
ax2b.grid(True, alpha=0.3)
ax2b.set_xlim(0, None)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'fig02_both_systems_error.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig02_both_systems_error.pdf")

# ============================================
# 図3: 拮抗圧力指令（両システム）
# ============================================
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

if len(data['time_pressure']) > 0:
    # System 1: X-Y 拮抗
    ax3a.plot(data['time_pressure'], data['pressure_x'], 'tab:red', linewidth=1.5, 
              label='Pressure X (+ direction)')
    ax3a.plot(data['time_pressure'], data['pressure_y'], 'tab:blue', linewidth=1.5, 
              label='Pressure Y (- direction)')
    ax3a.set_ylabel('System1 Pressure [MPa]', **label_font)
    ax3a.legend(loc='best', **legend_font)
    ax3a.grid(True, alpha=0.3)
    ax3a.set_ylim(0, 0.8)
    
    # System 2: Z-W 拮抗
    ax3b.plot(data['time_pressure'], data['pressure_z'], 'tab:green', linewidth=1.5, 
              label='Pressure Z (+ direction)')
    ax3b.plot(data['time_pressure'], data['pressure_w'], 'tab:purple', linewidth=1.5, 
              label='Pressure W (- direction)')
    ax3b.set_xlabel('Time [s]', **label_font)
    ax3b.set_ylabel('System2 Pressure [MPa]', **label_font)
    ax3b.legend(loc='best', **legend_font)
    ax3b.grid(True, alpha=0.3)
    ax3b.set_ylim(0, 0.8)

ax3b.set_xlim(0, None)
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'fig03_antagonistic_pressure.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig03_antagonistic_pressure.pdf")

# ============================================
# 図4: 圧力差分（トルク相当）
# ============================================
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

if len(data['time_pressure']) > 0:
    # System 1: ΔP = X - Y
    p_diff_1 = data['pressure_x'] - data['pressure_y']
    # System 2: ΔP = Z - W
    p_diff_2 = data['pressure_z'] - data['pressure_w']
    
    ax4a.plot(data['time_pressure'], p_diff_1, 'tab:purple', linewidth=1.5)
    ax4a.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax4a.set_ylabel('System1 ΔP (X-Y) [MPa]', **label_font)
    ax4a.grid(True, alpha=0.3)
    
    ax4b.plot(data['time_pressure'], p_diff_2, 'tab:green', linewidth=1.5)
    ax4b.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax4b.set_xlabel('Time [s]', **label_font)
    ax4b.set_ylabel('System2 ΔP (Z-W) [MPa]', **label_font)
    ax4b.grid(True, alpha=0.3)

ax4a.set_xlim(0, None)
plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, 'fig04_pressure_difference.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig04_pressure_difference.pdf")

# ============================================
# 図5: UAV位置誤差 + 姿勢（縦結合）- 旧fig8+fig9
# ============================================
fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# UAV位置誤差
if len(data['time_pid']) > 0:
    ax5a.plot(data['time_pid'], data['err_x'], 'r-', linewidth=1.5, label='X', alpha=0.8)
    ax5a.plot(data['time_pid'], data['err_y'], 'g-', linewidth=1.5, label='Y', alpha=0.8)
    ax5a.plot(data['time_pid'], data['err_z'], 'b-', linewidth=1.5, label='Z', alpha=0.8)
    ax5a.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    
    rmse_x = np.sqrt(np.mean(data['err_x']**2))
    rmse_y = np.sqrt(np.mean(data['err_y']**2))
    rmse_z = np.sqrt(np.mean(data['err_z']**2))

elif len(data['time_odom']) > 0:
    pos_x_centered = data['pos_x'] - np.mean(data['pos_x'])
    pos_y_centered = data['pos_y'] - np.mean(data['pos_y'])
    pos_z_centered = data['pos_z'] - np.mean(data['pos_z'])
    
    ax5a.plot(data['time_odom'], pos_x_centered, 'r-', linewidth=1.5, label='X deviation', alpha=0.8)
    ax5a.plot(data['time_odom'], pos_y_centered, 'g-', linewidth=1.5, label='Y deviation', alpha=0.8)
    ax5a.plot(data['time_odom'], pos_z_centered, 'b-', linewidth=1.5, label='Z deviation', alpha=0.8)
    ax5a.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax5a.set_title('(a) UAV Position Deviation from Mean', fontweight='bold', fontsize=14)

ax5a.set_ylabel('Position Error [m]', **label_font)
ax5a.legend(loc='upper right', ncol=3, **legend_font)
ax5a.grid(True, alpha=0.3)
ax5a.set_ylim(-0.15, 0.15)

# UAV姿勢
if len(data['time_odom']) > 0:
    ax5b.plot(data['time_odom'], np.rad2deg(data['roll']), 'r-', linewidth=1.5, label='Roll', alpha=0.8)
    ax5b.plot(data['time_odom'], np.rad2deg(data['pitch']), 'g-', linewidth=1.5, label='Pitch', alpha=0.8)
    ax5b.plot(data['time_odom'], np.rad2deg(data['yaw']), 'b-', linewidth=1.5, label='Yaw', alpha=0.8)
    ax5b.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    
    std_roll = np.rad2deg(np.std(data['roll']))
    std_pitch = np.rad2deg(np.std(data['pitch']))
    std_yaw = np.rad2deg(np.std(data['yaw']))

ax5b.set_xlabel('Time [s]', **label_font)
ax5b.set_ylabel('Attitude Error [deg]', **label_font)
ax5b.legend(loc='upper right', ncol=3, **legend_font)
ax5b.grid(True, alpha=0.3)
ax5b.set_xlim(0, None)
ax5b.set_ylim(-14, 14)

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, 'fig05_uav_position_attitude.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig05_uav_position_attitude.pdf")

# ============================================
# 図6: 統計サマリーテーブル（旧fig10）
# ============================================
fig6, ax6 = plt.subplots(figsize=(12, 10))
ax6.axis('off')

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

# 圧力-角度関係の説明を追加
table_data.append(['', '', '', '', ''])
table_data.append(['Pressure Mapping', '', '', '', ''])
table_data.append(['System 1', 'X (+)', 'Y (-)', '', ''])
table_data.append(['System 2', 'Z (+)', 'W (-)', '', ''])

table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# ヘッダー行装飾
for j in range(5):
    table[(0, j)].set_text_props(fontweight='bold')
    table[(0, j)].set_facecolor('#E6E6E6')

ax6.set_title('Performance Statistics Summary\n(Pressure: X-Y for Sys1, Z-W for Sys2)', 
              fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, 'fig06_statistics_table.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig06_statistics_table.pdf")

# ============================================
# 図7: 圧力-角度相関（横結合）- 旧fig13+fig14
# ============================================
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(16, 6))

if len(data['time_pressure']) > 0 and len(data['time_joint']) > 0:
    # 時刻を合わせるために補間
    px_interp = np.interp(data['time_joint'], data['time_pressure'], data['pressure_x'])
    py_interp = np.interp(data['time_joint'], data['time_pressure'], data['pressure_y'])
    pz_interp = np.interp(data['time_joint'], data['time_pressure'], data['pressure_z'])
    pw_interp = np.interp(data['time_joint'], data['time_pressure'], data['pressure_w'])
    
    # System 1: ΔP = X - Y
    p_diff_1 = px_interp - py_interp
    # System 2: ΔP = Z - W
    p_diff_2 = pz_interp - pw_interp
    
    # System 1
    scatter7a = ax7a.scatter(p_diff_1, data['theta_actual_1_sensor1'], 
                              c=data['time_joint'], cmap='coolwarm', s=10, alpha=0.7,
                              label='Joint 1 (X+/Y-)' )
    ax7a.scatter(p_diff_1, data['theta_actual_1_sensor2'], 
                 c=data['time_joint'], cmap='coolwarm', s=10, alpha=0.3,
                 marker='x', label='Sensor 2 (mirror corr.)')
    cbar7a = plt.colorbar(scatter7a, ax=ax7a)
    cbar7a.set_label('Time [s]')
    
    ax7a.set_xlabel('Pressure Difference ΔP (X-Y) [MPa]', **label_font)
    ax7a.set_ylabel('Angle [deg]', **label_font)
    ax7a.legend(loc='best', **legend_font)
    ax7a.grid(True, alpha=0.3)
    
    # System 2
    scatter7b = ax7b.scatter(p_diff_2, data['theta_actual_2_sensor1'], 
                              c=data['time_joint'], cmap='coolwarm', s=10, alpha=0.7,
                              label='Sensor 1')
    ax7b.scatter(p_diff_2, data['theta_actual_2_sensor2'], 
                 c=data['time_joint'], cmap='coolwarm', s=10, alpha=0.3,
                 marker='x', label='Sensor 2 (mirror corr.)')
    cbar7b = plt.colorbar(scatter7b, ax=ax7b)
    cbar7b.set_label('Time [s]')
    
    ax7b.set_xlabel('Pressure Difference ΔP (Z-W) [MPa]', **label_font)
    ax7b.set_ylabel('Angle [deg]', **label_font)
    ax7b.legend(loc='best', **legend_font)
    ax7b.grid(True, alpha=0.3)

plt.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, 'fig07_pressure_angle_correlation.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig07_pressure_angle_correlation.pdf")

# ============================================
# 図8: 全圧力指令（旧fig15）
# ============================================
fig8, ax8 = plt.subplots(figsize=(14, 6))

if len(data['time_pressure']) > 0:
    ax8.plot(data['time_pressure'], data['pressure_x'], 'tab:red', linewidth=1.5, 
             label='X (Sys1+)', alpha=0.8)
    ax8.plot(data['time_pressure'], data['pressure_y'], 'tab:blue', linewidth=1.5, 
             label='Y (Sys1-)', alpha=0.8)
    ax8.plot(data['time_pressure'], data['pressure_z'], 'tab:green', linewidth=1.5, 
             label='Z (Sys2+)', alpha=0.8)
    ax8.plot(data['time_pressure'], data['pressure_w'], 'tab:purple', linewidth=1.5, 
             label='W (Sys2-)', alpha=0.8)

ax8.set_xlabel('Time [s]', **label_font)
ax8.set_ylabel('Pressure [MPa]', **label_font)
ax8.legend(loc='best', ncol=4, **legend_font)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, None)
ax8.set_ylim(0, 0.8)
plt.tight_layout()
fig8.savefig(os.path.join(OUTPUT_DIR, 'fig08_all_pressure_commands.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig08_all_pressure_commands.pdf")

# ============================================
# 図9: 誤差スペクトル（両システム）
# ============================================
fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(14, 5))

if len(error_1_sensor1) > 10:
    dt_mean = np.mean(np.diff(data['time_joint']))
    fs = 1.0 / dt_mean
    n = len(error_1_sensor1)
    freqs = np.fft.rfftfreq(n, dt_mean)
    
    # System 1
    fft_1_s1 = np.abs(np.fft.rfft(error_1_sensor1))
    fft_1_s2 = np.abs(np.fft.rfft(error_1_sensor2))
    ax9a.semilogy(freqs, fft_1_s1, 'b-', linewidth=1.5, label='Sensor 1', alpha=0.8)
    ax9a.semilogy(freqs, fft_1_s2, 'r-', linewidth=1.5, label='Sensor 2 (mirror corr.)', alpha=0.8)
    ax9a.set_xlabel('Frequency [Hz]', **label_font)
    ax9a.set_ylabel('Magnitude', **label_font)
    ax9a.legend(loc='best', **legend_font)
    ax9a.grid(True, alpha=0.3, which='both')
    ax9a.set_xlim(0, min(50, fs/2))
    
    # System 2
    fft_2_s1 = np.abs(np.fft.rfft(error_2_sensor1))
    fft_2_s2 = np.abs(np.fft.rfft(error_2_sensor2))
    ax9b.semilogy(freqs, fft_2_s1, 'b-', linewidth=1.5, label='Sensor 1', alpha=0.8)
    ax9b.semilogy(freqs, fft_2_s2, 'r-', linewidth=1.5, label='Sensor 2 (mirror corr.)', alpha=0.8)
    ax9b.set_xlabel('Frequency [Hz]', **label_font)
    ax9b.set_ylabel('Magnitude', **label_font)
    ax9b.legend(loc='best', **legend_font)
    ax9b.grid(True, alpha=0.3, which='both')
    ax9b.set_xlim(0, min(50, fs/2))

plt.tight_layout()
fig9.savefig(os.path.join(OUTPUT_DIR, 'fig09_error_spectrum.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig09_error_spectrum.pdf")

# ============================================
# 図10: MPPI制御入力の時間応答（追加推奨図）
# ============================================
fig10, axes10 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

if len(data['time_joint']) > 0 and len(data['time_pressure']) > 0:
    # 目標角度と実角度（System 1）
    axes10[0].plot(data['time_target'], data['theta_target_1'], 'k-', linewidth=2, label='Target', alpha=0.8)
    axes10[0].plot(data['time_joint'], data['theta_actual_1_sensor2'], 'b-', linewidth=1.5, label='Actual (S1)', alpha=0.8)
    axes10[0].set_ylabel('System 1 Angle [deg]', **label_font)
    axes10[0].legend(loc='best', **legend_font)
    axes10[0].grid(True, alpha=0.3)
    
    # 圧力差分（System 1）
    axes10[1].plot(data['time_pressure'], data['pressure_x'] - data['pressure_y'], 
                   'tab:purple', linewidth=1.5, label='ΔP (X-Y)')
    axes10[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes10[1].set_ylabel('Sys1 ΔP [MPa]', **label_font)
    axes10[1].legend(loc='best', **legend_font)
    axes10[1].grid(True, alpha=0.3)
    
    # 目標角度と実角度（System 2）
    if len(data['time_target_2']) > 0:
        axes10[2].plot(data['time_target_2'], data['theta_target_2'], 'k-', linewidth=2, label='Target', alpha=0.8)
    axes10[2].plot(data['time_joint'], data['theta_actual_2_sensor1'], 'b-', linewidth=1.5, label='Actual (S1)', alpha=0.8)
    axes10[2].set_ylabel('System 2 Angle [deg]', **label_font)
    axes10[2].legend(loc='best', **legend_font)
    axes10[2].grid(True, alpha=0.3)
    
    # 圧力差分（System 2）
    axes10[3].plot(data['time_pressure'], data['pressure_z'] - data['pressure_w'], 
                   'tab:green', linewidth=1.5, label='ΔP (Z-W)')
    axes10[3].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes10[3].set_xlabel('Time [s]', **label_font)
    axes10[3].set_ylabel('Sys2 ΔP [MPa]', **label_font)
    axes10[3].legend(loc='best', **legend_font)
    axes10[3].grid(True, alpha=0.3)

axes10[0].set_xlim(0, None)
plt.tight_layout()
fig10.savefig(os.path.join(OUTPUT_DIR, 'fig10_mppi_control_response.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: fig10_mppi_control_response.pdf")

# ============================================
# 図11: ステップ応答特性の評価（目標値変化時の追従）
# ============================================
# 目標角度の微分を計算して変化点を検出
if len(data['time_target']) > 10:
    fig11, (ax11a, ax11b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 追従遅れの可視化
    ax11a.plot(data['time_target'], data['theta_target_1'], 'k-', linewidth=2, label='Target')
    ax11a.plot(data['time_joint'], data['theta_actual_1_sensor2'], 'b-', linewidth=1.5, 
               label='Actual (Sensor 2)', alpha=0.8)
    ax11a.fill_between(data['time_joint'], theta_target_1_interp, data['theta_actual_1_sensor2'], 
                       alpha=0.3, color='red', label='Tracking Error')
    ax11a.set_ylabel('System1 Angle [deg]', **label_font)
    ax11a.legend(loc='best', **legend_font)
    ax11a.grid(True, alpha=0.3)
    
    ax11b.plot(data['time_target_2'], data['theta_target_2'], 'k-', linewidth=2, label='Target')
    ax11b.plot(data['time_joint'], data['theta_actual_2_sensor1'], 'b-', linewidth=1.5, 
               label='Actual (Sensor 1)', alpha=0.8)
    ax11b.fill_between(data['time_joint'], theta_target_2_interp, data['theta_actual_2_sensor1'], 
                       alpha=0.3, color='red', label='Tracking Error')
    ax11b.set_xlabel('Time [s]', **label_font)
    ax11b.set_ylabel('System2 Angle [deg]', **label_font)
    ax11b.legend(loc='best', **legend_font)
    ax11b.grid(True, alpha=0.3)
    ax11b.set_xlim(0, None)
    
    plt.tight_layout()
    fig11.savefig(os.path.join(OUTPUT_DIR, 'fig11_tracking_visualization.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: fig11_tracking_visualization.pdf")

# ============================================
# 完了メッセージ
# ============================================
print("\n" + "="*60)
print(f"All figures saved to: {OUTPUT_DIR}/")
print("="*60)
print("\nFiles generated:")
print("  fig01_both_systems_tracking.pdf     - System 1 & 2 目標追従（縦結合）")
print("  fig02_both_systems_error.pdf        - System 1 & 2 追従誤差（縦結合）")
print("  fig03_antagonistic_pressure.pdf     - 拮抗圧力（X-Y, Z-W）")
print("  fig04_pressure_difference.pdf       - 圧力差分（トルク相当）")
print("  fig05_uav_position_attitude.pdf     - UAV位置誤差+姿勢（縦結合）")
print("  fig06_statistics_table.pdf          - 統計サマリー（単体）")
print("  fig07_pressure_angle_correlation.pdf - 圧力-角度相関（横結合）")
print("  fig08_all_pressure_commands.pdf     - 全圧力指令（単体）")
print("  fig09_error_spectrum.pdf            - 誤差スペクトル（横結合）")
print("  fig10_mppi_control_response.pdf     - MPPI制御応答（追加）")
print("  fig11_tracking_visualization.pdf    - 追従誤差可視化（追加）")
print("="*60)

print("\n" + "="*60)
print("圧力-角度対応関係（コード2準拠）:")
print("="*60)
print("  System 1 (θ₁): Pressure X (+方向) / Y (-方向) が拮抗")
print("  System 2 (θ₂): Pressure Z (+方向) / W (-方向) が拮抗")
print("="*60)

plt.show()