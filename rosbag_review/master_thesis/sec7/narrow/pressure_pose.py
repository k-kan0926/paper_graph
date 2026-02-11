#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_position_pressure.py
Position XYZ と Pressure を縦に並べて表示するスクリプト

使用方法:
    python plot_position_pressure.py

パラメータ:
    BAG_FILE: ROSbagファイルのパス
    TIME_OFFSET_START: 開始時刻オフセット [s]
    TIME_OFFSET_END: 終了時刻オフセット [s]
    OUTPUT_FILE: 出力ファイル名
"""

import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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

label_font = {'fontsize': 20, 'fontweight': 'bold'}
legend_font = {'fontsize': 18}

# ============================================
# パラメータ設定（ここを編集してください）
# ============================================
BAG_FILE = '/home/keiichiro/document/paper_graph/rosbag_review/master_thesis/20251221/side_perching_slachwall2.bag'
OUTPUT_DIR = './figures'
OUTPUT_FILE = 'position_pressure_combined.png'

# 解析区間の設定（秒単位、Noneの場合は全区間）
TIME_OFFSET_START = 41    # 開始オフセット [s]
TIME_OFFSET_END = 48     # 終了オフセット [s]

# ============================================
# 出力ディレクトリ作成
# ============================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# データ読み込み
# ============================================
print("=" * 60)
print("Loading ROS bag...")
print("=" * 60)

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
print("=" * 60)

# データ格納用
data = {
    'time_odom': [],
    'time_pressure': [],
    'time_pid': [],
    'pos_x': [], 'pos_y': [], 'pos_z': [],
    'err_x': [], 'err_y': [], 'err_z': [],
    'p1': [], 'p2': [], 'p3': [], 'p4': [],
}

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

# 位置（オドメトリ）
for topic, msg, t in bag.read_messages(topics=['/kinikun1/uav/baselink/odom'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - analysis_start
    data['time_odom'].append(t_sec)
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
print(f"  Odom: {len(data['time_odom'])} samples")
print(f"  Pressure: {len(data['time_pressure'])} samples")
print(f"  PID: {len(data['time_pid'])} samples")

# ============================================
# グラフ作成（Position XYZ + Pressure 縦並び）
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- 上段: Position XYZ ---
ax1 = axes[0]

if len(data['time_pid']) > 0:
    # PIDエラーがある場合はそれを表示
    ax1.plot(data['time_pid'], data['err_x'], 'r-', linewidth=1.5, label='X error', alpha=0.8)
    ax1.plot(data['time_pid'], data['err_y'], 'g-', linewidth=1.5, label='Y error', alpha=0.8)
    ax1.plot(data['time_pid'], data['err_z'], 'b-', linewidth=1.5, label='Z error', alpha=0.8)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    
    rmse_x = np.sqrt(np.mean(data['err_x']**2))
    rmse_y = np.sqrt(np.mean(data['err_y']**2))
    rmse_z = np.sqrt(np.mean(data['err_z']**2))

    ax1.set_ylabel('Position Error [m]', **label_font)
    
    
elif len(data['time_odom']) > 0:
    # PIDがない場合は位置の変動を表示
    pos_x_centered = data['pos_x']
    pos_y_centered = data['pos_y']
    pos_z_centered = data['pos_z']
    
    ax1.plot(data['time_odom'], pos_x_centered, 'r-', linewidth=1.5, label='X', alpha=0.8)
    ax1.plot(data['time_odom'], pos_y_centered, 'g-', linewidth=1.5, label='Y', alpha=0.8)
    ax1.plot(data['time_odom'], pos_z_centered, 'b-', linewidth=1.5, label='Z', alpha=0.8)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Position [m]', **label_font)

ax1.legend(loc='upper left', **legend_font)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.grid(True, alpha=0.3)

# --- 下段: Pressure ---
ax2 = axes[1]

if len(data['time_pressure']) > 0:
    ax2.plot(data['time_pressure'], data['p1'], 'tab:blue', linewidth=1.5,
             label='P1 (SysA)', alpha=0.8)
    ax2.plot(data['time_pressure'], data['p3'], 'tab:green', linewidth=1.5,
             label='P3 (SysB)', alpha=0.8)

ax2.set_xlabel('Time [s]', **label_font)
ax2.set_ylabel('Pressure [MPa]', **label_font)
ax2.legend(loc='upper left', ncol=2, **legend_font)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.8)
ax2.tick_params(axis='both', which='major', labelsize=16)

# X軸の範囲設定
ax2.set_xlim(0, None)

plt.tight_layout()

# 保存
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_path}")

plt.show()

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)