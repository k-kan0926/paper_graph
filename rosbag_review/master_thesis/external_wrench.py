#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
外力推定値（Force/Torque）の可視化
/kinikun1/estimated_external_wrench (geometry_msgs/WrenchStamped)
"""

import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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
BAG_FILE = '/home/keiichiro/document/paper_graph/rosbag_review/master_thesis/20251221/bringt_master.bag'

# ============================================
# 時間範囲指定モード設定
# ============================================
# MODE: 'flag' = is_trackingフラグで自動検出
#       'time' = 手動で時間指定（バッグ開始からの相対時間[秒]）
MODE = 'time'

# 時間指定モード用パラメータ（MODEが'time'の場合に使用）
# バッグ開始時刻からの相対時間[秒]で指定
START_TIME_SEC = 35.0    # 開始時間 [秒]
END_TIME_SEC = 103.0     # 終了時間 [秒]（Noneで最後まで）

# ============================================
# データ読み込み
# ============================================
bag = rosbag.Bag(BAG_FILE, 'r')
bag_start = bag.get_start_time()
bag_end = bag.get_end_time()

print("=" * 60)
print(f"Bag file: {BAG_FILE}")
print(f"Bag duration: {bag_end - bag_start:.2f}s")
print(f"Mode: {MODE}")
print("=" * 60)

# ============================================
# 時間範囲の決定
# ============================================
if MODE == 'flag':
    # is_trackingフラグから円軌道区間を特定
    tracking_start = None
    tracking_end = None

    for topic, msg, t in bag.read_messages(topics=['/circle_trajectory_follow/is_tracking']):
        if msg.data and tracking_start is None:
            tracking_start = t.to_sec()
        elif not msg.data and tracking_start is not None and tracking_end is None:
            tracking_end = t.to_sec()

    if tracking_start is None:
        tracking_start = bag_start
        print("Warning: is_tracking flag not found, using bag start time")
    if tracking_end is None:
        tracking_end = bag_end
        print("Warning: is_tracking end not found, using bag end time")

    print("Circle tracking period (flag mode): {:.2f}s - {:.2f}s (duration: {:.2f}s)".format(
        tracking_start - bag_start, tracking_end - bag_start, tracking_end - tracking_start))

elif MODE == 'time':
    # 手動時間指定モード
    tracking_start = bag_start + START_TIME_SEC
    
    if END_TIME_SEC is None:
        tracking_end = bag_end
    else:
        tracking_end = bag_start + END_TIME_SEC
    
    # 範囲チェック
    if tracking_start < bag_start:
        tracking_start = bag_start
        print("Warning: START_TIME_SEC adjusted to bag start")
    if tracking_end > bag_end:
        tracking_end = bag_end
        print("Warning: END_TIME_SEC adjusted to bag end")
    if tracking_start >= tracking_end:
        raise ValueError("START_TIME_SEC must be less than END_TIME_SEC")
    
    print("Analysis period (time mode): {:.2f}s - {:.2f}s (duration: {:.2f}s)".format(
        tracking_start - bag_start, tracking_end - bag_start, tracking_end - tracking_start))

else:
    raise ValueError(f"Invalid MODE: {MODE}. Use 'flag' or 'time'")

print("=" * 60)

start_time = rospy.Time(tracking_start)
end_time = rospy.Time(tracking_end)

# 外力推定データの読み込み
times_wrench = []
force_x, force_y, force_z = [], [], []
torque_x, torque_y, torque_z = [], [], []

for topic, msg, t in bag.read_messages(
        topics=['/kinikun1/estimated_external_wrench'],
        start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - tracking_start
    times_wrench.append(t_sec)
    
    # Force
    force_x.append(msg.wrench.force.x)
    force_y.append(msg.wrench.force.y)
    force_z.append(msg.wrench.force.z)
    
    # Torque
    torque_x.append(msg.wrench.torque.x)
    torque_y.append(msg.wrench.torque.y)
    torque_z.append(msg.wrench.torque.z)

bag.close()

# NumPy配列に変換
times_wrench = np.array(times_wrench)
force_x = np.array(force_x)
force_y = np.array(force_y)
force_z = np.array(force_z)
torque_x = np.array(torque_x)
torque_y = np.array(torque_y)
torque_z = np.array(torque_z)

print(f"Wrench data: {len(times_wrench)} samples")

# ============================================
# 統計計算
# ============================================
def calc_stats(data, name):
    rmse = np.sqrt(np.mean(data**2))
    mean = np.mean(data)
    std = np.std(data)
    max_abs = np.max(np.abs(data))
    print(f"  {name:10s}: Mean={mean:+.4f}, Std={std:.4f}, Max={max_abs:.4f}")
    return rmse, mean, std, max_abs

print("\n[Force Statistics]")
calc_stats(force_x, "Force X")
calc_stats(force_y, "Force Y")
calc_stats(force_z, "Force Z")

print("\n[Torque Statistics]")
calc_stats(torque_x, "Torque X")
calc_stats(torque_y, "Torque Y")
calc_stats(torque_z, "Torque Z")

# ============================================
# 図1: Force と Torque（2段プロット）
# ============================================
fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# (a) Force
axes1[0].plot(times_wrench, force_x, 'r-', linewidth=1, label='Force X', alpha=0.8)
axes1[0].plot(times_wrench, force_y, 'g-', linewidth=1, label='Force Y', alpha=0.8)
axes1[0].plot(times_wrench, force_z, 'b-', linewidth=1, label='Force Z', alpha=0.8)
axes1[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes1[0].set_ylabel('Force [N]', **label_font)
axes1[0].legend(loc='upper right', ncol=3, **legend_font)
axes1[0].grid(True, alpha=0.3)
axes1[0].set_xlim(0, None)
# axes1[0].set_ylim(-10, 10)  # 必要に応じてリミット設定

# (b) Torque
axes1[1].plot(times_wrench, torque_x, 'r-', linewidth=1, label='Torque X', alpha=0.8)
axes1[1].plot(times_wrench, torque_y, 'g-', linewidth=1, label='Torque Y', alpha=0.8)
axes1[1].plot(times_wrench, torque_z, 'b-', linewidth=1, label='Torque Z', alpha=0.8)
axes1[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes1[1].set_xlabel('Time [s]', **label_font)
axes1[1].set_ylabel('Torque [Nm]', **label_font)
axes1[1].legend(loc='upper right', ncol=3, **legend_font)
axes1[1].grid(True, alpha=0.3)
axes1[1].set_xlim(0, None)
# axes1[1].set_ylim(-5, 5)  # 必要に応じてリミット設定

plt.tight_layout()
fig1.savefig('fig_wrench_time.pdf', dpi=300, bbox_inches='tight')
print("\nSaved: fig_wrench_time.pdf")

# ============================================
# 図2: Force XYZ 個別プロット（3段）
# ============================================
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes2[0].plot(times_wrench, force_x, 'r-', linewidth=1)
axes2[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes2[0].set_ylabel('Force X [N]', **label_font)
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(times_wrench, force_y, 'g-', linewidth=1)
axes2[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes2[1].set_ylabel('Force Y [N]', **label_font)
axes2[1].grid(True, alpha=0.3)

axes2[2].plot(times_wrench, force_z, 'b-', linewidth=1)
axes2[2].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes2[2].set_xlabel('Time [s]', **label_font)
axes2[2].set_ylabel('Force Z [N]', **label_font)
axes2[2].grid(True, alpha=0.3)
axes2[2].set_xlim(0, None)

plt.tight_layout()
fig2.savefig('fig_force_individual.pdf', dpi=300, bbox_inches='tight')
print("Saved: fig_force_individual.pdf")

# ============================================
# 図3: Torque XYZ 個別プロット（3段）
# ============================================
fig3, axes3 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes3[0].plot(times_wrench, torque_x, 'r-', linewidth=1)
axes3[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes3[0].set_ylabel('Torque X [Nm]', **label_font)
axes3[0].grid(True, alpha=0.3)

axes3[1].plot(times_wrench, torque_y, 'g-', linewidth=1)
axes3[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes3[1].set_ylabel('Torque Y [Nm]', **label_font)
axes3[1].grid(True, alpha=0.3)

axes3[2].plot(times_wrench, torque_z, 'b-', linewidth=1)
axes3[2].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes3[2].set_xlabel('Time [s]', **label_font)
axes3[2].set_ylabel('Torque Z [Nm]', **label_font)
axes3[2].grid(True, alpha=0.3)
axes3[2].set_xlim(0, None)

plt.tight_layout()
fig3.savefig('fig_torque_individual.pdf', dpi=300, bbox_inches='tight')
print("Saved: fig_torque_individual.pdf")

# ============================================
# 完了
# ============================================
print("\n" + "=" * 60)
print("Figures saved:")
print("  - fig_wrench_time.pdf (Force & Torque combined)")
print("  - fig_force_individual.pdf (Force X/Y/Z separate)")
print("  - fig_torque_individual.pdf (Torque X/Y/Z separate)")
print("=" * 60)

plt.show()