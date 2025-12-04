#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================
# フォント設定（論文用）
# ============================================
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "pdf.use14corefonts": False,
    "font.family": "serif",  # Times New Romanがない場合の代替
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
BAG_FILE = '/home/kan/Documents/rosbag/20251204/t_circle.bag'
CIRCLE_RADIUS = 0.7  # 円の半径 [m]

# 時間範囲（Noneの場合はis_trackingフラグを使用）
TIME_OFFSET_START = None
TIME_OFFSET_END = None

# ============================================
# データ読み込み
# ============================================
bag = rosbag.Bag(BAG_FILE, 'r')
bag_start = bag.get_start_time()

# is_trackingフラグから追従区間を特定
tracking_start = None
tracking_end = None

for topic, msg, t in bag.read_messages(topics=['/circle_trajectory_follow/is_tracking']):
    if msg.data and tracking_start is None:
        tracking_start = t.to_sec()
    elif not msg.data and tracking_start is not None and tracking_end is None:
        tracking_end = t.to_sec()

# フラグがない場合は手動設定を使用
if tracking_start is None:
    if TIME_OFFSET_START is not None:
        tracking_start = bag_start + TIME_OFFSET_START
        tracking_end = bag_start + TIME_OFFSET_END
    else:
        tracking_start = bag_start
        tracking_end = bag.get_end_time()
else:
    if tracking_end is None:
        tracking_end = bag.get_end_time()

print("=" * 50)
print("Tracking period: {:.2f}s - {:.2f}s (duration: {:.2f}s)".format(
    tracking_start - bag_start, tracking_end - bag_start, tracking_end - tracking_start))
print("=" * 50)

start_time = rospy.Time(tracking_start)
end_time = rospy.Time(tracking_end)

# PID データ（位置誤差と目標位置）
times_pid = []
err_x = []; err_y = []; err_z = []
target_x = []; target_y = []
actual_x = []; actual_y = []

for topic, msg, t in bag.read_messages(topics=['/kinikun1/debug/pose/pid'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - tracking_start
    times_pid.append(t_sec)
    err_x.append(msg.x.err_p)
    err_y.append(msg.y.err_p)
    err_z.append(msg.z.err_p)
    target_x.append(msg.x.target_p)
    target_y.append(msg.y.target_p)
    actual_x.append(msg.x.target_p - msg.x.err_p)
    actual_y.append(msg.y.target_p - msg.y.err_p)

# Odometry データ（姿勢誤差）
times_odom = []
roll_vals = []; pitch_vals = []; yaw_vals = []

for topic, msg, t in bag.read_messages(topics=['/kinikun1/uav/baselink/odom'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - tracking_start
    times_odom.append(t_sec)
    roll_vals.append(msg.pose.pose.orientation.x)
    pitch_vals.append(msg.pose.pose.orientation.y)
    yaw_vals.append(msg.pose.pose.orientation.z)

bag.close()

# NumPy配列に変換
times_pid = np.array(times_pid)
err_x = np.array(err_x)
err_y = np.array(err_y)
err_z = np.array(err_z)
target_x = np.array(target_x)
target_y = np.array(target_y)
actual_x = np.array(actual_x)
actual_y = np.array(actual_y)

times_odom = np.array(times_odom)
roll_vals = np.array(roll_vals)
pitch_vals = np.array(pitch_vals)
yaw_vals = np.array(yaw_vals)

# ============================================
# 円軌道特有の誤差計算
# ============================================
# 円の中心を推定（目標軌道の平均）
center_x = np.mean(target_x)
center_y = np.mean(target_y)

# 半径方向誤差と接線方向誤差
radial_error = []
tangential_error = []
theta_actual = []

for i in range(len(actual_x)):
    dx = actual_x[i] - center_x
    dy = actual_y[i] - center_y
    actual_radius = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    theta_actual.append(theta)
    
    radial_error.append(actual_radius - CIRCLE_RADIUS)
    
    target_theta = np.arctan2(target_y[i] - center_y, target_x[i] - center_x)
    tangential_error.append(CIRCLE_RADIUS * np.arctan2(np.sin(theta - target_theta), 
                                                        np.cos(theta - target_theta)))

radial_error = np.array(radial_error)
tangential_error = np.array(tangential_error)
theta_actual = np.array(theta_actual)

# XY平面での誤差（ユークリッド）
xy_error = np.sqrt(err_x**2 + err_y**2)

# ============================================
# RMSE計算
# ============================================
def calc_rmse(data):
    return np.sqrt(np.mean(data**2))

def calc_stats(data, name):
    rmse = calc_rmse(data)
    mean = np.mean(data)
    std = np.std(data)
    max_abs = np.max(np.abs(data))
    return rmse, mean, std, max_abs

print("\n" + "=" * 60)
print("RMSE and Statistics (during circle tracking)")
print("=" * 60)

print("\n[Position Error]")
for name, data in [('X', err_x), ('Y', err_y), ('Z', err_z), ('XY (Euclidean)', xy_error)]:
    rmse, mean, std, max_abs = calc_stats(data, name)
    print("  {:15s}: RMSE={:7.4f}m, Mean={:+7.4f}m, Std={:7.4f}m, Max={:7.4f}m".format(
        name, rmse, mean, std, max_abs))

print("\n[Circle-specific Error]")
for name, data in [('Radial', radial_error), ('Tangential', tangential_error)]:
    rmse, mean, std, max_abs = calc_stats(data, name)
    print("  {:15s}: RMSE={:7.4f}m, Mean={:+7.4f}m, Std={:7.4f}m, Max={:7.4f}m".format(
        name, rmse, mean, std, max_abs))

print("\n[Attitude Error]")
for name, data in [('Roll', roll_vals), ('Pitch', pitch_vals), ('Yaw', yaw_vals)]:
    rmse, mean, std, max_abs = calc_stats(data, name)
    print("  {:15s}: RMSE={:7.4f}rad ({:5.2f}deg), Max={:7.4f}rad ({:5.2f}deg)".format(
        name, rmse, np.rad2deg(rmse), max_abs, np.rad2deg(max_abs)))

print("=" * 60 + "\n")

# ============================================
# 図1: XY平面での軌道追従
# ============================================
fig1, ax1 = plt.subplots(figsize=(7, 7))

# 目標円軌道
theta_ref = np.linspace(0, 2*np.pi, 100)
circle_x = center_x + CIRCLE_RADIUS * np.cos(theta_ref)
circle_y = center_y + CIRCLE_RADIUS * np.sin(theta_ref)
ax1.plot(circle_x, circle_y, 'k--', linewidth=2, label='Reference', zorder=1)

# 実軌道（時間でカラーマップ）
scatter = ax1.scatter(actual_x, actual_y, c=times_pid, cmap='viridis', 
                      s=3, label='Actual', zorder=2)
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
cbar.set_label('Time [s]')

# 開始・終了点
ax1.plot(actual_x[0], actual_y[0], 'go', markersize=10, label='Start', zorder=3)
ax1.plot(actual_x[-1], actual_y[-1], 'ro', markersize=10, label='End', zorder=3)

# 中心点
ax1.plot(center_x, center_y, 'k+', markersize=15, markeredgewidth=2, label='Center')

ax1.set_xlabel('X [m]', **label_font)
ax1.set_ylabel('Y [m]', **label_font)
ax1.set_aspect('equal')
ax1.legend(loc='upper right', **legend_font)
ax1.grid(True, alpha=0.3)
ax1.set_title('Circle Trajectory Tracking (XY Plane)', fontweight='bold')

# 誤差の拡大図（インセット）
axins = inset_axes(ax1, width="35%", height="35%", loc='lower left', borderpad=2)
axins.plot(circle_x, circle_y, 'k--', linewidth=1.5)
axins.scatter(actual_x, actual_y, c=times_pid, cmap='viridis', s=2)
x_center_view = center_x + CIRCLE_RADIUS * 0.7
y_center_view = center_y + CIRCLE_RADIUS * 0.7
view_range = CIRCLE_RADIUS * 0.3
axins.set_xlim(x_center_view - view_range, x_center_view + view_range)
axins.set_ylim(y_center_view - view_range, y_center_view + view_range)
axins.set_aspect('equal')
axins.grid(True, alpha=0.3)
axins.set_title('Zoomed', fontsize=9)

plt.tight_layout()

# ============================================
# 図2: 時系列誤差プロット（全てm単位）
# ============================================
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 位置誤差 [m]
axes2[0].plot(times_pid, err_x, label='X', color='tab:red', linewidth=1)
axes2[0].plot(times_pid, err_y, label='Y', color='tab:blue', linewidth=1)
axes2[0].plot(times_pid, err_z, label='Z', color='tab:green', linewidth=1)
axes2[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes2[0].set_ylabel('Position Error [m]', **label_font)
axes2[0].legend(loc='upper right', ncol=3, **legend_font)
axes2[0].grid(True, alpha=0.3)
axes2[0].set_title('Tracking Error during Circle Trajectory', fontweight='bold')

# 半径方向・接線方向誤差 [m]
axes2[1].plot(times_pid, radial_error, label='Radial', color='tab:purple', linewidth=1)
axes2[1].plot(times_pid, tangential_error, label='Tangential', color='tab:orange', linewidth=1)
axes2[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes2[1].set_ylabel('Circle Error [m]', **label_font)
axes2[1].legend(loc='upper right', ncol=2, **legend_font)
axes2[1].grid(True, alpha=0.3)

# 姿勢誤差 [deg]
axes2[2].plot(times_odom, np.rad2deg(roll_vals), label='Roll', color='tab:red', linewidth=1)
axes2[2].plot(times_odom, np.rad2deg(pitch_vals), label='Pitch', color='tab:blue', linewidth=1)
axes2[2].plot(times_odom, np.rad2deg(yaw_vals), label='Yaw', color='tab:green', linewidth=1)
axes2[2].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes2[2].set_xlabel('Time [s]', **label_font)
axes2[2].set_ylabel('Attitude Error [deg]', **label_font)
axes2[2].legend(loc='upper right', ncol=3, **legend_font)
axes2[2].grid(True, alpha=0.3)

plt.tight_layout()

# ============================================
# 図3: 極座標での誤差分布（全てm単位）
# ============================================
fig3 = plt.figure(figsize=(12, 5))

# 左: 角度 vs 半径方向誤差 [m]
ax3a = fig3.add_subplot(121)
scatter3a = ax3a.scatter(np.rad2deg(theta_actual), radial_error, 
                         c=times_pid, cmap='viridis', s=5, alpha=0.7)
ax3a.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3a.set_xlabel('Angle [deg]', **label_font)
ax3a.set_ylabel('Radial Error [m]', **label_font)
ax3a.set_xlim(-180, 180)
ax3a.grid(True, alpha=0.3)
ax3a.set_title('Radial Error vs Angle', fontweight='bold')
cbar3a = plt.colorbar(scatter3a, ax=ax3a)
cbar3a.set_label('Time [s]')

# 右: 誤差の極座標プロット [m]
ax3b = fig3.add_subplot(122, projection='polar')
scatter3b = ax3b.scatter(theta_actual, np.abs(radial_error), 
                         c=times_pid, cmap='viridis', s=5, alpha=0.7)
ax3b.set_title('Radial Error Distribution\n(Polar)', fontweight='bold', pad=15)
cbar3b = plt.colorbar(scatter3b, ax=ax3b, shrink=0.8)
cbar3b.set_label('Time [s]')

plt.tight_layout()

# ============================================
# 図4: 誤差統計（Box plot + RMSE表）（全てm単位）
# ============================================
fig4, axes4 = plt.subplots(1, 2, figsize=(11, 5))

# 左: Box plot [m]
box_data = [err_x, err_y, err_z, radial_error, tangential_error]
box_labels = ['X', 'Y', 'Z', 'Radial', 'Tangential']
bp = axes4[0].boxplot(box_data, labels=box_labels, patch_artist=True)
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes4[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes4[0].set_ylabel('Error [m]', **label_font)
axes4[0].set_title('Error Distribution', fontweight='bold')
axes4[0].grid(True, alpha=0.3, axis='y')

# 右: RMSE表 [m]
axes4[1].axis('off')
table_data = [
    ['Error Type', 'RMSE [m]', 'Max [m]'],
    ['X', '{:.4f}'.format(calc_rmse(err_x)), '{:.4f}'.format(np.max(np.abs(err_x)))],
    ['Y', '{:.4f}'.format(calc_rmse(err_y)), '{:.4f}'.format(np.max(np.abs(err_y)))],
    ['Z', '{:.4f}'.format(calc_rmse(err_z)), '{:.4f}'.format(np.max(np.abs(err_z)))],
    ['XY', '{:.4f}'.format(calc_rmse(xy_error)), '{:.4f}'.format(np.max(xy_error))],
    ['Radial', '{:.4f}'.format(calc_rmse(radial_error)), '{:.4f}'.format(np.max(np.abs(radial_error)))],
    ['Tangential', '{:.4f}'.format(calc_rmse(tangential_error)), '{:.4f}'.format(np.max(np.abs(tangential_error)))],
]
table = axes4[1].table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.35, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# ヘッダー行を太字に
for j in range(3):
    table[(0, j)].set_text_props(fontweight='bold')
    table[(0, j)].set_facecolor('#E6E6E6')

axes4[1].set_title('RMSE Summary', fontweight='bold', pad=20)

plt.tight_layout()

# ============================================
# 保存
# ============================================
fig1.savefig('circle_trajectory_xy.pdf', dpi=300, bbox_inches='tight')
fig2.savefig('circle_trajectory_error_time.pdf', dpi=300, bbox_inches='tight')
fig3.savefig('circle_trajectory_polar.pdf', dpi=300, bbox_inches='tight')
fig4.savefig('circle_trajectory_statistics.pdf', dpi=300, bbox_inches='tight')

print("Figures saved as PDF files.")

plt.show()