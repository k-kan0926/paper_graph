#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

label_font = {'fontsize': 14, 'fontweight': 'bold'}
legend_font = {'fontsize': 10}

# ============================================
# パラメータ設定
# ============================================
BAG_FILE = '/home/keiichiro/document/paper_graph/rosbag_review/master_thesis/20251215/mppi.bag'
CIRCLE_RADIUS = 0.5  # 円の半径 [m]

# ★ 変形区間の時間設定（bag開始からのオフセット [秒]）
DEFORM_START_OFFSET = 130.0   # 例: bag開始から5秒後
DEFORM_END_OFFSET = 162.0    # 例: bag開始から15秒後
#t 100-125 i 55 90 bring 35 103 narrow 130 162
# 圧力変換係数
PRESSURE_COEFF = 0.9 / 4096.0

# ============================================
# データ読み込み
# ============================================
bag = rosbag.Bag(BAG_FILE, 'r')
bag_start = bag.get_start_time()
bag_end = bag.get_end_time()

print("Bag duration: {:.2f}s".format(bag_end - bag_start))

# ----- is_trackingフラグから円軌道区間を特定 -----
tracking_start = None
tracking_end = None

for topic, msg, t in bag.read_messages(topics=['/circle_trajectory_follow/is_tracking']):
    if msg.data and tracking_start is None:
        tracking_start = t.to_sec()
    elif not msg.data and tracking_start is not None and tracking_end is None:
        tracking_end = t.to_sec()

if tracking_start is None:
    tracking_start = bag_start
if tracking_end is None:
    tracking_end = bag_end

# 変形区間の絶対時刻
deform_start = bag_start + DEFORM_START_OFFSET
deform_end = bag_start + DEFORM_END_OFFSET

print("=" * 60)
print("Deformation period: {:.2f}s - {:.2f}s (duration: {:.2f}s)".format(
    DEFORM_START_OFFSET, DEFORM_END_OFFSET, DEFORM_END_OFFSET - DEFORM_START_OFFSET))
print("Circle tracking period: {:.2f}s - {:.2f}s (duration: {:.2f}s)".format(
    tracking_start - bag_start, tracking_end - bag_start, tracking_end - tracking_start))
print("=" * 60)

# ============================================
# 関数: データ読み込み
# ============================================
def read_pid_data(bag, start_time, end_time, time_offset):
    """PIDデータを読み込む"""
    times, err_x, err_y, err_z = [], [], [], []
    target_x, target_y, target_z = [], [], []
    actual_x, actual_y, actual_z = [], [], []
    
    for topic, msg, t in bag.read_messages(
            topics=['/kinikun1/debug/pose/pid'],
            start_time=rospy.Time(start_time), 
            end_time=rospy.Time(end_time)):
        t_sec = t.to_sec() - time_offset
        times.append(t_sec)
        err_x.append(msg.x.err_p)
        err_y.append(msg.y.err_p)
        err_z.append(msg.z.err_p)
        target_x.append(msg.x.target_p)
        target_y.append(msg.y.target_p)
        target_z.append(msg.z.target_p)
        actual_x.append(msg.x.target_p - msg.x.err_p)
        actual_y.append(msg.y.target_p - msg.y.err_p)
        actual_z.append(msg.z.target_p - msg.z.err_p)
    
    return {
        'times': np.array(times),
        'err_x': np.array(err_x), 'err_y': np.array(err_y), 'err_z': np.array(err_z),
        'target_x': np.array(target_x), 'target_y': np.array(target_y), 'target_z': np.array(target_z),
        'actual_x': np.array(actual_x), 'actual_y': np.array(actual_y), 'actual_z': np.array(actual_z),
    }

def read_odom_data(bag, start_time, end_time, time_offset):
    """Odometryデータを読み込む（姿勢）"""
    times, roll, pitch, yaw = [], [], [], []
    
    for topic, msg, t in bag.read_messages(
            topics=['/kinikun1/uav/baselink/odom'],
            start_time=rospy.Time(start_time), 
            end_time=rospy.Time(end_time)):
        t_sec = t.to_sec() - time_offset
        times.append(t_sec)
        
        # クォータニオンからオイラー角に変換
        q = msg.pose.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        r, p, y = euler_from_quaternion(quaternion)
        
        roll.append(r)
        pitch.append(p)
        yaw.append(y)
    
    return {
        'times': np.array(times),
        'roll': np.array(roll), 'pitch': np.array(pitch), 'yaw': np.array(yaw),
    }

def read_pressure_data(bag, start_time, end_time, time_offset):
    """圧力データを読み込む"""
    times = []
    pressure_x, pressure_y, pressure_z, pressure_w = [], [], [], []
    
    for topic, msg, t in bag.read_messages(
            topics=['/mpa_cmd'],
            start_time=rospy.Time(start_time), 
            end_time=rospy.Time(end_time)):
        t_sec = t.to_sec() - time_offset
        times.append(t_sec)
        pressure_x.append(msg.x * PRESSURE_COEFF)
        pressure_y.append(msg.y * PRESSURE_COEFF)
        pressure_z.append(msg.z * PRESSURE_COEFF)
        pressure_w.append(msg.w * PRESSURE_COEFF)
    
    return {
        'times': np.array(times),
        'x': np.array(pressure_x), 'y': np.array(pressure_y),
        'z': np.array(pressure_z), 'w': np.array(pressure_w),
    }

def read_joint_data(bag, start_time, end_time, time_offset):
    """関節角度データを読み込む"""
    times = []
    joint1, joint2, joint3, joint4 = [], [], [], []
    
    for topic, msg, t in bag.read_messages(
            topics=['/kinikun1/joint_states'],
            start_time=rospy.Time(start_time), 
            end_time=rospy.Time(end_time)):
        t_sec = t.to_sec() - time_offset
        times.append(t_sec)
        joint1.append(msg.position[0])
        joint2.append(msg.position[1])
        joint3.append(msg.position[2])
        joint4.append(msg.position[3])
    
    return {
        'times': np.array(times),
        'joint1': np.array(joint1), 'joint2': np.array(joint2),
        'joint3': np.array(joint3), 'joint4': np.array(joint4),
    }

# ============================================
# データ読み込み実行
# ============================================
# 変形区間
deform_pid = read_pid_data(bag, deform_start, deform_end, deform_start)
deform_odom = read_odom_data(bag, deform_start, deform_end, deform_start)
deform_pressure = read_pressure_data(bag, deform_start, deform_end, deform_start)
deform_joint = read_joint_data(bag, deform_start, deform_end, deform_start)

# 円軌道区間
circle_pid = read_pid_data(bag, tracking_start, tracking_end, tracking_start)
circle_odom = read_odom_data(bag, tracking_start, tracking_end, tracking_start)

bag.close()

# ============================================
# RMSE計算関数
# ============================================
def calc_rmse(data):
    return np.sqrt(np.mean(data**2))

def calc_stats(data):
    rmse = calc_rmse(data)
    mean = np.mean(data)
    std = np.std(data)
    max_abs = np.max(np.abs(data))
    return rmse, mean, std, max_abs

# ============================================
# 図1: 変形中のグラフ
# ============================================
fig1, axes1 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# (a) 位置誤差
axes1[0].plot(deform_pid['times'], deform_pid['err_x'], label='X', color='tab:red', linewidth=1)
axes1[0].plot(deform_pid['times'], deform_pid['err_y'], label='Y', color='tab:blue', linewidth=1)
axes1[0].plot(deform_pid['times'], deform_pid['err_z'], label='Z', color='tab:green', linewidth=1)
axes1[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes1[0].set_ylabel('Position Error [m]', **label_font)
axes1[0].legend(loc='upper right', ncol=3, **legend_font)
axes1[0].grid(True, alpha=0.3)
axes1[0].set_ylim(-0.55, 0.55)

# (b) 姿勢誤差
axes1[1].plot(deform_odom['times'], np.rad2deg(deform_odom['roll']), label='Roll', color='tab:red', linewidth=1)
axes1[1].plot(deform_odom['times'], np.rad2deg(deform_odom['pitch']), label='Pitch', color='tab:blue', linewidth=1)
axes1[1].plot(deform_odom['times'], np.rad2deg(deform_odom['yaw']), label='Yaw', color='tab:green', linewidth=1)
axes1[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes1[1].set_ylabel('Attitude Error [deg]', **label_font)
axes1[1].legend(loc='upper right', ncol=3, **legend_font)
axes1[1].grid(True, alpha=0.3)
axes1[1].set_ylim(-18, 18)
# (c) 圧力
axes1[2].plot(deform_pressure['times'], deform_pressure['x'], label='Pressure X', color='tab:red', linewidth=1)
axes1[2].plot(deform_pressure['times'], deform_pressure['y'], label='Pressure Y', color='tab:blue', linewidth=1)
axes1[2].plot(deform_pressure['times'], deform_pressure['z'], label='Pressure Z', color='tab:green', linewidth=1)
axes1[2].plot(deform_pressure['times'], deform_pressure['w'], label='Pressure W', color='tab:purple', linewidth=1)
axes1[2].set_ylabel('Pressure [MPa]', **label_font)
axes1[2].legend(loc='upper right', ncol=4, **legend_font)
axes1[2].grid(True, alpha=0.3)
axes1[2].set_ylim(0, 0.7)
# (d) 関節角度
axes1[3].plot(deform_joint['times'], np.rad2deg(deform_joint['joint1']), label='Joint 1 (X+/Y-)', color='tab:red', linewidth=1)
axes1[3].plot(deform_joint['times'], np.rad2deg(deform_joint['joint2']), label='Joint 2 (W+/Z-)', color='tab:blue', linewidth=1)
axes1[3].plot(deform_joint['times'], np.rad2deg(deform_joint['joint3']), label='Joint 3 (Z+/W-)', color='tab:green', linewidth=1)
axes1[3].plot(deform_joint['times'], np.rad2deg(deform_joint['joint4']), label='Joint 4 (Y+/X-)', color='tab:purple', linewidth=1)
axes1[3].set_xlabel('Time [s]', **label_font)
axes1[3].set_ylabel('Joint Angle [deg]', **label_font)
axes1[3].legend(loc='upper right', ncol=2, **legend_font)
axes1[3].grid(True, alpha=0.3)
axes1[3].set_ylim(-40, 40)

plt.tight_layout()
fig1.savefig('fig1_deformation.pdf', dpi=300, bbox_inches='tight')

# ============================================
# 図2: 円軌道中の誤差（時系列）
# ============================================
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# (a) 位置誤差
axes2[0].plot(circle_pid['times'], circle_pid['err_x'], label='X', color='tab:red', linewidth=1)
axes2[0].plot(circle_pid['times'], circle_pid['err_y'], label='Y', color='tab:blue', linewidth=1)
axes2[0].plot(circle_pid['times'], circle_pid['err_z'], label='Z', color='tab:green', linewidth=1)
axes2[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes2[0].set_ylabel('Position Error [m]', **label_font)
axes2[0].legend(loc='upper right', ncol=3, **legend_font)
axes2[0].grid(True, alpha=0.3)
axes2[0].set_ylim(-0.2, 0.2)
# (b) 姿勢誤差
axes2[1].plot(circle_odom['times'], np.rad2deg(circle_odom['roll']), label='Roll', color='tab:red', linewidth=1)
axes2[1].plot(circle_odom['times'], np.rad2deg(circle_odom['pitch']), label='Pitch', color='tab:blue', linewidth=1)
axes2[1].plot(circle_odom['times'], np.rad2deg(circle_odom['yaw']), label='Yaw', color='tab:green', linewidth=1)
axes2[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes2[1].set_xlabel('Time [s]', **label_font)
axes2[1].set_ylabel('Attitude Error [deg]', **label_font)
axes2[1].legend(loc='upper right', ncol=3, **legend_font)
axes2[1].grid(True, alpha=0.3)
axes2[1].set_ylim(-12, 12)
plt.tight_layout()
fig2.savefig('fig2_circle_error.pdf', dpi=300, bbox_inches='tight')

# ============================================
# 図3: 円軌道のXY平面図（目標 vs 実測）
# ============================================
fig3, ax3 = plt.subplots(figsize=(8, 8))

# 目標軌道
ax3.plot(circle_pid['target_x'], circle_pid['target_y'], 'k--', linewidth=2, zorder=1)

# 実軌道（時間でカラーマップ）
scatter = ax3.scatter(circle_pid['actual_x'], circle_pid['actual_y'], 
                      c=circle_pid['times'], cmap='viridis', s=5, zorder=2)
cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
cbar.set_label('Time [s]')

# 開始・終了点
ax3.plot(circle_pid['actual_x'][0], circle_pid['actual_y'][0], 'go', markersize=12, label='Start', zorder=3)
ax3.plot(circle_pid['actual_x'][-1], circle_pid['actual_y'][-1], 'ro', markersize=12, label='End', zorder=3)

# 中心点
center_x = np.mean(circle_pid['target_x'])
center_y = np.mean(circle_pid['target_y'])
ax3.plot(center_x, center_y, 'k+', markersize=15, markeredgewidth=2, label='Center')

ax3.set_xlabel('X [m]', **label_font)
ax3.set_ylabel('Y [m]', **label_font)
ax3.set_aspect('equal')
ax3.legend(loc='upper right', **legend_font)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig('fig3_circle_xy.pdf', dpi=300, bbox_inches='tight')

# ============================================
# 図4: RMSE統計表
# ============================================
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 6))


# ----- 左: 変形中の統計 -----
axes4[0].axis('off')
deform_table_data = [
    ['Error Type', 'RMSE', 'Max'],
    ['X [m]', '{:.4f}'.format(calc_rmse(deform_pid['err_x'])), '{:.4f}'.format(np.max(np.abs(deform_pid['err_x'])))],
    ['Y [m]', '{:.4f}'.format(calc_rmse(deform_pid['err_y'])), '{:.4f}'.format(np.max(np.abs(deform_pid['err_y'])))],
    ['Z [m]', '{:.4f}'.format(calc_rmse(deform_pid['err_z'])), '{:.4f}'.format(np.max(np.abs(deform_pid['err_z'])))],
    ['Roll [deg]', '{:.2f}'.format(np.rad2deg(calc_rmse(deform_odom['roll']))), '{:.2f}'.format(np.rad2deg(np.max(np.abs(deform_odom['roll']))))],
    ['Pitch [deg]', '{:.2f}'.format(np.rad2deg(calc_rmse(deform_odom['pitch']))), '{:.2f}'.format(np.rad2deg(np.max(np.abs(deform_odom['pitch']))))],
    ['Yaw [deg]', '{:.2f}'.format(np.rad2deg(calc_rmse(deform_odom['yaw']))), '{:.2f}'.format(np.rad2deg(np.max(np.abs(deform_odom['yaw']))))],
]

table1 = axes4[0].table(cellText=deform_table_data, loc='center', cellLoc='center',
                         colWidths=[0.4, 0.3, 0.3])
table1.auto_set_font_size(False)
table1.set_fontsize(11)
table1.scale(1.0, 1.6)
for j in range(3):
    table1[(0, j)].set_text_props(fontweight='bold')
    table1[(0, j)].set_facecolor('#E6E6E6')
axes4[0].set_title('RMSE during Deformation', fontweight='bold', fontsize=14, pad=20)

# ----- 右: 円軌道中の統計 -----
axes4[1].axis('off')

# XY平面誤差
xy_error = np.sqrt(circle_pid['err_x']**2 + circle_pid['err_y']**2)

circle_table_data = [
    ['Error Type', 'RMSE', 'Max'],
    ['X [m]', '{:.4f}'.format(calc_rmse(circle_pid['err_x'])), '{:.4f}'.format(np.max(np.abs(circle_pid['err_x'])))],
    ['Y [m]', '{:.4f}'.format(calc_rmse(circle_pid['err_y'])), '{:.4f}'.format(np.max(np.abs(circle_pid['err_y'])))],
    ['Z [m]', '{:.4f}'.format(calc_rmse(circle_pid['err_z'])), '{:.4f}'.format(np.max(np.abs(circle_pid['err_z'])))],
    ['XY [m]', '{:.4f}'.format(calc_rmse(xy_error)), '{:.4f}'.format(np.max(xy_error))],
    ['Roll [deg]', '{:.2f}'.format(np.rad2deg(calc_rmse(circle_odom['roll']))), '{:.2f}'.format(np.rad2deg(np.max(np.abs(circle_odom['roll']))))],
    ['Pitch [deg]', '{:.2f}'.format(np.rad2deg(calc_rmse(circle_odom['pitch']))), '{:.2f}'.format(np.rad2deg(np.max(np.abs(circle_odom['pitch']))))],
    ['Yaw [deg]', '{:.2f}'.format(np.rad2deg(calc_rmse(circle_odom['yaw']))), '{:.2f}'.format(np.rad2deg(np.max(np.abs(circle_odom['yaw']))))],
]

table2 = axes4[1].table(cellText=circle_table_data, loc='center', cellLoc='center',
                         colWidths=[0.4, 0.3, 0.3])
table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1.0, 1.6)
for j in range(3):
    table2[(0, j)].set_text_props(fontweight='bold')
    table2[(0, j)].set_facecolor('#E6E6E6')
axes4[1].set_title('RMSE during Circle Tracking', fontweight='bold', fontsize=14, pad=20)
plt.subplots_adjust(wspace=0.4)
plt.tight_layout()
fig4.savefig('fig4_rmse_table.pdf', dpi=300, bbox_inches='tight')

# ============================================
# コンソール出力
# ============================================
print("\n" + "=" * 60)
print("STATISTICS SUMMARY")
print("=" * 60)

print("\n[Deformation Period]")
print("  Position Error:")
print("    X: RMSE={:.4f}m, Max={:.4f}m".format(calc_rmse(deform_pid['err_x']), np.max(np.abs(deform_pid['err_x']))))
print("    Y: RMSE={:.4f}m, Max={:.4f}m".format(calc_rmse(deform_pid['err_y']), np.max(np.abs(deform_pid['err_y']))))
print("    Z: RMSE={:.4f}m, Max={:.4f}m".format(calc_rmse(deform_pid['err_z']), np.max(np.abs(deform_pid['err_z']))))

print("\n[Circle Tracking Period]")
print("  Position Error:")
print("    X:  RMSE={:.4f}m, Max={:.4f}m".format(calc_rmse(circle_pid['err_x']), np.max(np.abs(circle_pid['err_x']))))
print("    Y:  RMSE={:.4f}m, Max={:.4f}m".format(calc_rmse(circle_pid['err_y']), np.max(np.abs(circle_pid['err_y']))))
print("    Z:  RMSE={:.4f}m, Max={:.4f}m".format(calc_rmse(circle_pid['err_z']), np.max(np.abs(circle_pid['err_z']))))
print("    XY: RMSE={:.4f}m, Max={:.4f}m".format(calc_rmse(xy_error), np.max(xy_error)))

print("\n" + "=" * 60)
print("Figures saved:")
print("  - fig1_deformation.pdf")
print("  - fig2_circle_error.pdf")
print("  - fig3_circle_xy.pdf")
print("  - fig4_rmse_table.pdf")
print("=" * 60)

plt.show()