#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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
BAG_FILE = '/home/keiichiro/document/paper_graph/rosbag_review/master_thesis/20251221/bringt_master.bag'

# ★ 変形区間の時間設定（bag開始からのオフセット [秒]）
DEFORM_START_OFFSET = 35.0
DEFORM_END_OFFSET =103.0

# ============================================
# データ読み込み
# ============================================
bag = rosbag.Bag(BAG_FILE, 'r')
bag_start = bag.get_start_time()
bag_end = bag.get_end_time()

print("Bag duration: {:.2f}s".format(bag_end - bag_start))

# 変形区間の絶対時刻
deform_start = bag_start + DEFORM_START_OFFSET
deform_end = bag_start + DEFORM_END_OFFSET

print("=" * 60)
print("Deformation period: {:.2f}s - {:.2f}s (duration: {:.2f}s)".format(
    DEFORM_START_OFFSET, DEFORM_END_OFFSET, DEFORM_END_OFFSET - DEFORM_START_OFFSET))
print("=" * 60)

# ============================================
# 関数: データ読み込み
# ============================================
def read_pid_data(bag, start_time, end_time, time_offset):
    """PIDデータを読み込む"""
    times, err_x, err_y, err_z = [], [], [], []
    
    for topic, msg, t in bag.read_messages(
            topics=['/kinikun1/debug/pose/pid'],
            start_time=rospy.Time(start_time), 
            end_time=rospy.Time(end_time)):
        t_sec = t.to_sec() - time_offset
        times.append(t_sec)
        err_x.append(msg.x.err_p)
        err_y.append(msg.y.err_p)
        err_z.append(msg.z.err_p)
    
    return {
        'times': np.array(times),
        'err_x': np.array(err_x), 'err_y': np.array(err_y), 'err_z': np.array(err_z),
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

# ============================================
# データ読み込み実行
# ============================================
deform_pid = read_pid_data(bag, deform_start, deform_end, deform_start)
deform_odom = read_odom_data(bag, deform_start, deform_end, deform_start)

bag.close()

# ============================================
# 図: 変形中の位置誤差と姿勢誤差のみ
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# (a) 位置誤差
axes[0].plot(deform_pid['times'], deform_pid['err_x'], label='X', color='tab:red', linewidth=1)
axes[0].plot(deform_pid['times'], deform_pid['err_y'], label='Y', color='tab:blue', linewidth=1)
axes[0].plot(deform_pid['times'], deform_pid['err_z'], label='Z', color='tab:green', linewidth=1)
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('Position Error [m]', **label_font)
axes[0].legend(loc='upper right', ncol=3, **legend_font)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.55, 0.55)

# (b) 姿勢誤差
axes[1].plot(deform_odom['times'], np.rad2deg(deform_odom['roll']), label='Roll', color='tab:red', linewidth=1)
axes[1].plot(deform_odom['times'], np.rad2deg(deform_odom['pitch']), label='Pitch', color='tab:blue', linewidth=1)
axes[1].plot(deform_odom['times'], np.rad2deg(deform_odom['yaw']), label='Yaw', color='tab:green', linewidth=1)
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes[1].set_xlabel('Time [s]', **label_font)
axes[1].set_ylabel('Attitude Error [deg]', **label_font)
axes[1].legend(loc='upper right', ncol=3, **legend_font)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-18, 18)

plt.tight_layout()
fig.savefig('fig1_position_attitude.pdf', dpi=300, bbox_inches='tight')

print("\nFigure saved: fig1_position_attitude.pdf")

plt.show()