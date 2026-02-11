import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "pdf.use14corefonts": False,
    "font.family": "Times New Roman",
})

mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Times New Roman'
label_font = {'fontsize': 16, 'fontweight': 'bold'}
legend_font = {'fontsize': 12}

# =============================================================================
# データ読み込み
# =============================================================================
bag = rosbag.Bag('/home/keiichiro/document/paper_graph/rosbag_review/master_thesis/20251215/narrow_path2.bag', 'r')
bag_start = bag.get_start_time()
start_time = rospy.Time(bag_start + 148.0)
end_time   = rospy.Time(bag_start + 162.0)
#T 95,135, H  50,110 X 200,235

# Joint States
times_js = []
arm1 = []; arm2 = []; arm3 = []; arm4 = []
for topic, msg, t in bag.read_messages(topics=['/kinikun1/joint_states'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - bag_start
    times_js.append(t_sec)
    if len(msg.position) >= 4:
        arm1.append(msg.position[0])
        arm2.append(msg.position[1])
        arm3.append(msg.position[2])
        arm4.append(msg.position[3])

# PID
times_pid = []
err_x = []; err_y = []; err_z = []
for topic, msg, t in bag.read_messages(topics=['/kinikun1/debug/pose/pid'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - bag_start
    times_pid.append(t_sec)
    err_x.append(msg.x.err_p)
    err_y.append(msg.y.err_p)
    err_z.append(msg.z.err_p)

# Odometry
times_odom = []
roll_vals = []; pitch_vals = []; yaw_vals = []
for topic, msg, t in bag.read_messages(topics=['/kinikun1/uav/baselink/odom'],
                                       start_time=start_time, end_time=end_time):
    t_sec = t.to_sec() - bag_start
    times_odom.append(t_sec)
    roll_vals.append(msg.pose.pose.orientation.x)
    pitch_vals.append(msg.pose.pose.orientation.y)
    yaw_vals.append(msg.pose.pose.orientation.z)

bag.close()

# NumPy配列に変換
times_js = np.array(times_js)
arm1, arm2, arm3, arm4 = np.array(arm1), np.array(arm2), np.array(arm3), np.array(arm4)
times_pid = np.array(times_pid)
err_x, err_y, err_z = np.array(err_x), np.array(err_y), np.array(err_z)
times_odom = np.array(times_odom)
roll_vals, pitch_vals, yaw_vals = np.array(roll_vals), np.array(pitch_vals), np.array(yaw_vals)
# グラフ①: Joint State
arm1 = np.rad2deg(arm1)
arm2 = np.rad2deg(arm2)
arm3 = np.rad2deg(arm3)
arm4 = np.rad2deg(arm4)

# グラフ③: Roll, Pitch, Yaw
roll_vals = np.rad2deg(roll_vals)
pitch_vals = np.rad2deg(pitch_vals)
yaw_vals = np.rad2deg(yaw_vals)
# =============================================================================
# アニメーション設定
# =============================================================================
# 時間範囲（絶対時間）
t_min_abs = min(times_js.min() if len(times_js) else np.inf,
                times_pid.min() if len(times_pid) else np.inf,
                times_odom.min() if len(times_odom) else np.inf)
t_max_abs = max(times_js.max() if len(times_js) else -np.inf,
                times_pid.max() if len(times_pid) else -np.inf,
                times_odom.max() if len(times_odom) else -np.inf)

# 相対時間に変換（開始時間を0とする）
times_js = times_js - t_min_abs
times_pid = times_pid - t_min_abs
times_odom = times_odom - t_min_abs

# 相対時間での範囲
t_min = 0.0
t_max = t_max_abs - t_min_abs

# Y軸の範囲を事前計算
y1_min = min(arm1.min(), arm2.min(), arm3.min(), arm4.min()) if len(arm1) else -1
y1_max = max(arm1.max(), arm2.max(), arm3.max(), arm4.max()) if len(arm1) else 1
y2_min = min(err_x.min(), err_y.min(), err_z.min()) if len(err_x) else -1
y2_max = max(err_x.max(), err_y.max(), err_z.max()) if len(err_x) else 1
y3_min = min(roll_vals.min(), pitch_vals.min(), yaw_vals.min()) if len(roll_vals) else -1
y3_max = max(roll_vals.max(), pitch_vals.max(), yaw_vals.max()) if len(roll_vals) else 1

# マージンを追加
margin = 0.1
y1_range = y1_max - y1_min
y2_range = y2_max - y2_min
y3_range = y3_max - y3_min

# =============================================================================
# グラフ初期化
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# グラフ①: Joint State
line1_1, = axes[0].plot([], [], label=r'$\theta_{i=1}$', color='tab:blue')
line1_2, = axes[0].plot([], [], label=r'$\theta_{i=2}$', color='tab:orange')
line1_3, = axes[0].plot([], [], label=r'$\theta_{i=3}$', color='tab:green')
line1_4, = axes[0].plot([], [], label=r'$\theta_{i=4}$', color='tab:red')
axes[0].set_xlim(t_min, t_max)
axes[0].set_ylim(-40, 40)
axes[0].set_ylabel(r'$\theta_i$ [deg]', **label_font)
axes[0].legend(loc='upper right', **legend_font)
axes[0].grid(True)

# グラフ②: PID err_p
line2_1, = axes[1].plot([], [], label='x', color='tab:purple')
line2_2, = axes[1].plot([], [], label='y', color='tab:brown')
line2_3, = axes[1].plot([], [], label='z', color='tab:pink')
axes[1].set_xlim(t_min, t_max)
axes[1].set_ylim(-0.55, 0.55)
axes[1].set_ylabel('Position error [m]', **label_font)
axes[1].legend(loc='upper right', **legend_font)
axes[1].grid(True)

# グラフ③: Roll, Pitch, Yaw
line3_1, = axes[2].plot([], [], label='roll', color='tab:gray')
line3_2, = axes[2].plot([], [], label='pitch', color='tab:olive')
line3_3, = axes[2].plot([], [], label='yaw', color='tab:cyan')
axes[2].set_xlim(t_min, t_max)
axes[2].set_ylim(-15, 15)
axes[2].set_xlabel('Time [s]', **label_font)
axes[2].set_ylabel('Attitude error [deg]', **label_font)
axes[2].legend(loc='upper right', **legend_font)
axes[2].grid(True)

# 現在時刻を示す縦線（オプション）
vline1 = axes[0].axvline(x=t_min, color='red', linestyle='--', alpha=0.5)
vline2 = axes[1].axvline(x=t_min, color='red', linestyle='--', alpha=0.5)
vline3 = axes[2].axvline(x=t_min, color='red', linestyle='--', alpha=0.5)

# 時刻表示テキスト
time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# =============================================================================
# アニメーション関数
# =============================================================================
# フレーム数と動画の長さ
fps = 30
duration_sec = end_time.to_sec() - start_time.to_sec()  # 動画の長さ（秒）
total_frames = int(fps * duration_sec)

def init():
    """初期化関数"""
    line1_1.set_data([], [])
    line1_2.set_data([], [])
    line1_3.set_data([], [])
    line1_4.set_data([], [])
    line2_1.set_data([], [])
    line2_2.set_data([], [])
    line2_3.set_data([], [])
    line3_1.set_data([], [])
    line3_2.set_data([], [])
    line3_3.set_data([], [])
    vline1.set_xdata([t_min])
    vline2.set_xdata([t_min])
    vline3.set_xdata([t_min])
    time_text.set_text('')
    return (line1_1, line1_2, line1_3, line1_4,
            line2_1, line2_2, line2_3,
            line3_1, line3_2, line3_3,
            vline1, vline2, vline3, time_text)

def animate(frame):
    """アニメーション更新関数"""
    # 現在の時刻を計算
    progress = frame / total_frames
    current_time = t_min + progress * (t_max - t_min)
    
    # 現在時刻までのデータを取得（Joint States）
    mask_js = times_js <= current_time
    line1_1.set_data(times_js[mask_js], arm1[mask_js])
    line1_2.set_data(times_js[mask_js], arm2[mask_js])
    line1_3.set_data(times_js[mask_js], arm3[mask_js])
    line1_4.set_data(times_js[mask_js], arm4[mask_js])
    
    # PID
    mask_pid = times_pid <= current_time
    line2_1.set_data(times_pid[mask_pid], err_x[mask_pid])
    line2_2.set_data(times_pid[mask_pid], err_y[mask_pid])
    line2_3.set_data(times_pid[mask_pid], err_z[mask_pid])
    
    # Odometry
    mask_odom = times_odom <= current_time
    line3_1.set_data(times_odom[mask_odom], roll_vals[mask_odom])
    line3_2.set_data(times_odom[mask_odom], pitch_vals[mask_odom])
    line3_3.set_data(times_odom[mask_odom], yaw_vals[mask_odom])
    
    # 縦線を更新
    vline1.set_xdata([current_time, current_time])
    vline2.set_xdata([current_time, current_time])
    vline3.set_xdata([current_time, current_time])
    
    # 時刻表示を更新（開始時間を0として相対表示）
    relative_time = current_time - t_min
    time_text.set_text(f'Time: {relative_time:.2f} s')
    
    return (line1_1, line1_2, line1_3, line1_4,
            line2_1, line2_2, line2_3,
            line3_1, line3_2, line3_3,
            vline1, vline2, vline3, time_text)

# =============================================================================
# アニメーション生成・保存
# =============================================================================
print("アニメーション生成中...")
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=total_frames, interval=1000/fps,
                               blit=True)

# MP4で保存（ffmpegが必要）
print("MP4ファイルを保存中...")
writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='ROSbag Plotter'),
                                 bitrate=3000)
anim.save('rosbag_animationX.mp4', writer=writer)
print("保存完了: rosbag_animationX.mp4")

# GIFで保存する場合（pillowが必要）
# print("GIFファイルを保存中...")
# anim.save('rosbag_animation.gif', writer='pillow', fps=fps)
# print("保存完了: rosbag_animation.gif")

# プレビュー表示（オプション）
# plt.show()