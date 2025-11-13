import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,       # ← 重要（True だとType3の温床）
    "mathtext.fontset": "stix", # か "cm"
    "pdf.use14corefonts": False,
    "font.family": "Times New Roman",  # 日本語混在なら 'IPAexGothic'
})

# フォント・スタイルの統一
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Times New Roman'  # 日本語なら 'IPAPGothic' など
label_font = {'fontsize': 16, 'fontweight': 'bold'}
legend_font = {'fontsize': 12}

bag = rosbag.Bag('I_transformation.bag', 'r')
bag_start = bag.get_start_time()
start_time = rospy.Time(bag_start + 130.0)
end_time   = rospy.Time(bag_start + 170.0)

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

# グラフ描画
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# グラフ①: Joint State
axes[0].plot(times_js, arm1, label=r'$\theta_{i=1}$', color='tab:blue')
axes[0].plot(times_js, arm2, label=r'$\theta_{i=2}$', color='tab:orange')
axes[0].plot(times_js, arm3, label=r'$\theta_{i=3}$', color='tab:green')
axes[0].plot(times_js, arm4, label=r'$\theta_{i=4}$', color='tab:red')
axes[0].set_ylabel(r'$\theta_i$ [rad]', **label_font)
axes[0].legend(loc='upper right', **legend_font)
axes[0].grid(True)

# グラフ②: PID err_p
axes[1].plot(times_pid, err_x, label='x', color='tab:purple')
axes[1].plot(times_pid, err_y, label='y', color='tab:brown')
axes[1].plot(times_pid, err_z, label='z', color='tab:pink')
axes[1].set_ylabel('Position error [m]', **label_font)
axes[1].legend(loc='upper right', **legend_font)
axes[1].grid(True)

# グラフ③: Roll, Pitch, Yaw
axes[2].plot(times_odom, roll_vals, label='roll', color='tab:gray')
axes[2].plot(times_odom, pitch_vals, label='pitch', color='tab:olive')
axes[2].plot(times_odom, yaw_vals, label='yaw', color='tab:cyan')
axes[2].set_xlabel('Time [s]', **label_font)
axes[2].set_ylabel('Attitude error [rad]', **label_font)
axes[2].legend(loc='upper right', **legend_font)
axes[2].grid(True)

plt.tight_layout()
plt.show()
