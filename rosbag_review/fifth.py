import rosbag
import matplotlib.pyplot as plt
import bisect

# === ここにrosbagのファイルパスを入力 ===
bag_file = "2025-02-08-17-04-59.bag"

# データを格納する辞書（タイムスタンプをキーにする）
arm2_positions = {}
arm3_positions = {}
mpa_x_values = {}
mpa_y_values = {}

# rosbag からデータを読み取る
print("Reading rosbag data...")

with rosbag.Bag(bag_file, "r") as bag:
    for topic, msg, t in bag.read_messages():
        timestamp = t.to_sec()  # タイムスタンプを秒単位で取得

        if topic == "/quadrotor1/joint_states":
            arm2_positions[timestamp] = msg.position[0]  # arm2_joint の position
            arm3_positions[timestamp] = msg.position[1]  # arm3_joint の position

        elif topic == "/mpa_cmd":  # ✅ 修正: 1回の `elif` で `x` と `y` を取得
            mpa_x_values[timestamp] = msg.x
            mpa_y_values[timestamp] = msg.y  # ✅ `mpa_y_values` にデータを入れる
#            print(f"[DEBUG] mpa_cmd received at {timestamp}: x={msg.x}, y={msg.y}")

# ** データ数を確認（デバッグ用）**
#print(f"\n[INFO] Data Count:")
#print(f"  arm2_positions: {len(arm2_positions)}")
#print(f"  arm3_positions: {len(arm3_positions)}")
#print(f"  mpa_x_values: {len(mpa_x_values)}")
#print(f"  mpa_y_values: {len(mpa_y_values)}")  # ✅ `mpa_y_values` が 0 でないか確認

#if len(mpa_y_values) == 0:
#    print("[ERROR] No data found for mpa_cmd (y)!")

# 近似タイムスタンプでデータをマッチング
def find_closest(timestamp_list, target_time):
    """target_time に最も近い timestamp_list の値を返す"""
    pos = bisect.bisect_left(timestamp_list, target_time)
    if pos == 0:
        return timestamp_list[0]
    if pos == len(timestamp_list):
        return timestamp_list[-1]
    before = timestamp_list[pos - 1]
    after = timestamp_list[pos]
    return before if abs(before - target_time) < abs(after - target_time) else after

mpa_x_timestamps = sorted(mpa_x_values.keys())
mpa_y_timestamps = sorted(mpa_y_values.keys())

matched_arm2 = []
matched_mpa_x = []
matched_arm3 = []
matched_mpa_y = []

for timestamp in arm2_positions.keys():
    if mpa_x_timestamps:
        closest_x_time = find_closest(mpa_x_timestamps, timestamp)
        matched_arm2.append(arm2_positions[timestamp])
        matched_mpa_x.append(mpa_x_values[closest_x_time])

for timestamp in arm3_positions.keys():
    if mpa_y_timestamps:
        closest_y_time = find_closest(mpa_y_timestamps, timestamp)
        matched_arm3.append(arm3_positions[timestamp])
        matched_mpa_y.append(mpa_y_values[closest_y_time])

#print(f"\n[INFO] Matched Data Count:")
#print(f"  matched_arm2: {len(matched_arm2)}")
#print(f"  matched_mpa_x: {len(matched_mpa_x)}")
#print(f"  matched_arm3: {len(matched_arm3)}")
#print(f"  matched_mpa_y: {len(matched_mpa_y)}")  # ✅ `matched_mpa_y` が 0 でないか確認

#if len(matched_mpa_y) == 0:
#    print("[ERROR] No valid matched data for plotting arm3_joint!")

# グラフ作成
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

axes[0].scatter(matched_arm2, matched_mpa_x, color="blue", alpha=0.6)
axes[0].set_xlim(-1.0, 0.8)
axes[0].set_ylim(0, 2500)
axes[0].set_xlabel("arm2_joint position")
axes[0].set_ylabel("mpa_cmd x")
axes[0].grid(True)

axes[1].scatter(matched_arm3, matched_mpa_y, color="red", alpha=0.6)
axes[1].set_xlim(-1.0, 0.8)
axes[1].set_ylim(0, 2500)
axes[1].set_xlabel("arm3_joint position")
axes[1].set_ylabel("mpa_cmd y")
axes[1].grid(True)

plt.tight_layout()
plt.show()
