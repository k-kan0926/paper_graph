import pandas as pd
import matplotlib.pyplot as plt
import io

# データのヘッダーと提供された1行のデータ
# 実際には、この文字列をファイルから読み込むことを想定しています。
csv_data = """t[s],p_sum[MPa],p_diff[MPa],p1[MPa],p2[MPa],p1_cmd[MPa],p2_cmd[MPa],theta[rad],theta[deg],z[m],dz[m]
2.906850576400756836e+00,7.072727084159851074e-01,-2.242424190044403076e-01,2.415151447057723999e-01,4.657575637102127075e-01,2.415151515151515715e-01,4.657575757575757591e-01,-1.202071449047805926e-02,-6.887362070361445765e-01,7.902783751487731934e-01,-5.742967128753662109e-03
# ここに大量のデータ行が続くと仮定
"""

# CSVデータを読み込む（ファイルからの読み込みに置き換える場合は次の行のコメントを解除）
df = pd.read_csv('/home/keiichiro/documents/paper_graph/rosbag_review/two_arm_modeling_with_height/graph_from_csv/diff_run1_h_data.csv')
# df = pd.read_csv(io.StringIO(csv_data))

# 't[s]'を横軸、'theta[deg]'を縦軸とする
x_data = df['t[s]']
y_data = df['theta[deg]']

# グラフの作成
plt.figure(figsize=(12, 6)) # グラフのサイズを設定
plt.plot(x_data.values, y_data.values, label='Theta (deg)') # 折れ線グラフを作成
plt.xlabel('Time $t$ [s]') # 横軸ラベル
plt.ylabel('Angle $\\theta$ [deg]') # 縦軸ラベル
plt.title('Time Series of Angle $\\theta$') # グラフタイトル
plt.grid(True) # グリッド線を表示

# 縦軸の範囲を設定
plt.ylim(-45, 45)

plt.legend() # 凡例を表示
plt.show() # グラフを表示

# グラフのファイル保存
# plt.savefig('theta_vs_time.png')