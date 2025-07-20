#!/usr/bin/env python3
# plot_mpa_csv.py
# 使い方:  python plot_mpa_csv.py  your_data.csv

import sys, re, csv, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  ← 3-D 描画用
import matplotlib.font_manager as fm

# ──────────────────────────────────────────────────────────────
# 1.  CSV を読み取って   {dataset_name: (x, y, Z)}  を作る
# ──────────────────────────────────────────────────────────────
def load_datasets(csv_path: pathlib.Path):
    with csv_path.open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.reader(f)]

    # 空行を除去
    rows = [r for r in rows if not all(c.strip() == '' for c in r)]

    datasets = {}
    i = 0
    while i < len(rows):
        r = rows[i]
        if len(r) > 1 and r[1].strip() in ('step_minus', 'step_plus'):
            name = r[1].strip()
            x_vals = [float(c) for c in r[3:] if c.strip() != '']      # 0,0.05,…
            y_vals, z_rows = [], []

            i += 1
            while i < len(rows):
                r = rows[i]
                # 次のデータセットに到達したら break
                if len(r) > 1 and r[1].strip() in ('step_minus', 'step_plus'):
                    break
                # “fix p1 0.25” / “fix p2 0.3” の行だけ拾う
                if len(r) > 1 and r[1].startswith('fix'):
                    m = re.search(r'([0-9.]+)$', r[1])
                    if m:
                        y_vals.append(float(m.group(1)))
                        # データは 4 列目 (index 3) から
                        z_rows.append([float(c) for c in r[3:3 + len(x_vals)]])
                i += 1

            datasets[name] = (
                np.array(x_vals),
                np.array(y_vals),
                np.array(z_rows)        # shape = (len(y), len(x))
            )
        else:
            i += 1
    return datasets


# ──────────────────────────────────────────────────────────────
# 2.  グラフ描画
# ──────────────────────────────────────────────────────────────
def plot_two_surfaces(d1, d2, font_path=None):
    (x1, y1, z1_rad), (x2, y2, z2_rad) = d1, d2
    z2_rad = np.transpose(z2_rad) 
    z1, z2 = np.degrees(z1_rad), np.degrees(z2_rad)

    # 共通のカラーマップ範囲
    vmin, vmax = float(np.min([z1, z2])), float(np.max([z1, z2]))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Times New Roman（環境に合わせて変更）
    prop = fm.FontProperties(fname=font_path) if font_path else None

    # メッシュグリッド
    X1, Y1 = np.meshgrid(x1, y1)  # step_minus
    X2, Y2 = np.meshgrid(x2, y2)  # step_plus

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    # (a) step_minus ----------------------------------------------------------
    ax1 = fig.add_subplot(gs[0], projection='3d')
    surf1 = ax1.plot_surface(X1, Y1, z1, cmap='viridis', norm=norm)
    ax1.set_title('(a) step_minus', pad=-350, fontproperties=prop)
    ax1.set_xlabel('R-MPA Pressure [MPa]', fontproperties=prop)
    ax1.set_ylabel('Fixed L-MPA Pressure [MPa]', fontproperties=prop)
    ax1.set_zlabel('α [deg]', fontproperties=prop)
    ax1.set_zticks(np.arange(np.floor(vmin), np.ceil(vmax) + 1, 10))
    ax1.view_init(elev=30, azim=-120)

    # (b) step_plus -----------------------------------------------------------
    ax2 = fig.add_subplot(gs[1], projection='3d')
    surf2 = ax2.plot_surface(X2, Y2, z2, cmap='viridis', norm=norm)
    ax2.set_title('(b) step_plus', pad=-350, fontproperties=prop)
    ax2.set_xlabel('Fixed R-MPA Pressure [MPa]', fontproperties=prop)
    ax2.set_ylabel('L-MPA Pressure [MPa]', fontproperties=prop)
    ax2.set_zlabel('α [deg]', fontproperties=prop)
    ax2.set_zticks(np.arange(np.floor(vmin), np.ceil(vmax) + 1, 10))
    ax2.view_init(elev=30, azim=-120)

    # カラーバー（共通）
    cax = fig.add_subplot(gs[2])
    cbar = fig.colorbar(surf2, cax=cax)
    cbar.set_label('α [deg]', fontproperties=prop)

    plt.subplots_adjust(left=0.08, right=0.92, top=0.92,
                        bottom=0.08, wspace=0.1)
    plt.show()


# ──────────────────────────────────────────────────────────────
# 3.  エントリポイント
# ──────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        print('Usage:  python plot_mpa_csv.py  your_data.csv')
        sys.exit(1)

    csv_path = pathlib.Path(sys.argv[1]).expanduser()
    if not csv_path.is_file():
        print(f'File not found: {csv_path}')
        sys.exit(1)

    data = load_datasets(csv_path)
    try:
        step_minus = data['step_minus']
        step_plus  = data['step_plus']
    except KeyError as e:
        print(f'CSV に {e.args[0]} ブロックが見つかりませんでした')
        sys.exit(1)

    # 必要ならフォントパスを変更
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    plot_two_surfaces(step_minus, step_plus, font_path=font_path)


if __name__ == '__main__':
    main()
