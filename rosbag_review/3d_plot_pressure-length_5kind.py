#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, re, csv, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

def load_dataset(csv_path: pathlib.Path):
    with csv_path.open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.reader(f) if not all(c.strip() == '' for c in r)]

    datasets = {}
    i = 0
    while i < len(rows):
        r = rows[i]
        if len(r) > 1 and r[1].strip() in ('step_minus', 'step_plus'):
            name = r[1].strip()
            x_vals = [float(c) for c in r[3:] if c.strip() != '']
            y_vals, z_rows = [], []

            i += 1
            while i < len(rows):
                r = rows[i]
                if len(r) > 1 and r[1].strip() in ('step_minus', 'step_plus'):
                    break
                if len(r) > 1 and r[1].startswith('fix'):
                    m = re.search(r'([0-9.]+)$', r[1])
                    if m:
                        y_vals.append(float(m.group(1)))
                        z_rows.append([float(c) for c in r[3:3 + len(x_vals)]])
                i += 1

            datasets[name] = (
                np.array(x_vals),
                np.array(y_vals),
                np.array(z_rows)
            )
        else:
            i += 1
    return datasets

def plot_5_pairs(csv_paths, labels, output_pdf_path, font_path=None):
    font_prop = fm.FontProperties(fname=font_path) if font_path else None
    fig = plt.figure(figsize=(14, 22))
    gs = gridspec.GridSpec(nrows=5, ncols=2, wspace=0.3, hspace=0.6)

    global_vmin, global_vmax = float('inf'), float('-inf')
    data_list = []

    for csv_path in csv_paths:
        data = load_dataset(pathlib.Path(csv_path))
        x1, y1, z1 = data['step_minus']
        x2, y2, z2 = data['step_plus']
        z1_deg = np.degrees(z1)
        z2_deg = np.degrees(np.transpose(z2))
        global_vmin = min(global_vmin, z1_deg.min(), z2_deg.min())
        global_vmax = max(global_vmax, z1_deg.max(), z2_deg.max())
        data_list.append(((x1, y1, z1_deg), (x2, y2, z2_deg)))

    norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
    labeltitel_fontsize = 14
    label_fontsize = 8
    title_fontsize = 16

    for idx, ((x1, y1, z1), (x2, y2, z2)) in enumerate(data_list):
        X1, Y1 = np.meshgrid(x1, y1)
        X2, Y2 = np.meshgrid(x2, y2)

        # (a) step_minus
        ax1 = fig.add_subplot(gs[idx, 0], projection='3d')
        ax1.plot_surface(X1, Y1, z1, cmap='viridis', norm=norm)
        ax1.set_xlabel('R-MPA Pressure [MPa]', fontproperties=font_prop, fontsize=labeltitel_fontsize)
        ax1.set_ylabel('Fixed L-MPA Pressure [MPa]', fontproperties=font_prop, fontsize=labeltitel_fontsize)
        ax1.set_zlabel(r'$\theta$ [deg]', fontproperties=font_prop, fontsize=labeltitel_fontsize)
        ax1.tick_params(axis='x', labelsize=label_fontsize)
        ax1.tick_params(axis='y', labelsize=label_fontsize)
        ax1.tick_params(axis='z', labelsize=label_fontsize)
        ax1.view_init(elev=40, azim=-140)
        ax1.text2D(0.5, -0.15, f"({chr(97+idx*2)}) {labels[idx]} step_minus", transform=ax1.transAxes,
                   ha='center', va='top', fontproperties=font_prop, fontsize=title_fontsize)

        # (b) step_plus
        ax2 = fig.add_subplot(gs[idx, 1], projection='3d')
        ax2.plot_surface(X2, Y2, z2, cmap='viridis', norm=norm)
        ax2.set_xlabel('Fixed R-MPA Pressure [MPa]', fontproperties=font_prop, fontsize=labeltitel_fontsize)
        ax2.set_ylabel('L-MPA Pressure [MPa]', fontproperties=font_prop, fontsize=labeltitel_fontsize)
        ax2.set_zlabel(r'$\theta$ [deg]', fontproperties=font_prop, fontsize=labeltitel_fontsize)
        ax2.tick_params(axis='x', labelsize=label_fontsize)
        ax2.tick_params(axis='y', labelsize=label_fontsize)
        ax2.tick_params(axis='z', labelsize=label_fontsize)
        ax2.view_init(elev=40, azim=-140)
        ax2.text2D(0.5, -0.15, f"({chr(98+idx*2)}) {labels[idx]} step_plus", transform=ax2.transAxes,
                   ha='center', va='top', fontproperties=font_prop, fontsize=title_fontsize)

    # カラーバー（右端）
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('θ [deg]', fontproperties=font_prop, fontsize=title_fontsize)

    plt.savefig(output_pdf_path)
    plt.close()

# 使用例（実行時）
if __name__ == '__main__':
    csv_files = [
        "csv/240mm.csv",
        "csv/245mm.csv",
        "csv/250mm.csv",
        "csv/255mm.csv",
        "csv/260mm.csv"
    ]
    labels = ["240mm", "245mm", "250mm", "255mm", "260mm"]
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"  # なければ None に
    output_path = "step_surfaces_5pairs_improved.pdf"

    plot_5_pairs(csv_files, labels, output_path, font_path=font_path)
