#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, csv, pathlib
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt


# =========================
# Data loader (same logic)
# =========================
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
                np.array(x_vals, dtype=float),
                np.array(y_vals, dtype=float),
                np.array(z_rows, dtype=float)
            )
        else:
            i += 1
    return datasets


def ensure_z_shape(x, y, z):
    """
    pcolormesh expects z.shape == (len(y), len(x)).
    If transposed, fix automatically.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got {z.ndim}D")
    if z.shape == (len(y), len(x)):
        return z
    if z.T.shape == (len(y), len(x)):
        return z.T
    raise ValueError(f"z shape mismatch: z{z.shape}, expected ({len(y)},{len(x)}) (or transpose)")


def mask_by_pressure(x, y, z, xmin=0.15, xmax=0.65, ymin=0.15, ymax=0.65):
    xi = (x >= xmin) & (x <= xmax)
    yi = (y >= ymin) & (y <= ymax)
    return x[xi], y[yi], z[np.ix_(yi, xi)]


def summarize_range(csv_paths, lengths_mm, pr=(0.15, 0.65)):
    theta_min, theta_max = [], []
    for p in csv_paths:
        data = load_dataset(pathlib.Path(p))

        x1, y1, z1 = data['step_minus']
        x2, y2, z2 = data['step_plus']

        z1 = ensure_z_shape(x1, y1, z1)
        z2 = ensure_z_shape(x2, y2, z2)

        z1 = np.degrees(z1)
        z2 = np.degrees(z2)

        x1m, y1m, z1m = mask_by_pressure(x1, y1, z1, pr[0], pr[1], pr[0], pr[1])
        x2m, y2m, z2m = mask_by_pressure(x2, y2, z2, pr[0], pr[1], pr[0], pr[1])

        z_all = np.concatenate([z1m.ravel(), z2m.ravel()])
        theta_min.append(np.nanmin(z_all))
        theta_max.append(np.nanmax(z_all))

    return np.array(lengths_mm, dtype=float), np.array(theta_min, dtype=float), np.array(theta_max, dtype=float)


def plot_slide8(csv_paths, lengths_mm, chosen_idx=1, out="slide8_fig.svg",
                pr=(0.15, 0.65), heatmap_use_full_range=True, heatmap_mode="both"):
    """
    - heatmap_mode: "both" (default), "minus" (step_minus only), "plus" (step_plus only)
    - Left/Mid: heatmap(s) depending on mode
    - Right: angle range vs length (within pr pressure window)
    - Bottom row: shared colorbar
    """

    # Validate heatmap_mode
    if heatmap_mode not in ("both", "minus", "plus"):
        raise ValueError(f"heatmap_mode must be 'both', 'minus', or 'plus', got '{heatmap_mode}'")

    # --- summarize over lengths ---
    L, tmin, tmax = summarize_range(csv_paths, lengths_mm, pr=pr)

    # --- load chosen length for heatmaps ---
    data = load_dataset(pathlib.Path(csv_paths[chosen_idx]))
    x1, y1, z1 = data['step_minus']
    x2, y2, z2 = data['step_plus']

    z1 = np.degrees(ensure_z_shape(x1, y1, z1))
    z2 = np.degrees(ensure_z_shape(x2, y2, z2))

    # optional: restrict heatmap region to pr
    if not heatmap_use_full_range:
        x1, y1, z1 = mask_by_pressure(x1, y1, z1, pr[0], pr[1], pr[0], pr[1])
        x2, y2, z2 = mask_by_pressure(x2, y2, z2, pr[0], pr[1], pr[0], pr[1])

    # --- consistent colormap range ---
    vmin = min(np.nanmin(z1), np.nanmin(z2))
    vmax = max(np.nanmax(z1), np.nanmax(z2))

    # --- layout depends on heatmap_mode ---
    if heatmap_mode == "both":
        # Original layout: 3 columns (2 heatmaps + range plot)
        fig = plt.figure(figsize=(12, 4.6))
        gs = fig.add_gridspec(
            nrows=2, ncols=3,
            height_ratios=[1.0, 0.08],
            width_ratios=[1.0, 1.0, 1.2],
            wspace=0.35, hspace=0.28
        )

        # heatmap 1 (step_minus)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.pcolormesh(x1, y1, z1, shading='auto', vmin=vmin, vmax=vmax)
        ax1.set_xlabel("p2 (sweep) [MPa]")
        ax1.set_ylabel("p1 (fixed) [MPa]")
        ax1.set_title("step_minus")

        # heatmap 2 (step_plus)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.pcolormesh(x2, y2, z2, shading='auto', vmin=vmin, vmax=vmax)
        ax2.set_xlabel("p1 (sweep) [MPa]")
        ax2.set_ylabel("p2 (fixed) [MPa]")
        ax2.set_title("step_plus")

        # shared colorbar
        cax = fig.add_subplot(gs[1, 0:2])
        cb = fig.colorbar(im2, cax=cax, orientation='horizontal')
        cb.set_label(r'$\theta$ [deg]')

        # range vs length
        ax3 = fig.add_subplot(gs[:, 2])

    else:
        # Single heatmap layout: 2 columns (1 heatmap + range plot)
        fig = plt.figure(figsize=(9, 4.6))
        gs = fig.add_gridspec(
            nrows=2, ncols=2,
            height_ratios=[1.0, 0.08],
            width_ratios=[1.0, 1.2],
            wspace=0.35, hspace=0.28
        )

        ax_heat = fig.add_subplot(gs[0, 0])

        if heatmap_mode == "minus":
            im = ax_heat.pcolormesh(x1, y1, z1, shading='auto', vmin=vmin, vmax=vmax)
            ax_heat.set_xlabel("p2 ($\mathbf{sweep}$) [MPa]")
            ax_heat.set_ylabel("p1 ($\mathbf{fixed}$) [MPa]")
        else:  # "plus"
            im = ax_heat.pcolormesh(x2, y2, z2, shading='auto', vmin=vmin, vmax=vmax)
            ax_heat.set_xlabel("p1 ($\mathbf{sweep}$) [MPa]")
            ax_heat.set_ylabel("p2 ($\mathbf{fixed}$) [MPa]")
        # colorbar
        cax = fig.add_subplot(gs[1, 0])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_label(r'$\theta$ [deg]')

        # range vs length
        ax3 = fig.add_subplot(gs[:, 1])

    # --- range vs length plot (common) ---
    ax3.plot(L, tmax, marker='o', label=r'$\theta_{\max}$')
    ax3.plot(L, tmin, marker='o', label=r'$\theta_{\min}$')
    ax3.fill_between(L, tmin, tmax, alpha=0.2, label='usable range')

    # highlight chosen length
    ax3.scatter([L[chosen_idx]], [tmax[chosen_idx]], s=70, zorder=3, color='red')
    ax3.scatter([L[chosen_idx]], [tmin[chosen_idx]], s=70, zorder=3, color='red')
    ax3.axvline(L[chosen_idx], linestyle='--', linewidth=1, color='gray')

    ax3.set_xlabel("MPA length [mm]")
    ax3.set_ylabel(r'$\theta$ [deg]')
    ax3.legend(frameon=False, loc='best')

    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == '__main__':
    csv_files = [
        "csv/240mm.csv",
        "csv/245mm.csv",
        "csv/250mm.csv",
        "csv/255mm.csv",
        "csv/260mm.csv"
    ]
    lengths = [240, 245, 250, 255, 260]

    # --- 使用例 ---
    
    # 両方のヒートマップを表示（従来通り）
    plot_slide8(
        csv_files, lengths,
        chosen_idx=1,
        out="slide8_both.pdf",
        pr=(0.15, 0.65),
        heatmap_use_full_range=True,
        heatmap_mode="both"
    )

    # step_minus のみ表示
    plot_slide8(
        csv_files, lengths,
        chosen_idx=1,
        out="slide8_minus.svg",
        pr=(0.15, 0.65),
        heatmap_use_full_range=True,
        heatmap_mode="minus"
    )

    # step_plus のみ表示
    plot_slide8(
        csv_files, lengths,
        chosen_idx=1,
        out="slide8_plus.pdf",
        pr=(0.15, 0.65),
        heatmap_use_full_range=True,
        heatmap_mode="plus"
    )