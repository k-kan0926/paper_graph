#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, csv, pathlib
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    # SVGでフォント崩れを避けたいなら次をON
    # "svg.fonttype": "path",
})
import matplotlib.pyplot as plt


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

            datasets[name] = (np.array(x_vals), np.array(y_vals), np.array(z_rows))
        else:
            i += 1
    return datasets


def ensure_z_shape(x, y, z):
    # expected z.shape == (len(y), len(x))
    if z.shape == (len(y), len(x)):
        return z
    if z.T.shape == (len(y), len(x)):
        return z.T
    raise ValueError(f"z shape mismatch: z{z.shape}, expected ({len(y)},{len(x)}) or transpose")


def collect_dp_theta(csv_path, deg=True, pr=None):
    """
    Returns:
      dp_all: (p1 - p2) flattened
      th_all: theta flattened
    pr: (pmin, pmax) to filter points by feasible pressure window (optional)
    """
    data = load_dataset(pathlib.Path(csv_path))

    dp_list, th_list = [], []

    for key in ("step_minus", "step_plus"):
        x, y, z = data[key]
        z = ensure_z_shape(x, y, z)
        if deg:
            z = np.degrees(z)

        # CSV meaning:
        # step_minus: y = fix p1, x = sweep p2 -> p1=Y, p2=X
        # step_plus : y = fix p2, x = sweep p1 -> p1=X, p2=Y
        if key == "step_minus":
            P2, P1 = np.meshgrid(x, y)  # x=p2, y=p1
        else:
            P1, P2 = np.meshgrid(x, y)  # x=p1, y=p2

        dp = (P1 - P2).ravel()
        th = z.ravel()

        if pr is not None:
            pmin, pmax = pr
            p1 = P1.ravel()
            p2 = P2.ravel()
            m = (p1 >= pmin) & (p1 <= pmax) & (p2 >= pmin) & (p2 <= pmax)
            dp = dp[m]
            th = th[m]

        dp_list.append(dp)
        th_list.append(th)

    return np.concatenate(dp_list), np.concatenate(th_list)


def plot_dp_vs_theta(csv_path, out="dp_vs_theta.svg", pr=None):
    dp, th = collect_dp_theta(csv_path, deg=True, pr=pr)

    fig = plt.figure(figsize=(5.5, 4.0))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(dp, th, s=10, alpha=0.5)  # シンプル散布図

    ax.set_xlabel(r'$\Delta p = p_1 - p_2$ [MPa]')
    ax.set_ylabel(r'$\theta$ [deg]')

    title = "Δp vs θ"
    if pr is not None:
        title += f"  (p1,p2 ∈ [{pr[0]:.2f}, {pr[1]:.2f}] MPa)"
    ax.set_title(title)

    plt.savefig(out, bbox_inches="tight")  # outが.svgならSVG出力
    plt.close(fig)


if __name__ == "__main__":
    plot_dp_vs_theta(
        "csv/245mm.csv",
        out="dp_vs_theta_245mm.svg",
        pr=None  # 例: (0.15, 0.65) にすると圧力窓でフィルタ
    )
