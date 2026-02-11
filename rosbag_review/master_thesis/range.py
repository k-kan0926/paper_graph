#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import glob
import pathlib
import argparse
import numpy as np
import matplotlib as mpl

# =========================
# Font & paper-friendly settings (match your 2nd script)
# =========================
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "pdf.use14corefonts": False,

    # Times New Roman (fallbacks included)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],

    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    # optional: minus sign rendering
    "axes.unicode_minus": False,
})

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =========================
# CSV loader (same logic)
# =========================
def load_dataset(csv_path: pathlib.Path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if not all(c.strip() == "" for c in r)]

    datasets = {}
    i = 0
    while i < len(rows):
        r = rows[i]
        if len(r) > 1 and r[1].strip() in ("step_minus", "step_plus"):
            name = r[1].strip()
            x_vals = [float(c) for c in r[3:] if c.strip() != ""]
            y_vals, z_rows = [], []

            i += 1
            while i < len(rows):
                r = rows[i]
                if len(r) > 1 and r[1].strip() in ("step_minus", "step_plus"):
                    break
                if len(r) > 1 and r[1].startswith("fix"):
                    m = re.search(r"([0-9.]+)$", r[1])
                    if m:
                        y_vals.append(float(m.group(1)))
                        z_rows.append([float(c) for c in r[3:3 + len(x_vals)]])
                i += 1

            datasets[name] = (
                np.array(x_vals, dtype=float),
                np.array(y_vals, dtype=float),
                np.array(z_rows, dtype=float),
            )
        else:
            i += 1
    return datasets


def ensure_z_shape(x, y, z):
    """pcolormesh期待形状: z.shape == (len(y), len(x))。転置なら自動修正。"""
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got {z.ndim}D")
    if z.shape == (len(y), len(x)):
        return z
    if z.T.shape == (len(y), len(x)):
        return z.T
    raise ValueError(f"z shape mismatch: z{z.shape}, expected ({len(y)},{len(x)})")


def mask_by_pressure(x, y, z, xmin, xmax, ymin, ymax):
    xi = (x >= xmin) & (x <= xmax)
    yi = (y >= ymin) & (y <= ymax)
    return x[xi], y[yi], z[np.ix_(yi, xi)]


def collect_csvs(glob_pattern: str, length_regex=r"(\d+(?:\.\d+)?)mm"):
    """ファイル名から '245mm' のような長さを抽出してソート。"""
    paths = [pathlib.Path(p) for p in glob.glob(glob_pattern)]
    items = []
    for p in paths:
        m = re.search(length_regex, p.name)
        if m:
            L = float(m.group(1))
            items.append((L, p))
    items.sort(key=lambda t: t[0])
    if not items:
        raise RuntimeError(f"No csv matched pattern '{glob_pattern}' with regex '{length_regex}'")
    lengths = [t[0] for t in items]
    csvs = [str(t[1]) for t in items]
    return csvs, lengths


def summarize_range(csv_paths, lengths_mm, pr=(0.0, 0.7), robust_percentile=None):
    """
    pr: pressure window (min,max) for both axes
    robust_percentile:
      - None: use min/max
      - (p_low, p_high): use percentiles
    """
    theta_min, theta_max = [], []
    for p in csv_paths:
        data = load_dataset(pathlib.Path(p))
        if "step_minus" not in data or "step_plus" not in data:
            raise KeyError(f"Missing step_minus/step_plus in {p}")

        x1, y1, z1 = data["step_minus"]
        x2, y2, z2 = data["step_plus"]

        z1 = np.degrees(ensure_z_shape(x1, y1, z1))
        z2 = np.degrees(ensure_z_shape(x2, y2, z2))

        # ---- pressure window (now default 0.0-0.7) ----
        x1m, y1m, z1m = mask_by_pressure(x1, y1, z1, pr[0], pr[1], pr[0], pr[1])
        x2m, y2m, z2m = mask_by_pressure(x2, y2, z2, pr[0], pr[1], pr[0], pr[1])

        z_all = np.concatenate([z1m.ravel(), z2m.ravel()])
        z_all = z_all[np.isfinite(z_all)]
        if z_all.size == 0:
            theta_min.append(np.nan)
            theta_max.append(np.nan)
            continue

        if robust_percentile is None:
            theta_min.append(float(np.min(z_all)))
            theta_max.append(float(np.max(z_all)))
        else:
            pl, ph = robust_percentile
            theta_min.append(float(np.percentile(z_all, pl)))
            theta_max.append(float(np.percentile(z_all, ph)))

    L = np.array(lengths_mm, dtype=float)
    tmin = np.array(theta_min, dtype=float)
    tmax = np.array(theta_max, dtype=float)
    dr = tmax - tmin
    return L, tmin, tmax, dr


def plot_range_only(L, tmin, tmax, dr, out, pr, highlight_length=None, show_value_labels=True):
    valid = np.isfinite(dr)
    if not np.any(valid):
        raise RuntimeError("All ranges are NaN. Check pressure window / CSV contents.")
    best_idx = int(np.nanargmax(dr))

    if highlight_length is not None:
        hi = int(np.argmin(np.abs(L - float(highlight_length))))
    else:
        hi = best_idx

    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    ax.plot(L, tmax, marker="o", label=r"$\theta_{\max}$")
    ax.plot(L, tmin, marker="o", label=r"$\theta_{\min}$")
    ax.fill_between(L, tmin, tmax, alpha=0.2, label=r"usable range ($\Delta\theta$)")

    # emphasize best (by Δθ)
    ax.scatter([L[best_idx]], [tmax[best_idx]], s=90, marker="*", zorder=5, color="red")
    ax.scatter([L[best_idx]], [tmin[best_idx]], s=90, marker="*", zorder=5, color="red")
    ax.axvline(L[best_idx], linestyle="--", linewidth=1)

    if highlight_length is not None and hi != best_idx:
        ax.axvline(L[hi], linestyle=":", linewidth=1)

    # ---- FIX #1: x-axis ticks (avoid 2.5 step auto ticks) ----
    # Use measured lengths as major ticks
    ax.set_xticks(L)
    # show as integers if they are close to integers
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    # small margin
    ax.set_xlim(float(np.min(L)) - 1.0, float(np.max(L)) + 1.0)

    # value labels (Δθ)
    if show_value_labels:
        for i in range(len(L)):
            if np.isfinite(dr[i]):
                ax.text(L[i], tmax[i] + 1.0, f"{dr[i]:.0f}°",
                        ha="center", va="bottom", fontsize=9)

    # annotation (pressure window now typically 0.00-0.70)

    ax.set_xlabel("MPA length [mm]")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="csv/*mm.csv", help="CSV glob pattern (default: csv/*mm.csv)")

    # ---- FIX #3: default pressure window 0.0-0.7 ----
    ap.add_argument("--pmin", type=float, default=0.0)
    ap.add_argument("--pmax", type=float, default=0.7)

    ap.add_argument("--out", default="usable_range_vs_length.png")
    ap.add_argument("--highlight", type=float, default=None, help="force highlight length (e.g., 245)")
    ap.add_argument("--robust", default=None,
                    help="use robust percentiles '1,99' instead of min/max (e.g., --robust 1,99)")
    ap.add_argument("--no_labels", action="store_true", help="disable Δθ value labels")
    args = ap.parse_args()

    pr = (args.pmin, args.pmax)
    csvs, lengths = collect_csvs(args.glob)

    robust = None
    if args.robust is not None:
        pl, ph = args.robust.split(",")
        robust = (float(pl), float(ph))

    L, tmin, tmax, dr = summarize_range(csvs, lengths, pr=pr, robust_percentile=robust)
    plot_range_only(
        L, tmin, tmax, dr,
        out=args.out,
        pr=pr,
        highlight_length=args.highlight,
        show_value_labels=(not args.no_labels),
    )


if __name__ == "__main__":
    main()
