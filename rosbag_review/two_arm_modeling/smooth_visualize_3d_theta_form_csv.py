#!/usr/bin/env python3
#python smooth_visualize_3d_theta_form_csv.py   --csv out/diff_run1_data.csv   --out-prefix out/vis3d   --downsample 12000   --nbins 80   --smooth cubic#
#作成したcsvからなめらかな3D散布図とサーフェスを描く
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def make_surface_mean_bin(x, y, z, nbins=40, agg=np.nanmean):
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    xbins = np.linspace(np.nanmin(x), np.nanmax(x), nbins+1)
    ybins = np.linspace(np.nanmin(y), np.nanmax(y), nbins+1)
    Xi = 0.5*(xbins[:-1]+xbins[1:]); Yi = 0.5*(ybins[:-1]+ybins[1:])
    Zg = np.full((nbins, nbins), np.nan)
    xi = np.digitize(x, xbins)-1; yi = np.digitize(y, ybins)-1
    mask=(xi>=0)&(xi<nbins)&(yi>=0)&(yi<nbins)&np.isfinite(z)
    xi,yi,z = xi[mask], yi[mask], z[mask]
    from collections import defaultdict
    buckets=defaultdict(list)
    for a,b,val in zip(xi,yi,z): buckets[(a,b)].append(val)
    for (a,b), vals in buckets.items(): Zg[b,a] = agg(vals)
    Xg,Yg = np.meshgrid(Xi, Yi)
    return Xg,Yg,Zg

def maybe_griddata(x, y, z, nbins=80, method="cubic"):
    """SciPyがあれば griddata で滑らか補間、無ければ None を返す。"""
    try:
        from scipy.interpolate import griddata
    except Exception:
        return None, None, None
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    X = np.linspace(np.nanmin(x), np.nanmax(x), nbins)
    Y = np.linspace(np.nanmin(y), np.nanmax(y), nbins)
    Xg, Yg = np.meshgrid(X, Y)
    Zg = griddata((x, y), z, (Xg, Yg), method=method)
    return Xg, Yg, Zg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", default="out/vis3d")
    ap.add_argument("--downsample", type=int, default=0)
    ap.add_argument("--nbins", type=int, default=60)
    ap.add_argument("--smooth", choices=["cubic","linear","off"], default="cubic",
                    help="SciPy griddata の補間法。SciPy無しや off なら平均ビニング面。")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    df = pd.read_csv(args.csv, comment="#")
    cols = df.columns
    # 既定CSV列（analyze_bag_* と同じ）
    need = ["p_sum[MPa]","p_diff[MPa]","p1[MPa]","p2[MPa]","theta[deg]"]
    for k in need:
        if k not in cols: raise RuntimeError(f"CSV列がありません: {k}")

    if args.downsample and args.downsample>0 and args.downsample < len(df):
        df = df.sample(args.downsample, random_state=42).sort_index()

    ps, pdiff = df["p_sum[MPa]"].to_numpy(), df["p_diff[MPa]"].to_numpy()
    p1, p2 = df["p1[MPa]"].to_numpy(), df["p2[MPa]"].to_numpy()
    th_deg  = df["theta[deg]"].to_numpy()

    def plot_surf(x, y, z, xlabel, ylabel, title, tag, cmap="viridis"):
        # try smooth (SciPy) else mean-bin
        Xg=Yg=Zg=None
        if args.smooth!="off":
            Xg, Yg, Zg = maybe_griddata(x, y, z, nbins=args.nbins, method=args.smooth)
        if Zg is None or np.all(np.isnan(Zg)):
            Xg, Yg, Zg = make_surface_mean_bin(x, y, z, nbins=max(30, args.nbins//2))

        import numpy.ma as ma
        Zm = ma.masked_invalid(Zg)

        # 3D surface
        from matplotlib import cm
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(Xg, Yg, Zm, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel("theta [deg]")
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.7, label="theta [deg]")
        ax.view_init(elev=28, azim=-55)
        plt.tight_layout()
        path = f"{args.out_prefix}_surface_{tag}.png"
        plt.savefig(path, dpi=150); print("[OK] wrote", path)

        # heatmap
        fig = plt.figure(figsize=(7,6))
        plt.pcolormesh(Xg, Yg, Zg, shading="auto", cmap=cmap)
        plt.colorbar(label="theta [deg]")
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.title(title + " (heatmap)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        path = f"{args.out_prefix}_heatmap_{tag}.png"
        plt.savefig(path, dpi=150); print("[OK] wrote", path)

    # (p_diff, p_sum) -> theta
    plot_surf(pdiff, ps, th_deg,
              "p_diff [MPa]", "p_sum [MPa]",
              "theta(p_diff, p_sum)", "pdiff_psum", cmap="viridis")

    # (p1, p2) -> theta
    plot_surf(p1, p2, th_deg,
              "p1 [MPa]", "p2 [MPa]",
              "theta(p1, p2)", "p1_p2", cmap="plasma")

    print("[DONE]")

if __name__ == "__main__":
    main()
