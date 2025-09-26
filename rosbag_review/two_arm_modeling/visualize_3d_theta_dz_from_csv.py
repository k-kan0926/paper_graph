#!/usr/bin/env python3
#python visualize_3d_theta_dz_from_csv.py   --csv out/diff_run1_h_data.csv   --out-prefix out/vis3d_h   --downsample 8000   --nbins 40   --view 25,-60
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def make_surface(x, y, z, nbins=40, agg=np.nanmean):
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    xbins = np.linspace(np.nanmin(x), np.nanmax(x), nbins+1)
    ybins = np.linspace(np.nanmin(y), np.nanmax(y), nbins+1)
    Xi = 0.5*(xbins[:-1]+xbins[1:])
    Yi = 0.5*(ybins[:-1]+ybins[1:])
    Zg = np.full((nbins, nbins), np.nan)
    xi = np.digitize(x, xbins) - 1
    yi = np.digitize(y, ybins) - 1
    m = (xi>=0)&(xi<nbins)&(yi>=0)&(yi<nbins)&np.isfinite(z)
    xi, yi, z = xi[m], yi[m], z[m]
    from collections import defaultdict
    buckets = defaultdict(list)
    for a,b,val in zip(xi, yi, z): buckets[(a,b)].append(val)
    for (a,b), vals in buckets.items(): Zg[b,a] = agg(vals)
    Xg, Yg = np.meshgrid(Xi, Yi)
    return Xg, Yg, Zg

def scatter3(ax, x, y, z, c=None, cmap="viridis", title="", xlabel="", ylabel="", zlabel=""):
    sc = ax.scatter(x, y, z, s=6, alpha=0.6, c=(c if c is not None else z), cmap=cmap, rasterized=True)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    return sc

def surface3(ax, Xg, Yg, Zg, cmap="viridis", title="", xlabel="", ylabel="", zlabel=""):
    Zm = np.ma.masked_invalid(Zg)
    surf = ax.plot_surface(Xg, Yg, Zm, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95, rcount=100, ccount=100)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    return surf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="analyze_bag_*_data.csv（高さ列入り版）")
    ap.add_argument("--out-prefix", default="out/vis3d")
    ap.add_argument("--downsample", type=int, default=0, help=">0 でランダム間引き数（例 8000）")
    ap.add_argument("--nbins", type=int, default=40, help="サーフェスのビン数")
    ap.add_argument("--view", type=str, default="25,-60", help="3D視点 elev,azim 例: '25,-60'")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    elev, azim = map(float, args.view.split(","))

    df = pd.read_csv(args.csv, comment="#")
    req = ["p_sum[MPa]","p_diff[MPa]","p1[MPa]","p2[MPa]","theta[deg]","dz[m]"]
    for k in req:
        if k not in df.columns: raise RuntimeError(f"CSVに列 {k} がありません。")

    if args.downsample and args.downsample>0 and args.downsample < len(df):
        df = df.sample(args.downsample, random_state=42).sort_index()

    ps, pdiff = df["p_sum[MPa]"].to_numpy(), df["p_diff[MPa]"].to_numpy()
    p1, p2 = df["p1[MPa]"].to_numpy(), df["p2[MPa]"].to_numpy()
    th_deg, dz = df["theta[deg]"].to_numpy(), df["dz[m]"].to_numpy()

    # ---- theta vs (p_diff, p_sum)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    sc = scatter3(ax, pdiff, ps, th_deg, c=th_deg, title="theta vs (p_diff, p_sum)",
                  xlabel="p_diff [MPa]", ylabel="p_sum [MPa]", zlabel="theta [deg]")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="theta [deg]")
    ax.view_init(elev=elev, azim=azim); plt.tight_layout()
    plt.savefig(args.out_prefix+"_3d_theta_pdiff_psum.png", dpi=150)

    Xg, Yg, Zg = make_surface(pdiff, ps, th_deg, nbins=args.nbins)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    surf = surface3(ax, Xg, Yg, Zg, title="Mean surface: theta(p_diff, p_sum)",
                    xlabel="p_diff [MPa]", ylabel="p_sum [MPa]", zlabel="theta [deg]")
    fig.colorbar(surf, ax=ax, shrink=0.7, label="theta [deg]")
    ax.view_init(elev=elev, azim=azim); plt.tight_layout()
    plt.savefig(args.out_prefix+"_surf_theta_pdiff_psum.png", dpi=150)

    # ---- theta vs (p1, p2)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    sc = scatter3(ax, p1, p2, th_deg, c=ps, cmap="plasma", title="theta vs (p1, p2)",
                  xlabel="p1 [MPa]", ylabel="p2 [MPa]", zlabel="theta [deg]")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="p_sum [MPa]")
    ax.view_init(elev=elev, azim=45); plt.tight_layout()
    plt.savefig(args.out_prefix+"_3d_theta_p1_p2.png", dpi=150)

    Xg2, Yg2, Zg2 = make_surface(p1, p2, th_deg, nbins=args.nbins)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    surf2 = surface3(ax, Xg2, Yg2, Zg2, cmap="plasma", title="Mean surface: theta(p1, p2)",
                     xlabel="p1 [MPa]", ylabel="p2 [MPa]", zlabel="theta [deg]")
    fig.colorbar(surf2, ax=ax, shrink=0.7, label="theta [deg]")
    ax.view_init(elev=elev, azim=45); plt.tight_layout()
    plt.savefig(args.out_prefix+"_surf_theta_p1_p2.png", dpi=150)

    # ---- dz vs (p_diff, p_sum)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    sc = scatter3(ax, pdiff, ps, dz, c=dz, cmap="cividis", title="dz vs (p_diff, p_sum)",
                  xlabel="p_diff [MPa]", ylabel="p_sum [MPa]", zlabel="dz [m]")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="dz [m]")
    ax.view_init(elev=elev, azim=azim); plt.tight_layout()
    plt.savefig(args.out_prefix+"_3d_dz_pdiff_psum.png", dpi=150)

    Xg3, Yg3, Zg3 = make_surface(pdiff, ps, dz, nbins=args.nbins)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    surf3 = surface3(ax, Xg3, Yg3, Zg3, cmap="cividis", title="Mean surface: dz(p_diff, p_sum)",
                     xlabel="p_diff [MPa]", ylabel="p_sum [MPa]", zlabel="dz [m]")
    fig.colorbar(surf3, ax=ax, shrink=0.7, label="dz [m]")
    ax.view_init(elev=elev, azim=azim); plt.tight_layout()
    plt.savefig(args.out_prefix+"_surf_dz_pdiff_psum.png", dpi=150)

    # ---- dz vs (p1, p2)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    sc = scatter3(ax, p1, p2, dz, c=ps, cmap="magma", title="dz vs (p1, p2)",
                  xlabel="p1 [MPa]", ylabel="p2 [MPa]", zlabel="dz [m]")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="p_sum [MPa]")
    ax.view_init(elev=elev, azim=45); plt.tight_layout()
    plt.savefig(args.out_prefix+"_3d_dz_p1_p2.png", dpi=150)

    Xg4, Yg4, Zg4 = make_surface(p1, p2, dz, nbins=args.nbins)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    surf4 = surface3(ax, Xg4, Yg4, Zg4, cmap="magma", title="Mean surface: dz(p1, p2)",
                     xlabel="p1 [MPa]", ylabel="p2 [MPa]", zlabel="dz [m]")
    fig.colorbar(surf4, ax=ax, shrink=0.7, label="dz [m]")
    ax.view_init(elev=elev, azim=45); plt.tight_layout()
    plt.savefig(args.out_prefix+"_surf_dz_p1_p2.png", dpi=150)

    print("[DONE] saved 3D figures to", os.path.dirname(args.out_prefix))

if __name__ == "__main__":
    main()
