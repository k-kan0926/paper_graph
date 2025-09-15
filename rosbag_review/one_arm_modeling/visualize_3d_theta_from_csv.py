#!/usr/bin/env python3
#python visualize_3d_theta_from_csv.py   --csv out/diff_run1_data.csv   --out-prefix out/vis3d   --downsample 8000   --nbins 40#
#作成したcsvから3D散布図とサーフェスを描く
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D)

def make_surface(x, y, z, nbins=40, agg=np.nanmean):
    """(x,y) をグリッド化して z の平均面を作る（SciPy不要）。"""
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    xbins = np.linspace(np.nanmin(x), np.nanmax(x), nbins+1)
    ybins = np.linspace(np.nanmin(y), np.nanmax(y), nbins+1)
    Xi = 0.5*(xbins[:-1]+xbins[1:])
    Yi = 0.5*(ybins[:-1]+ybins[1:])
    Zg = np.full((nbins, nbins), np.nan)
    # bin index
    xi = np.digitize(x, xbins)-1
    yi = np.digitize(y, ybins)-1
    mask = (xi>=0)&(xi<nbins)&(yi>=0)&(yi<nbins)&np.isfinite(z)
    xi = xi[mask]; yi = yi[mask]; z = z[mask]
    # 集計
    from collections import defaultdict
    buckets = defaultdict(list)
    for a,b,val in zip(xi,yi,z): buckets[(a,b)].append(val)
    for (a,b), vals in buckets.items():
        Zg[b,a] = agg(vals)  # 行=Y、列=X
    Xg, Yg = np.meshgrid(Xi, Yi)
    return Xg, Yg, Zg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="analyze_bag_*_data.csv")
    ap.add_argument("--out-prefix", default="out/vis3d")
    ap.add_argument("--downsample", type=int, default=0, help=">0 でランダム間引き数（例 8000）")
    ap.add_argument("--nbins", type=int, default=40, help="サーフェス用のビン数")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    df = pd.read_csv(args.csv, comment="#")
    # 列名は以前のCSV仕様に合わせています：
    # t[s],p_sum[MPa],p_diff[MPa],p1[MPa],p2[MPa],p1_cmd[MPa],p2_cmd[MPa],theta[rad],theta[deg]
    need = ["p_sum[MPa]","p_diff[MPa]","p1[MPa]","p2[MPa]","theta[rad]","theta[deg]"]
    for k in need:
        if k not in df.columns:
            raise RuntimeError(f"CSV列が見つかりません: {k}")

    # 間引き（任意）
    if args.downsample and args.downsample>0 and args.downsample < len(df):
        df = df.sample(args.downsample, random_state=42).sort_index()

    ps = df["p_sum[MPa]"].to_numpy()
    pdiff = df["p_diff[MPa]"].to_numpy()
    p1 = df["p1[MPa]"].to_numpy()
    p2 = df["p2[MPa]"].to_numpy()
    th = df["theta[rad]"].to_numpy()
    th_deg = df["theta[deg]"].to_numpy()

    # ===== 3D scatter: (p_diff, p_sum, theta) =====
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pdiff, ps, th_deg, s=6, alpha=0.6, c=th_deg, cmap="viridis")
    ax.set_xlabel("p_diff [MPa]"); ax.set_ylabel("p_sum [MPa]"); ax.set_zlabel("theta [deg]")
    ax.set_title("3D scatter: theta vs (p_diff, p_sum)")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="theta [deg]")
    ax.view_init(elev=25, azim=-60)  # 見やすい初期視点
    plt.tight_layout()
    f1 = args.out_prefix + "_3d_pdiff_psum_theta.png"
    plt.savefig(f1, dpi=150); print("[OK] wrote", f1)

    # 擬似サーフェス（平均面）
    Xg, Yg, Zg = make_surface(pdiff, ps, th_deg, nbins=args.nbins)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    # 欠損が多いと plot_surface が警告を出すので mask
    from matplotlib import cm
    Zm = np.ma.masked_invalid(Zg)
    surf = ax.plot_surface(Xg, Yg, Zm, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("p_diff [MPa]"); ax.set_ylabel("p_sum [MPa]"); ax.set_zlabel("theta [deg]")
    ax.set_title("Mean surface: theta(p_diff, p_sum)")
    fig.colorbar(surf, ax=ax, shrink=0.7, label="theta [deg]")
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    f1s = args.out_prefix + "_surface_pdiff_psum_theta.png"
    plt.savefig(f1s, dpi=150); print("[OK] wrote", f1s)

    # 2D heatmap（上から見た等高図）
    fig = plt.figure(figsize=(7,6))
    plt.pcolormesh(Xg, Yg, Zg, shading="auto", cmap="viridis")
    plt.colorbar(label="theta [deg]")
    plt.xlabel("p_diff [MPa]"); plt.ylabel("p_sum [MPa]")
    plt.title("theta mean heatmap (p_diff vs p_sum)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    f1h = args.out_prefix + "_heatmap_pdiff_psum_theta.png"
    plt.savefig(f1h, dpi=150); print("[OK] wrote", f1h)

    # ===== 3D scatter: (p1, p2, theta) =====
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    sc2 = ax.scatter(p1, p2, th_deg, s=6, alpha=0.6, c=ps, cmap="plasma")  # 色に p_sum を使用
    ax.set_xlabel("p1 [MPa]"); ax.set_ylabel("p2 [MPa]"); ax.set_zlabel("theta [deg]")
    ax.set_title("3D scatter: theta vs (p1, p2)")
    fig.colorbar(sc2, ax=ax, shrink=0.7, label="p_sum [MPa]")
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    f2 = args.out_prefix + "_3d_p1_p2_theta.png"
    plt.savefig(f2, dpi=150); print("[OK] wrote", f2)

    # 擬似サーフェス：theta(p1,p2)
    Xg2, Yg2, Zg2 = make_surface(p1, p2, th_deg, nbins=args.nbins)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")
    Zm2 = np.ma.masked_invalid(Zg2)
    surf2 = ax.plot_surface(Xg2, Yg2, Zm2, cmap="plasma", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("p1 [MPa]"); ax.set_ylabel("p2 [MPa]"); ax.set_zlabel("theta [deg]")
    ax.set_title("Mean surface: theta(p1, p2)")
    fig.colorbar(surf2, ax=ax, shrink=0.7, label="theta [deg]")
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    f2s = args.out_prefix + "_surface_p1_p2_theta.png"
    plt.savefig(f2s, dpi=150); print("[OK] wrote", f2s)

    # 2D heatmap（p1,p2）
    fig = plt.figure(figsize=(7,6))
    plt.pcolormesh(Xg2, Yg2, Zg2, shading="auto", cmap="plasma")
    plt.colorbar(label="theta [deg]")
    plt.xlabel("p1 [MPa]"); plt.ylabel("p2 [MPa]")
    plt.title("theta mean heatmap (p1 vs p2)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    f2h = args.out_prefix + "_heatmap_p1_p2_theta.png"
    plt.savefig(f2h, dpi=150); print("[OK] wrote", f2h)

    print("[DONE]")

if __name__ == "__main__":
    main()
