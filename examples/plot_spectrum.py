"""Plot a spectrum.csv or spatial.csv produced by the Julia examples.

Usage: python examples/plot_spectrum.py outputs/grid_.../spectrum.csv
Dependencies: numpy and matplotlib (only needed for this optional plot step).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        parser.error("empty CSV")
    if not all(np.isfinite(list(row.values())).all() for row in rows):
        parser.error("non-finite values in CSV")
    values = np.array([abs(complex(row["rho_re"], row["rho_im"])) for row in rows])
    vmax = values.max()
    plt.rcParams.update({"font.size": 10, "axes.grid": False, "savefig.dpi": 300})
    if {"x", "y", "z"}.issubset(rows[0]):
        fig = plt.figure(figsize=(4.3, 4.3), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        xyz = np.array([[row[k] for k in ("x", "y", "z")] for row in rows])
        sizes = 15 + 160 * values / (vmax or 1)
        artist = ax.scatter(*xyz.T, c=values, s=sizes, cmap="viridis", vmin=0, vmax=vmax or 1)
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.grid(False)
    else:
        sites = sorted({int(row["site0"]) for row in rows})
        fig, axes = plt.subplots(1, len(sites), figsize=(3.5*len(sites), 3.6),
                                 squeeze=False, layout="constrained")
        ax = axes.ravel().tolist()
        for axis, site in zip(ax, sites):
            part = [row for row in rows if row["site0"] == site]
            points = {(row["re"], row["im"]): abs(complex(row["rho_re"], row["rho_im"])) for row in part}
            if len(points) != len(part):
                parser.error("duplicate site/energy rows; select one parameter set first")
            re = sorted({r["re"] for r in part})
            im = sorted({r["im"] for r in part})
            if len(re)>1 and len(im)>1 and len(part)==len(re)*len(im):
                z = np.array([[points[(r,i)] for i in im] for r in re])
                artist = axis.pcolormesh(im, re, z, shading="nearest", cmap="viridis", vmin=0, vmax=vmax or 1)
            else:
                artist = axis.scatter([r["im"] for r in part], [r["re"] for r in part],
                    c=list(points.values()), cmap="viridis", vmin=0, vmax=vmax or 1, s=45)
            axis.set(xlabel="Im(E)", ylabel="Re(E)", title=f"site0 = {site}")
    cbar = fig.colorbar(artist, ax=ax, orientation="horizontal", pad=0.13, fraction=0.08)
    cbar.set_label(r"$|\rho|$")
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    path = args.output or args.csv.with_suffix(".png")
    fig.savefig(path)
    print(path)


if __name__ == "__main__":
    main()
