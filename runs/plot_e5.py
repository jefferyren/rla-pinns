#!/usr/bin/env python3
"""The E5 four-panel figure: L2 error vs wall-clock on the four published systems.

Replicates the layout of Figure 3 of arXiv:2505.12149 (Guzman-Cordero, Dangel,
Goldshlager, Zeinhofer) -- same four panels, same axes, same D values, same
budgets -- with two of its four arms swapped: its `ENGD-W (Line Search)` and its
`KFAC` reference line are dropped, and PRIME-SR and SS-SPRING take their place.
`ENGD (Woodbury)` and `SPRING` keep the reference figure's blue and green so the
two shared arms line up visually when a reader lays the figures side by side.

Unlike Figure 3, which plots the single best run of each sweep, each curve here
is the MEDIAN over model seeds 1-3 with a min-max band, because four arms can
sit inside seed noise (PINN_EXPERIMENTS.md, E5(c)).

Reads runs/e5_curves.csv (written by runs/e5_fetch_curves.py) -- no wandb login
needed to re-make the figure.

Usage:
    python runs/plot_e5.py
    python runs/plot_e5.py --usetex --out paper/figures/e5_four_panel.pdf
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from palettable.colorbrewer import sequential
from tueplots import bundles

HERE = Path(__file__).parent
REPO = HERE.parent

# Panel order, titles, and D. Every D was checked by direct parameter count
# against the D printed in that paper's own panel titles and all four match
# exactly (PINN_EXPERIMENTS.md, E5). Titles are the reference figure's, verbatim.
PANELS = [
    ("p5", "5d Poisson", 10065),
    ("p100", "100d Poisson", 1325057),
    ("heat4", "4d Heat", 116865),
    ("lfp9", "9+1d log-Fokker–Planck", 118145),
]

ARMS = ["engdw", "spring", "primesr", "ssspring"]
LABEL = {
    "engdw": "ENGD (Woodbury)",
    "spring": "SPRING",
    "primesr": "PRIME-SR",
    "ssspring": "SS-SPRING (ours)",
}
COLOR = {
    # the reference figure's own two colours, kept for the two shared arms
    "engdw": sequential.Blues_5.mpl_colors[-3],
    "spring": sequential.Greens_4.mpl_colors[-3],
    # the two swapped-in arms
    "primesr": sequential.Reds_4.mpl_colors[-2],
    "ssspring": "black",
}
LINEWIDTH = {"engdw": 1.0, "spring": 1.0, "primesr": 1.0, "ssspring": 1.3}
ZORDER = {"engdw": 2, "spring": 3, "primesr": 4, "ssspring": 5}

NGRID = 300  # points on the shared log-time grid


def band(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and min-max of one arm's seeds on a shared log-spaced time grid.

    The seeds take different numbers of steps in the same wall-clock budget, so
    they are interpolated onto a common grid before being combined. The grid is
    the interval where all seeds are defined -- from the latest first log to the
    earliest last log -- so no part of the band is carried by fewer seeds than
    the rest of it. Interpolation is linear in log-log, matching the axes.
    """
    curves, lo, hi = [], [], []
    for _, run in sub.groupby("seed"):
        run = run.sort_values("time")
        t, l2 = run["time"].to_numpy(), run["l2_error"].to_numpy()
        keep = (t > 0) & np.isfinite(l2) & (l2 > 0)
        t, l2 = t[keep], l2[keep]
        curves.append((np.log10(t), np.log10(l2)))
        lo.append(t[0])
        hi.append(t[-1])

    grid = np.logspace(np.log10(max(lo)), np.log10(min(hi)), NGRID)
    stack = np.vstack([np.interp(np.log10(grid), lt, ll) for lt, ll in curves])
    return grid, 10 ** np.median(stack, 0), 10 ** stack.min(0), 10 ** stack.max(0)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=HERE / "e5_curves.csv")
    parser.add_argument(
        "--out", type=Path, default=REPO / "paper" / "figures" / "e5_four_panel.pdf"
    )
    parser.add_argument(
        "--usetex", action="store_true", help="render text with LaTeX (needs a TeX install)"
    )
    parser.add_argument(
        "--no-band", action="store_true", help="median only, no min-max band"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.curves)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # nrows/ncols here set the figure's aspect, not the grid: 4x4 is what the
    # reference figure's own plotting scripts use, and it is what puts the text
    # at the right size relative to the panels.
    rc = bundles.neurips2023(rel_width=1.0, nrows=4, ncols=4, usetex=args.usetex)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2)

        for ax, (system, name, num_params) in zip(axes.flatten(), PANELS):
            ax.set_title(f"{name} ($D = {num_params}$)")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"$L_2$ error")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.5)

            for arm in ARMS:
                sub = df[(df.system == system) & (df.arm == arm)]
                if sub.empty:
                    print(f"  WARNING: no runs for {arm}/{system}")
                    continue
                grid, med, lo, hi = band(sub)
                ax.plot(
                    grid,
                    med,
                    label=LABEL[arm],
                    color=COLOR[arm],
                    linewidth=LINEWIDTH[arm],
                    zorder=ZORDER[arm],
                )
                if not args.no_band:
                    ax.fill_between(
                        grid,
                        lo,
                        hi,
                        color=COLOR[arm],
                        alpha=0.2,
                        linewidth=0,
                        zorder=ZORDER[arm] - 0.5,
                    )
                print(
                    f"  {system:6s} {arm:9s} n_seeds={sub.seed.nunique()} "
                    f"final median={med[-1]:.3e} [{lo[-1]:.3e}, {hi[-1]:.3e}]"
                )
            ax.set_xlim(left=1.0)

        handles, labels = axes.flatten()[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            handlelength=1.35,
            ncols=4,
            columnspacing=0.9,
        )

        fig.savefig(args.out, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=300)

    print(f"\nwrote {args.out}\n      {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
