#!/usr/bin/env python3
"""Cache the E5(c) learning curves from wandb into one tidy CSV.

`runs/plot_e5.py` reads that CSV, so the figure can be re-made offline and the
numbers behind it are inspectable without a wandb login.

The project is `pinn_e5_final`, written by `runs/e5_final.sh` -- 48 runs, named
`{arm}_{sys}_m{model_seed}` for the four arms, four systems and seeds 1-3.

Usage:
    python runs/e5_fetch_curves.py                     # writes runs/e5_curves.csv
    python runs/e5_fetch_curves.py --entity ENT --out PATH
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from re import fullmatch

import pandas as pd
import wandb

PROJECT = "pinn_e5_final"
NAME_RE = r"(?P<arm>engdw|spring|primesr|ssspring)_(?P<sys>p5|p100|heat4|lfp9)_m(?P<seed>\d+)"
# Config keys worth carrying alongside the curve, for the hyperparameter table.
CFG_KEYS = ("lr", "damping", "momentum", "norm_constraint", "num_seconds", "equation")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=None, help="wandb entity (default: yours)")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "e5_curves.csv",
    )
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    runs = list(api.runs(f"{entity}/{args.project}"))
    print(f"{entity}/{args.project}: {len(runs)} runs")

    frames = []
    for run in sorted(runs, key=lambda r: r.name):
        m = fullmatch(NAME_RE, run.name)
        if m is None:
            print(f"  SKIP {run.name} (name does not match the E5-final pattern)")
            continue
        if run.state != "finished":
            print(f"  SKIP {run.name} (state={run.state})")
            continue

        hist = run.history(keys=["step", "time", "l2_error", "loss"], pandas=True)
        if hist.empty:
            print(f"  SKIP {run.name} (empty history)")
            continue

        hist = hist[["step", "time", "l2_error", "loss"]].copy()
        hist["arm"] = m["arm"]
        hist["system"] = m["sys"]
        hist["seed"] = int(m["seed"])
        cfg = run.config
        for key in CFG_KEYS:
            hist[key] = cfg.get(key)
        frames.append(hist)
        print(f"  {run.name}: {len(hist)} logged points, final l2={hist.l2_error.iloc[-1]:.3e}")

    df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows, {df.groupby(['arm','system','seed']).ngroups} runs)")


if __name__ == "__main__":
    main()
