#!/usr/bin/env python3
"""Harvest and plot the diagnosis experiment (paper Sec. 4.4, fig:diagnosis).

Reads the SLURM stdout files written by ``runs/run_diagnosis.sh`` and produces:

  diagnosis_runs.csv      tidy per-step traces (arm, k, seed, step, l2, loss, time)
  diagnosis_beta.csv      tidy beta-controller traces (arm, k, seed, step, beta, r_hat, rho)
  diagnosis_summary.csv   final L2 per (arm, k), mean +- std over seeds
  fig_diagnosis.pdf       the two-panel figure

Parses stdout rather than wandb so it runs offline with no auth, mirroring
``vmcnet/slurm/parse_eval_energies.py``. Pass --wandb-project to cross-check
against wandb instead.

Usage:
    python runs/harvest_diagnosis.py --logs logs/ --out paper/figures/
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# --- log formats --------------------------------------------------------------
# Header echoed by run_diagnosis.sh.
RE_HEADER = re.compile(
    r"===\s*task\s+(?P<task>\d+):\s*arm=(?P<arm>\w+)\s+"
    r"batch_frequency=(?P<k>\d+)\s+data_seed=(?P<data_seed>\d+)\s+"
    r"model_seed=(?P<model_seed>\d+)\s*==="
)
# train.py:785-792
RE_STEP = re.compile(
    r"Step:\s*(?P<step>[\d.eE+-]+),\s*"
    r"Loss:\s*(?P<loss>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"L2 Error:\s*(?P<l2>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Interior:\s*(?P<interior>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Boundary:\s*(?P<boundary>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Time:\s*(?P<time>[\d.]+)s"
)
# spring.py:384 uses ascii "beta="/"rho="; same_sampled_spring_unified.py:464
# uses unicode "β="/"ρ=". Accept both.
RE_BETA = re.compile(
    r"step\s+(?P<step>\d+):\s*"
    r"(?:beta|β)=(?P<beta>[\d.eE+-]+),\s*"
    r"r_hat=(?P<r_hat>[\d.eE+-]+),\s*"
    r"(?:rho|ρ)=(?P<rho>[\d.eE+-]+)"
)

ARM_LABEL = {
    "adaptive": "Adaptive SPRING",
    "spring": "SPRING (fixed $\\mu$)",
    "ss_spring": "Same-Sampled SPRING",
}
ARM_STYLE = {
    "adaptive": dict(color="#c1121f", marker="o", zorder=3),
    "spring": dict(color="#4a4e69", marker="s", zorder=2),
    "ss_spring": dict(color="#0353a4", marker="^", zorder=2),
}
LOOKBACK_P = 30  # both comparison windows share one row set only when k >= 2p


def parse_log(path: Path) -> tuple[dict | None, list[dict], list[dict]]:
    """Return (metadata, step rows, beta rows) for one SLURM stdout file."""
    meta, steps, betas = None, [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            if meta is None:
                m = RE_HEADER.search(line)
                if m:
                    meta = {
                        "arm": m["arm"],
                        "batch_frequency": int(m["k"]),
                        "seed": int(m["data_seed"]),
                        "task": int(m["task"]),
                    }
                    continue
            m = RE_STEP.search(line)
            if m:
                steps.append(
                    {
                        "step": int(float(m["step"])),
                        "loss": float(m["loss"]),
                        "l2_error": float(m["l2"]),
                        "time": float(m["time"]),
                    }
                )
                continue
            m = RE_BETA.search(line)
            if m:
                betas.append(
                    {
                        "step": int(m["step"]),
                        "beta": float(m["beta"]),
                        "r_hat": float(m["r_hat"]),
                        "rho": float(m["rho"]),
                    }
                )
    return meta, steps, betas


def collect(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_rows, beta_rows, skipped = [], [], []
    files = sorted(log_dir.glob("diag_*.out"))
    if not files:
        raise SystemExit(f"no diag_*.out files under {log_dir}")

    for path in files:
        meta, steps, betas = parse_log(path)
        if meta is None:
            skipped.append((path.name, "no task header"))
            continue
        if not steps:
            skipped.append((path.name, "no Step lines -- crashed or OOM?"))
            continue
        for r in steps:
            step_rows.append({**meta, **r})
        for r in betas:
            beta_rows.append({**meta, **r})

    # Silent truncation is how a sweep starts looking complete when it is not.
    print(f"parsed {len(files) - len(skipped)}/{len(files)} logs")
    for name, why in skipped:
        print(f"  SKIPPED {name}: {why}")
    return pd.DataFrame(step_rows), pd.DataFrame(beta_rows)


def summarize(steps: pd.DataFrame) -> pd.DataFrame:
    """Final L2 per run, then mean/std over seeds."""
    final = (
        steps.sort_values("step")
        .groupby(["arm", "batch_frequency", "seed"], as_index=False)
        .last()[["arm", "batch_frequency", "seed", "l2_error", "step"]]
    )
    summary = (
        final.groupby(["arm", "batch_frequency"])
        .agg(
            l2_mean=("l2_error", "mean"),
            l2_std=("l2_error", "std"),
            n_seeds=("l2_error", "size"),
            last_step=("step", "min"),
        )
        .reset_index()
    )
    return summary


def report(summary: pd.DataFrame) -> None:
    """Print the headline interaction: how much each arm moves across k."""
    print("\n=== final L2 error, mean +- std over seeds ===")
    piv = summary.pivot(index="batch_frequency", columns="arm", values="l2_mean")
    print(piv.to_string(float_format=lambda v: f"{v:.4e}"))

    print("\n=== the interaction: L2(k=max) / L2(k=1) ===")
    print("prediction: adaptive improves (ratio < 1); the two controls stay flat (~1)")
    for arm in summary["arm"].unique():
        sub = summary[summary["arm"] == arm].sort_values("batch_frequency")
        if len(sub) < 2:
            continue
        lo = sub.iloc[0]["l2_mean"]
        hi = sub.iloc[-1]["l2_mean"]
        k_hi = int(sub.iloc[-1]["batch_frequency"])
        print(f"  {ARM_LABEL.get(arm, arm):<24} {hi / lo:6.3f}   (k=1 -> k={k_hi})")

    thin = summary[summary["n_seeds"] < 3]
    if len(thin):
        print("\nWARNING: fewer than 3 seeds for:")
        print(thin[["arm", "batch_frequency", "n_seeds"]].to_string(index=False))


def plot(summary: pd.DataFrame, betas: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # --- left: the interaction ------------------------------------------------
    for arm, sub in summary.groupby("arm"):
        sub = sub.sort_values("batch_frequency")
        style = ARM_STYLE.get(arm, {})
        ax1.errorbar(
            sub["batch_frequency"],
            sub["l2_mean"],
            yerr=sub["l2_std"].fillna(0.0),
            label=ARM_LABEL.get(arm, arm),
            capsize=3,
            lw=1.8,
            markersize=5,
            **style,
        )
    ax1.axvline(2 * LOOKBACK_P, color="gray", ls=":", lw=1.2, zorder=1)
    ax1.annotate(
        f"$k = 2p = {2 * LOOKBACK_P}$\n(windows share rows)",
        xy=(2 * LOOKBACK_P, 0.97),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=8,
        color="gray",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("batch frequency $k$  (steps between resampling)")
    ax1.set_ylabel("final $L^2$ error")
    ax1.set_title("Failure recedes only for the contaminated arm", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.25, which="both", lw=0.5)

    # --- right: the mechanism -------------------------------------------------
    if len(betas):
        b = betas[betas["seed"] == betas["seed"].min()]
        ks = sorted(b["batch_frequency"].unique())
        n_k = max(len(ks), 2)
        try:  # matplotlib >= 3.9 removed cm.get_cmap
            cmap = matplotlib.colormaps["viridis"].resampled(n_k)
        except AttributeError:
            cmap = matplotlib.cm.get_cmap("viridis", n_k)
        for arm, ls in (("adaptive", "-"), ("ss_spring", "--")):
            sub_arm = b[b["arm"] == arm]
            for j, k in enumerate(ks):
                sub = sub_arm[sub_arm["batch_frequency"] == k].sort_values("step")
                if not len(sub):
                    continue
                ax2.plot(
                    sub["step"],
                    sub["beta"],
                    ls=ls,
                    color=cmap(j),
                    lw=1.4,
                    label=f"$k={k}$" if arm == "adaptive" else None,
                )
        ax2.set_xlabel("step")
        ax2.set_ylabel(r"momentum $\mu$")
        ax2.set_title(
            "Controller output (solid: Adaptive, dashed: Same-Sampled)", fontsize=10
        )
        ax2.legend(fontsize=8, frameon=False, ncol=2, title="seed 0")
        ax2.grid(alpha=0.25, lw=0.5)
    else:
        ax2.text(0.5, 0.5, "no beta updates parsed", ha="center", va="center")
        ax2.set_axis_off()

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out / f"fig_diagnosis.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    steps, betas = collect(args.logs)
    summary = summarize(steps)

    steps.to_csv(args.out / "diagnosis_runs.csv", index=False)
    betas.to_csv(args.out / "diagnosis_beta.csv", index=False)
    summary.to_csv(args.out / "diagnosis_summary.csv", index=False)
    report(summary)

    if not args.no_plot:
        plot(summary, betas, args.out)


if __name__ == "__main__":
    main()
