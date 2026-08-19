#!/usr/bin/env python3
"""Harvest E2 (equal-budget tuning) from stdout logs -- no wandb auth needed.

Emits two things:

  best_hparams.csv   one row per (arm, pde): the winning (lr, damping, momentum).
                     runs/e3_final.sh reads this, so E3 cannot silently run on
                     hand-typed numbers.

  spread table       min / median / max final l2_error over all 13 draws, per
                     (arm, pde). THIS is the C1 evidence -- report it, not just
                     the winner. If Adaptive SPRING's best of 13 still trails
                     tuned SPRING's best of 13, the failure is not a tuning
                     artifact.

Usage:
    python runs/harvest_e2.py --logs logs                  # summary + csv
    python runs/harvest_e2.py --logs logs --out runs/best_hparams.csv
"""
from __future__ import annotations

import csv
import re
from argparse import ArgumentParser
from pathlib import Path
from statistics import median

# Header echoed by runs/e2_tune.sh.
RE_HEADER = re.compile(
    r"===\s*E2\s+task\s+(?P<task>\d+):\s*arm=(?P<arm>\w+)\s+pde=(?P<pde>\w+)\s+"
    r"draw=(?P<draw>\d+)/(?P<ndraws>\d+)\s+lr=(?P<lr>\S+)\s+"
    r"damping=(?P<damping>\S+)\s+momentum=(?P<momentum>\S+)\s*==="
)
# train.py:803-810
RE_STEP = re.compile(
    r"Step:\s*(?P<step>[\d.eE+-]+),\s*"
    r"Loss:\s*(?P<loss>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"L2 Error:\s*(?P<l2>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Interior:\s*(?P<interior>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Boundary:\s*(?P<boundary>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Time:\s*(?P<time>[\d.]+)s"
)

ARM_LABEL = {
    "a1_spring": "SPRING (tuned, fixed mu)",
    "a2_adaptive": "Adaptive SPRING",
    "a3_ss": "SS-SPRING (ours)",
}


def parse_log(path: Path) -> dict | None:
    """Return one run's metadata plus its final metrics, or None if unusable."""
    text = path.read_text(errors="replace")
    h = RE_HEADER.search(text)
    if h is None:
        return None
    steps = list(RE_STEP.finditer(text))
    if not steps:
        # Ran but never logged: died in setup, or was killed before the first
        # log. Distinguishable from a diverged run and must not be silently
        # dropped, so it is reported as a failure rather than skipped.
        return dict(h.groupdict(), l2=None, final_step=None, elapsed=None,
                    log=path.name, status="no-metrics")
    last = steps[-1]
    l2 = float(last.group("l2"))
    status = "ok"
    if l2 != l2 or l2 in (float("inf"), float("-inf")):  # NaN / inf
        status = "diverged"
    return dict(h.groupdict(), l2=l2, final_step=float(last.group("step")),
                elapsed=float(last.group("time")), log=path.name, status=status)


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument("--logs", default="logs", help="directory of e2_*.out files")
    p.add_argument("--out", default="runs/best_hparams.csv",
                   help="where to write the winners csv")
    p.add_argument("--pattern", default="e2_*.out")
    args = p.parse_args()

    logdir = Path(args.logs)
    if not logdir.is_dir():
        raise SystemExit(f"no such directory: {logdir}")
    runs = [r for r in (parse_log(f) for f in sorted(logdir.glob(args.pattern)))
            if r is not None]
    if not runs:
        raise SystemExit(f"no parseable {args.pattern} logs in {logdir}")

    ok = [r for r in runs if r["status"] == "ok"]
    bad = [r for r in runs if r["status"] != "ok"]

    expected = 78
    print(f"parsed {len(runs)} logs: {len(ok)} usable, {len(bad)} unusable")
    if len(runs) < expected:
        print(f"  WARNING: expected {expected} tasks, found {len(runs)}."
              " Check for tasks that never wrote a log at all.")
    for r in bad:
        print(f"  {r['status']:12s} {r['arm']:12s} {r['pde']:5s} "
              f"draw {r['draw']:>2s}  ({r['log']})")

    cells: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        cells.setdefault((r["arm"], r["pde"]), []).append(r)

    print(f"\n{'arm':26s} {'pde':6s} {'n':>3s} {'best':>11s} {'median':>11s} "
          f"{'worst':>11s}  winning (lr, damping, momentum)")
    print("-" * 108)
    winners = []
    for pde in ("p100", "lfp9"):
        for arm in ("a1_spring", "a2_adaptive", "a3_ss"):
            rs = cells.get((arm, pde))
            if not rs:
                print(f"{ARM_LABEL[arm]:26s} {pde:6s}   -   "
                      f"{'NO USABLE RUNS':>11s}")
                continue
            vals = sorted(r["l2"] for r in rs)
            best = min(rs, key=lambda r: r["l2"])
            print(f"{ARM_LABEL[arm]:26s} {pde:6s} {len(rs):3d} "
                  f"{vals[0]:11.4e} {median(vals):11.4e} {vals[-1]:11.4e}"
                  f"  ({best['lr']}, {best['damping']}, {best['momentum']})"
                  f"  draw {best['draw']}")
            winners.append(dict(
                arm=arm, pde=pde, lr=best["lr"], damping=best["damping"],
                momentum=best["momentum"], l2_error=f"{best['l2']:.6e}",
                draw=best["draw"], n_usable=len(rs), log=best["log"]))
        print()

    if len(winners) < 6:
        print("WARNING: fewer than 6 (arm, pde) cells have a winner."
              " E3 needs all six before it can run.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(winners[0]))
        w.writeheader()
        w.writerows(winners)
    print(f"wrote {out}  ({len(winners)} rows)")
    print("\nC1 reading: compare a2_adaptive's BEST against a1_spring's BEST.")
    print("If a2 still trails on both PDEs, the failure is not a tuning artifact.")
    print("If a2 matches a1 on both, C1 is dead as stated -- see E2's kill criterion.")


if __name__ == "__main__":
    main()
