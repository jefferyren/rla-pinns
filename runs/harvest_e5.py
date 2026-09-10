#!/usr/bin/env python3
"""Harvest E5 from stdout logs -- no wandb auth needed.

Two stages, because they answer different questions.

  --stage tune   reads logs/e5tune_*.out and emits

                   e5_best_hparams.csv  one row per (arm, system): the winning
                                        (lr, damping, momentum, norm_constraint).
                                        runs/e5_final.sh reads this and refuses
                                        to run without it, so E5(c) can never
                                        quietly execute on hand-typed numbers.

                   spread table         min / median / max final l2_error over
                                        all 24 draws, per (arm, system). REPORT
                                        THIS, not just the winner: a method
                                        whose median draw diverges is a
                                        different claim from one whose median
                                        draw is merely worse than its best, and
                                        the winner alone cannot tell them apart.

  --stage final  reads logs/e5final_*.out and emits the per-(arm, system) median
                 and min-max over model seeds 1-3, plus measured s/step. This is
                 the table that goes under the four-panel figure.

Usage:
    python runs/harvest_e5.py --stage tune  --logs logs
    python runs/harvest_e5.py --stage tune  --logs logs --out runs/e5_best_hparams.csv
    python runs/harvest_e5.py --stage final --logs logs
"""
from __future__ import annotations

import csv
from argparse import ArgumentParser
from pathlib import Path
from re import compile as re_compile
from statistics import median

# Header echoed by runs/e5_tune.sh.
RE_TUNE_HEADER = re_compile(
    r"===\s*E5-TUNE\s+task\s+(?P<task>\d+):\s*arm=(?P<arm>\w+)\s+sys=(?P<sys>\w+)\s+"
    r"draw=(?P<draw>\d+)/(?P<ndraws>\d+)\s+lr=(?P<lr>\S+)\s+damping=(?P<damping>\S+)\s+"
    r"momentum=(?P<momentum>\S+)\s+nc=(?P<nc>\S+)\s*==="
)
# Header echoed by runs/e5_final.sh.
RE_FINAL_HEADER = re_compile(
    r"===\s*E5-FINAL\s+task\s+(?P<task>\d+):\s*arm=(?P<arm>\w+)\s+sys=(?P<sys>\w+)\s+"
    r"model_seed=(?P<seed>\d+)\s+source=(?P<source>\w+)\s+lr=(?P<lr>\S+)\s+"
    r"damping=(?P<damping>\S+)\s+momentum=(?P<momentum>\S+)\s+nc=(?P<nc>\S+)\s+"
    r"budget=(?P<budget>\d+)\s*==="
)
# train.py's per-log line.
RE_STEP = re_compile(
    r"Step:\s*(?P<step>[\d.eE+-]+),\s*"
    r"Loss:\s*(?P<loss>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"L2 Error:\s*(?P<l2>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Interior:\s*(?P<interior>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Boundary:\s*(?P<boundary>[\d.eE+-]+|nan|inf|-inf),\s*"
    r"Time:\s*(?P<time>[\d.]+)s"
)

SYSTEMS = ("p5", "p100", "heat4", "lfp9")
TUNE_ARMS = ("primesr", "ssspring")
FINAL_ARMS = ("engdw", "spring", "primesr", "ssspring")
ARM_LABEL = {
    "engdw": "ENGD (Woodbury)",
    "spring": "SPRING",
    "primesr": "PRIME-SR",
    "ssspring": "SS-SPRING (ours)",
}
SYS_LABEL = {
    "p5": "5d Poisson",
    "p100": "100d Poisson",
    "heat4": "4+1d Heat",
    "lfp9": "9+1d log-FP",
}


def parse_log(path: Path, header_re) -> dict | None:
    """Return one run's metadata plus its final metrics, or None if unusable."""
    text = path.read_text(errors="replace")
    h = header_re.search(text)
    if h is None:
        return None
    steps = list(RE_STEP.finditer(text))
    if not steps:
        # Ran but never logged: died in setup, or was killed before the first
        # log. Distinguishable from a diverged run and must not be silently
        # dropped, so it is reported as a failure rather than skipped.
        return dict(h.groupdict(), l2=None, final_step=None, elapsed=None,
                    s_per_step=None, log=path.name, status="no-metrics")
    last = steps[-1]
    l2 = float(last.group("l2"))
    step = float(last.group("step"))
    elapsed = float(last.group("time"))
    status = "ok"
    if l2 != l2 or l2 in (float("inf"), float("-inf")):  # NaN / inf
        status = "diverged"
    return dict(h.groupdict(), l2=l2, final_step=step, elapsed=elapsed,
                s_per_step=(elapsed / step if step else None),
                log=path.name, status=status)


def load(logdir: Path, pattern: str, header_re) -> tuple[list[dict], list[dict]]:
    if not logdir.is_dir():
        raise SystemExit(f"no such directory: {logdir}")
    runs = [r for r in (parse_log(f, header_re) for f in sorted(logdir.glob(pattern)))
            if r is not None]
    if not runs:
        raise SystemExit(f"no parseable {pattern} logs in {logdir}")
    ok = [r for r in runs if r["status"] == "ok"]
    bad = [r for r in runs if r["status"] != "ok"]
    print(f"parsed {len(runs)} logs: {len(ok)} usable, {len(bad)} unusable")
    for r in bad:
        print(f"  {r['status']:12s} {r['arm']:9s} {r['sys']:6s}  ({r['log']})")
    return ok, bad


def harvest_tune(logdir: Path, pattern: str, out: Path, expected: int) -> None:
    ok, _ = load(logdir, pattern, RE_TUNE_HEADER)
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        cells.setdefault((r["arm"], r["sys"]), []).append(r)

    print(f"\n{'arm':18s} {'system':13s} {'n':>3s} {'best':>11s} {'median':>11s} "
          f"{'worst':>11s}  winning (lr, damping, momentum, nc)")
    print("-" * 118)
    winners = []
    for sys_ in SYSTEMS:
        for arm in TUNE_ARMS:
            rs = cells.get((arm, sys_))
            if not rs:
                print(f"{ARM_LABEL[arm]:18s} {SYS_LABEL[sys_]:13s}   -   "
                      f"{'NO USABLE RUNS':>11s}")
                continue
            vals = sorted(r["l2"] for r in rs)
            best = min(rs, key=lambda r: r["l2"])
            print(f"{ARM_LABEL[arm]:18s} {SYS_LABEL[sys_]:13s} {len(rs):3d} "
                  f"{vals[0]:11.4e} {median(vals):11.4e} {vals[-1]:11.4e}"
                  f"  ({best['lr']}, {best['damping']}, {best['momentum']}, "
                  f"{best['nc']})  draw {best['draw']}")
            winners.append(dict(
                arm=arm, sys=sys_, lr=best["lr"], damping=best["damping"],
                momentum=best["momentum"], norm_constraint=best["nc"],
                l2_error=f"{best['l2']:.6e}", draw=best["draw"],
                n_usable=len(rs), log=best["log"]))
        print()

    if len(ok) < expected:
        print(f"WARNING: expected {expected} tune tasks, found {len(ok)} usable."
              " Check for tasks that never wrote a log at all.")
    if len(winners) < len(SYSTEMS) * len(TUNE_ARMS):
        print(f"WARNING: only {len(winners)} of {len(SYSTEMS) * len(TUNE_ARMS)}"
              " (arm, system) cells have a winner. E5(c) needs all of them.")
        print("         A cell where every draw diverged is a FINDING -- record"
              " it, do not widen the search until it goes away.")

    if not winners:
        raise SystemExit("no winners to write")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(winners[0]))
        w.writeheader()
        w.writerows(winners)
    print(f"wrote {out}  ({len(winners)} rows)")
    print("\nNOTE: engdw and spring are deliberately absent. They run at"
          " arXiv:2505.12149's\n      published values, never from this csv.")


def harvest_final(logdir: Path, pattern: str, expected: int) -> None:
    ok, _ = load(logdir, pattern, RE_FINAL_HEADER)
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        cells.setdefault((r["arm"], r["sys"]), []).append(r)

    print(f"\n{'system':13s} {'arm':18s} {'n':>2s} {'median l2':>11s} {'min':>11s} "
          f"{'max':>11s} {'s/step':>8s} {'steps':>8s}")
    print("-" * 92)
    for sys_ in SYSTEMS:
        for arm in FINAL_ARMS:
            rs = cells.get((arm, sys_))
            if not rs:
                print(f"{SYS_LABEL[sys_]:13s} {ARM_LABEL[arm]:18s}  -  "
                      f"{'NO USABLE RUNS':>11s}")
                continue
            vals = sorted(r["l2"] for r in rs)
            sps = [r["s_per_step"] for r in rs if r["s_per_step"]]
            steps = [r["final_step"] for r in rs]
            if len(rs) < 3:
                flag = f"  <-- only {len(rs)} seed(s)"
            else:
                flag = ""
            print(f"{SYS_LABEL[sys_]:13s} {ARM_LABEL[arm]:18s} {len(rs):2d} "
                  f"{median(vals):11.4e} {vals[0]:11.4e} {vals[-1]:11.4e} "
                  f"{(median(sps) if sps else float('nan')):8.3f} "
                  f"{median(steps):8.0f}{flag}")
        print()

    if len(ok) < expected:
        print(f"WARNING: expected {expected} final tasks, found {len(ok)} usable.")
    print("READING THE TABLE")
    print("  A gap between two arms that is smaller than that arm's own min-max")
    print("  spread over three seeds is not a result. Say 'indistinguishable at")
    print("  three seeds' and name the system, rather than reporting the median")
    print("  ordering as if it were a win.")


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("tune", "final"), required=True)
    p.add_argument("--logs", default="logs", help="directory of e5*_*.out files")
    p.add_argument("--out", default="runs/e5_best_hparams.csv",
                   help="where to write the winners csv (tune stage only)")
    p.add_argument("--pattern", default=None,
                   help="override the log glob (default: by stage)")
    args = p.parse_args()

    logdir = Path(args.logs)
    if args.stage == "tune":
        harvest_tune(logdir, args.pattern or "e5tune_*.out", Path(args.out),
                     expected=len(SYSTEMS) * len(TUNE_ARMS) * 24)
    else:
        harvest_final(logdir, args.pattern or "e5final_*.out",
                      expected=len(SYSTEMS) * len(FINAL_ARMS) * 3)


if __name__ == "__main__":
    main()
