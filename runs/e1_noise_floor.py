#!/usr/bin/env python3
"""Measure the Monte-Carlo noise floor of `l2_error` from an E1 checkpoint.

WHY THIS EXISTS
    `l2_error` is not a deterministic quantity. pinn_utils.l2_error computes
    sqrt(mean((u_theta - u)^2)) over N_eval points drawn at random from Omega, so
    the reported number carries its own sampling error -- and nobody has measured
    it. That standard error is the MINIMUM REPORTABLE DIFFERENCE for Tab A: if
    E3's SS-SPRING vs tuned-SPRING gap lands inside it, C2 is not established at
    that PDE, no matter how clean the curves look.

    Costs about 1 GPU-h and can invalidate a headline claim. Run it before E3.

WHAT IT DOES
    Holds theta FIXED (one checkpoint), then redraws the evaluation set `repeats`
    times at each N_eval and reports the spread. All variation is metric noise:
    the model never changes.

USAGE
    # from the repo root (~/rla-pinns-use), after E1 task 7 has run
    python runs/e1_noise_floor.py --checkpoint_dir ckpt_noise
    python runs/e1_noise_floor.py --checkpoint_dir ckpt_noise \
        --N_eval 1000 2000 5000 30000 --repeats 20
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import device, load, manual_seed, no_grad
from torch.nn import Sequential

from rla_pinns.pinn_utils import l2_error
from rla_pinns.train import SOLUTIONS, create_data_loader, set_up_layers


def resolve_dtype(raw) -> torch.dtype:
    """config['dtype'] may round-trip as a torch.dtype or as its string name."""
    if isinstance(raw, torch.dtype):
        return raw
    return {"float32": torch.float32, "float64": torch.float64,
            "torch.float32": torch.float32, "torch.float64": torch.float64}[str(raw)]


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_dir", default="ckpt_noise")
    p.add_argument("--checkpoint", default=None,
                   help="specific .pt file; default is the newest in the dir")
    p.add_argument("--N_eval", type=int, nargs="+",
                   default=[1000, 2000, 5000, 30000])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed0", type=int, default=9000,
                   help="base seed; draw r at size N uses seed0 + 1000*i + r")
    args = p.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        cands = sorted(Path(args.checkpoint_dir).glob("*.pt"))
        if not cands:
            raise SystemExit(
                f"no .pt files in {args.checkpoint_dir}. Run E1 task 7 first:\n"
                f"  SLURM_ARRAY_TASK_ID=7 sbatch runs/e1_smoke.sh")
        ckpt_path = cands[-1]

    dev = device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load(ckpt_path, map_location=dev)
    cfg = ckpt["config"]
    equation = cfg["equation"]
    condition = cfg["boundary_condition"]
    dim_Omega = int(cfg["dim_Omega"])
    dt = resolve_dtype(cfg["dtype"])

    print(f"checkpoint : {ckpt_path}")
    print(f"step       : {ckpt.get('step')}")
    print(f"problem    : {equation} / {condition} / dim_Omega={dim_Omega}")
    print(f"model      : {cfg['model']}  dtype={dt}  device={dev}")
    print(f"optimizer  : {cfg.get('optimizer')}")
    print(f"N_eval used during training: {cfg.get('N_eval')}")

    # Rebuild theta exactly as train.py built it, then freeze it.
    # Built exactly as train.py:614-616 does it, so the state_dict loads and the
    # dtype/device placement matches the run that produced the checkpoint.
    layers = set_up_layers(cfg["model"], equation, dim_Omega)
    layers = [layer.to(dev, dt) for layer in layers]
    model = Sequential(*layers).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    u = SOLUTIONS[equation][condition]

    print(f"\nredrawing the evaluation set {args.repeats}x per size "
          f"(theta held fixed, so all spread is metric noise)\n")
    header = (f"{'N_eval':>8s} {'mean l2':>12s} {'std':>12s} {'SE':>12s} "
              f"{'SE/mean':>9s} {'min':>12s} {'max':>12s}")
    print(header)
    print("-" * len(header))

    rows = []
    for i, N in enumerate(args.N_eval):
        vals = []
        for r in range(args.repeats):
            # Reseed before building the loader: the draw happens on the global
            # RNG, so a fresh seed per repeat is what makes the sets independent.
            manual_seed(args.seed0 + 1000 * i + r)
            loader = iter(create_data_loader(
                0, "interior", equation, condition, dim_Omega, N, dev, dt))
            X, _ = next(loader)
            with no_grad():
                vals.append(float(l2_error(model, X, u)))
        m, sd = mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)
        se = sd / (len(vals) ** 0.5)
        rows.append((N, m, sd, se))
        print(f"{N:8d} {m:12.5e} {sd:12.5e} {se:12.5e} {se / m:8.2%} "
              f"{min(vals):12.5e} {max(vals):12.5e}")

    # The number that matters downstream.
    biggest = max(args.N_eval)
    row = next(r for r in rows if r[0] == biggest)
    _, m, sd, se = row
    print(f"\n--- for PINN_EXPERIMENTS.md E1(c) ---")
    print(f"metric SE at N_eval={biggest}: {se:.4e}  ({se / m:.2%} of the mean)")
    print(f"minimum reportable gap for Tab A (2*SE, ~95% two-sided): {2 * se:.4e}")
    print(f"i.e. treat two arms as indistinguishable at {equation}/{dim_Omega}d")
    print(f"unless their final l2_error differs by more than {2 * se:.4e}.")
    print("\nNOTE: this is the noise of the METRIC at fixed theta. It is a floor,")
    print("not the whole story -- seed-to-seed spread in E3 is a separate and")
    print("normally larger source of variation, which is why Tab A needs 3 seeds.")


if __name__ == "__main__":
    main()
