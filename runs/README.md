# PINN experiments on Savio — operational quick reference

Full plan, rationale and kill criteria: `PINN_EXPERIMENTS.md` at the repo root.
This file is just the commands and the gates between them.

**Two claims, nothing else.** C1: Adaptive SPRING fails on PINNs, and not because it was
under-tuned. C2: SS-SPRING beats *properly tuned* fixed-μ SPRING.

**Three arms.** `a1_spring` = tuned fixed μ (`lb_window=0`) · `a2_adaptive` = the method
under test (`lb_window=30`) · `a3_ss` = ours (`SameSampledSPRINGUnified`).

**Two PDEs.** `p100` = Poisson-100d · `lfp9` = log-Fokker-Planck-9d at `N_Omega=300`
(**not** the archived 3000 — see the plan, §2).

| Stage | Script | Tasks | GPU-h | Produces |
|---|---|---|---|---|
| E1 | `e1_smoke.sh` | 7 | 4 | s/step, metric noise floor |
| E2 | `e2_tune.sh` | 78 | 52 | **C1** + `app:hyperparams` |
| E3 | `e3_final.sh` | 18 | 36 | **C2** — Fig A + Tab A |
| E4 | `e4_mechanism.sh` | 18 | 27 | *(optional)* Fig B |

**E5 is a separate, optional programme** — see the section at the bottom. It replicates
arXiv:2505.12149's four-panel figure with PRIME-SR and SS-SPRING swapped in for that
paper's line-search and KFAC arms. It shares no scripts, wandb project or seeds with
E1–E4, and it is the first thing to cut if the allocation tightens.

| Stage | Script | Tasks | Charged h | Produces |
|---|---|---|---|---|
| E5(a) | `e5_submit.sh smoke` | 16 | 7 | that PRIME-SR runs at all; the 11 GiB memory check |
| E5(b) | `e5_submit.sh tune` | 192 | 112 | PRIME-SR + SS-SPRING hyperparameters, 4 systems |
| E5(c) | `e5_submit.sh final` | 48 | 106 | *(optional)* the four-panel head-to-head |

Run everything from the **repo root** (`~/rla-pinns-use`), not from `rla_pinns/`.
The scripts `cd` into the package themselves; `logs/` and `ckpt_noise/` land at the root.

---

## wandb entity

The scripts do **not** pass `--wandb_entity`, so runs land in your personal entity.
Do not hard-code `rla-pinns` — that is the upstream authors' team, and a non-member gets
`403 Forbidden` inside `wandb.init()`, which kills the job ~11 s in, *after* the model is
built but before the first step. To log to a team you actually belong to:

```bash
PINN_WANDB_ENTITY=my-team sbatch runs/e1_smoke.sh
```

## Step 0 — commit first (not optional)

The working tree carries the private-RNG fix for `x_star`. Without it, `a3_ss` and
`a1_spring` at the same `--model_seed` **train on different collocation data**, and E3's
comparison means nothing.

```bash
git add -A && git commit -m "SPRING/SS-SPRING: private probe RNG, trust region, beta_max, data_seed guard" && git rev-parse HEAD
```

Record that SHA next to the results.

## Step 1 — dry-run everything (no cluster, no GPU)

Prints every command line that would be submitted, then exits.

```bash
for i in $(seq 1 7);  do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash runs/e1_smoke.sh; done
```

```bash
for i in $(seq 1 78); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash runs/e2_tune.sh; done
```

## Step 2 — E1

```bash
sbatch runs/e1_smoke.sh
```

```bash
python runs/e1_noise_floor.py --checkpoint_dir ckpt_noise
```

**Gate before spending 52 GPU-h on E2 — check all three:**

1. All 6 smoke tasks reached the end, and wandb shows both `l2_error` **and**
   `decay_factor`.
2. Measured s/step per (arm, PDE). Use it to confirm E2's `--time=00:40:00` really covers
   `--num_seconds=1200`, and to size E3. **Never size `--time` from `--num_seconds` alone**
   — archived 100d runs at `N_total=625` measured 16.4–16.7 s/step.
3. `a1_spring` and `a3_ss` at `model_seed=1` drew the **same** data: identical `Step: 0000000`
   loss at matched `(lr, damping)` on the same PDE. If they differ, Step 0 didn't take.

## Step 3 — E2, then harvest

```bash
sbatch runs/e2_tune.sh
```

```bash
python runs/harvest_e2.py --logs logs --out runs/best_hparams.csv
```

The harvester prints the min/median/max spread over all 13 draws per arm. **That spread is
the C1 evidence — report it, not just the winner.** Divergence is expected at the
small-damping draws; `nan`/`inf` runs are reported as `diverged` and never-logged runs as
`no-metrics`, never dropped silently. A cell with fewer than ~10 usable draws is not an
equal budget any more — say so, or replace the dead draws.

**Gate:** compare `a2_adaptive`'s **best** against `a1_spring`'s **best**. If a2 still
trails on both PDEs, C1 holds. If a2 matches a1 on both, **C1 is dead as stated** — reframe
before running E3 (see the plan, §6).

## Step 4 — E3 (the paper's result)

```bash
sbatch runs/e3_final.sh
```

Reads `runs/best_hparams.csv` and **refuses to start without it**, so it can never quietly
train on hand-typed hyperparameters. To inspect command lines before E2 finishes:

```bash
for i in $(seq 1 18); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 ALLOW_MISSING_CSV=1 bash runs/e3_final.sh; done
```

**Gate:** if SS-SPRING doesn't beat `a1_spring` by more than E1's noise floor on **both**
PDEs, C2 is not established. Name the PDE; don't generalize from one.

## Step 5 — E4 (optional, decide first)

```bash
sbatch runs/e4_mechanism.sh
```

Without E4 the paper says *Adaptive SPRING fails, same-sampling fixes it* — supported by
E2+E3, mechanism as prose. With it you can say *why*. Plot `decay_factor` (β), not just
loss: the loss shows *that* a2 recovers, β shows the controller doing what the paper claims.

---

## Traps these scripts already avoid

Each of these is a real defect found in the existing scripts or YAMLs:

- **Never pass `--data_seed`.** `train.py` raises on any non-zero value — the flag was inert
  and lying. Replicates come from `--model_seed`. This is what kills 30 of
  `run_diagnosis.sh`'s 45 tasks; **that script is dead as written**, superseded by E4.
- **`--N_eval` is pinned explicitly everywhere.** The default is `10 * N_Omega`, so it
  silently tracks the batch. 179 of 180 existing sweep YAMLs have this bug.
- **Numeric `lr` always.** The unified class defaults to `grid_line_search`, which costs 64
  loss evaluations per step *and bypasses `norm_constraint` entirely*.
- **`beta_max` pinned on both classes.** Defaults are asymmetric (SPRING 1.0 uncapped,
  Unified 0.99), so leaving them default confounds the arms.
- **`adaptive_eta` and `adaptive_probe` both on for a3.** Without `adaptive_probe`, `eta_pr`
  stays at `probe_lr` and the probe is not a Kaczmarz++ iteration.
- **Never `--optimizer=SameSampledSPRING`** (non-unified): no trust region at all.
- **`mkdir -p logs` before the `cd`.** SLURM won't create `--output` dirs; an array
  otherwise dies instantly with no log explaining why.
- **`set +eu` around `source ~/.bashrc`.** A stock bashrc opens with
  `[ -z "$PS1" ] && return`, and `PS1` is unbound non-interactively, so under `set -u` the
  job aborts at 00:00:00 elapsed.
- **`python -u`.** `r_hat` and `rho` reach stdout only via `print()`; a buffered stream
  loses them if the job is killed.

## Known, unfixed, needs doing at write-up time

`wandb_utils.py:143-170` `HYPERPARAMETERS` has no keys for SPRING or either same-sampled
class, and `to_tex` does an unguarded lookup at `:210` — so **`app:hyperparams` cannot be
generated for the paper's own methods** until entries are added. While there, fix the
labeller at `:222-226`, which renames RNGD to "SPRING" whenever momentum ≠ 0. That function
is how the existing figures came to mislabel their baseline.

---

# E5 — replicating arXiv:2505.12149's four systems

Full rationale, caveats and kill criteria: `PINN_EXPERIMENTS.md` §4 / E5.

**Four systems**, at that paper's own architectures, batch sizes, boundary conditions and
time budgets. All four D values were verified by direct parameter count against the D in
its figure's panel titles.

| Key | equation | dim | model | D | N_Ω / N_∂Ω | budget |
|---|---|---|---|---|---|---|
| `p5` | poisson | 5 | `64-64-48-48` | 10 065 | 3000 / 500 | 7000 s |
| `p100` | poisson | 100 | `768-768-512-512` | 1 325 057 | 100 / 50 | 10 000 s |
| `heat4` | heat | 4 | `256-256-128-128` | 116 865 | 3000 / 500 | 3000 s |
| `lfp9` | log-fokker-planck-isotropic | 9 | `256-256-128-128` | 118 145 | 3000 / 1000 | 6000 s |

**Four arms.** `engdw` = ENGD (Woodbury) · `spring` = SPRING · `primesr` = PRIME-SR ·
`ssspring` = SS-SPRING (ours). The first two run at the paper's **published fixed-lr
hyperparameters, verbatim** — they are never tuned here and never read the csv. Only
`primesr` and `ssspring` are tuned.

All four system configs, both sets of published hyperparameters, and all four arms' command
lines live in **one place**, `e5_systems.sh`, which the other three scripts source. Do not
inline them anywhere else.

**One system per submission**, because each has a different budget and a shared `--time`
would either truncate `p100` or reserve three hours to run `heat4` for fifty minutes.
`e5_submit.sh` handles that; use it rather than calling `sbatch` directly.

## Step 1 — dry-run (no cluster, no GPU)

```bash
for i in $(seq 1 16); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash runs/e5_smoke.sh; done
```

```bash
for s in p5 p100 heat4 lfp9; do for i in $(seq 1 48); do SLURM_ARRAY_TASK_ID=$i SYS=$s DRY_RUN=1 bash runs/e5_tune.sh; done; done
```

```bash
DRY=1 bash runs/e5_submit.sh final
```

## Step 2 — E5(a) smoke

```bash
bash runs/e5_submit.sh smoke
```

**Gate on all three before spending 112 hours on tuning:**

1. **PRIME-SR takes steps.** `optim/prime_sr.py` has no test and no run anywhere in this
   repo's history. It compiles and is correctly wired into `train.py`'s step dispatch and
   wandb logging, but it has never run. It also cannot be checked off-cluster: the repo
   needs `torch.func` (torch ≥ 2.0).
2. **No OOM on 11 GiB.** The paper used RTX 6000 (24 GiB); `savio3_gpu`'s GTX2080TI has 11.
   `lfp9` is binding — N = 4000 against D = 118 145 in float64 is a **3.8 GiB** Jacobian
   before workspace (`heat4`: 3.3 GiB). An OOM means a smaller batch on that system, which
   is a deviation to write up.
3. **`engdw`/`spring` track the published figure**, especially on `heat4` — the paper's
   fixed-lr heat hyperparameters sit above a figure titled *5d Heat (D = 117121)*, a
   6-input network, not the 4+1d one the four-panel figure plots. If `heat4` looks wrong,
   re-tune those two arms with `exp15_heat4d_fixed`'s ranges and say so in the write-up.

## Step 3 — E5(b) tuning, then harvest

```bash
bash runs/e5_submit.sh tune
```

```bash
python runs/harvest_e5.py --stage tune --logs logs --out runs/e5_best_hparams.csv
```

24 draws × 2 arms × 4 systems at 20% of each budget, `--model_seed=101` (a dedicated tuning
seed, held out from E5(c)'s seeds 1–3). Draw 1 is the paper's own SPRING winner for that
system, so neither arm can be said to have searched only where it liked. Both arms see the
identical `(lr, damping)` list.

**Report the min/median/max spread, not just the winner.** A cell where every draw diverged
is a finding — record it, don't widen the search until it goes away.

## Step 4 — E5(c) final

```bash
bash runs/e5_submit.sh final
```

```bash
python runs/harvest_e5.py --stage final --logs logs
```

Refuses to start without `runs/e5_best_hparams.csv`. Re-derive `e5_submit.sh`'s `--time`
values from E5(a)'s **measured** s/step first — the defaults carry ~20% headroom over
`--num_seconds`, which is not the same as being right.

**Gate:** a gap smaller than that arm's own min–max over three seeds is not a result. Write
"indistinguishable at three seeds" and name the system.

## E5-specific traps

- **ENGD-W and SPRING are both `--optimizer=RNGD`**, differing only in `--RNGD_momentum`.
  That is how the upstream code implements them. `--optimizer=SPRING` (`optim/spring.py`)
  is a *different class* with a trust region and an adaptive-β controller, and is **not**
  what produced the published numbers.
- **`--RNGD_norm_constraint` is fatal.** `optim/rngd.py` has no such flag and
  `check_all_args_parsed()` raises on leftover argv, so the job dies at startup. Several
  archived yamls pass it and are dead for exactly this reason
  (`exp8_poisson5d_fixedlr/sweeps/SPRING.yaml`). Consequence: `engdw`/`spring` have **no**
  trust region while `primesr`/`ssspring` have one. That asymmetry is inherent to running
  the paper's arms as published — state it, don't hide it.
- **PRIME-SR has no momentum knob.** It sets μ per step from the sampled Gram matrix. It
  tunes `(lr, damping, norm_constraint)`.
- **`p100`'s batch sizes are recovered, not quoted.** Appendix A.4 omits them; `100 / 50`
  comes from `exp6_poisson100d_fixedlr` and from the paper's own "N = 150".
- The E1–E4 traps above (numeric `lr`, pinned `--N_eval` and `beta_max`, never
  `--data_seed`, `mkdir -p logs`, `set +eu` around `source ~/.bashrc`, `python -u`) all
  apply here too and are already handled in `e5_systems.sh`.
