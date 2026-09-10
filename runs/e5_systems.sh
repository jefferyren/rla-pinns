#!/bin/bash
# =============================================================================
# E5 -- shared definitions: the four benchmark systems, the paper's published
# hyperparameters, and the four arms' command lines.
#
# SOURCED by e5_smoke.sh / e5_tune.sh / e5_final.sh. It exists so the four
# system configs live in ONE place: PINN_EXPERIMENTS.md section 7 records that
# nine ledger rows already went stale from configs drifting between copies.
# Do not inline these anywhere else.
#
# Provenance for every number below: Guzman-Cordero, Dangel, Goldshlager, and
# Zeinhofer, "Improving Energy Natural Gradient Descent through Woodbury,
# Momentum, and Randomization", arXiv:2505.12149. Section references are to
# that paper's Appendix A. The target we are replicating is its Figure 3.
# =============================================================================

# --- The four systems --------------------------------------------------------
#
# D was verified by direct parameter count against the D printed in the paper's
# Figure 3 panel titles -- all four match exactly, so the architectures and the
# input dimensions below are certain:
#
#   p5     5 -> 64 -> 64 -> 48 -> 48 -> 1        D =    10 065   (paper A.2)
#   p100   100 -> 768 -> 768 -> 512 -> 512 -> 1  D = 1 325 057   (paper A.4)
#   heat4  5 -> 256 -> 256 -> 128 -> 128 -> 1    D =   116 865   (paper A.5)
#          (input is 4 spatial + 1 time)
#   lfp9   10 -> 256 -> 256 -> 128 -> 128 -> 1   D =   118 145   (paper A.6)
#          (input is 9 spatial + 1 time)
#
# BUDGET is the paper's stated per-run wall-clock allocation. TUNE_BUDGET is
# 20% of it -- tuning at the full budget would cost 5x and E5 does not have it.
# This is a deviation from the paper and is written up as one.
#
# p100's N_Omega=100 / N_dOmega=50 is NOT in Appendix A.4, which omits the
# batch sizes. It is recovered from two independent sources that agree:
# rla_pinns/exp6_poisson100d_fixedlr/sweeps/*.yaml, and the paper's own
# effective-dimension section, which states "N = 150" for the 100D Poisson
# experiment (100 + 50 = 150).
e5_sys_config() {
  case "$1" in
    p5)
      CFG="--equation=poisson --boundary_condition=cos_sum --dim_Omega=5 \
--model=mlp-tanh-64-64-48-48 --N_Omega=3000 --N_dOmega=500 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"
      BUDGET=7000
      TUNE_BUDGET=1400
      ;;
    p100)
      CFG="--equation=poisson --boundary_condition=u_weinan_norm --dim_Omega=100 \
--model=mlp-tanh-768-768-512-512 --N_Omega=100 --N_dOmega=50 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"
      BUDGET=10000
      TUNE_BUDGET=2000
      ;;
    heat4)
      CFG="--equation=heat --boundary_condition=sin_sum --dim_Omega=4 \
--model=mlp-tanh-256-256-128-128 --N_Omega=3000 --N_dOmega=500 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"
      BUDGET=3000
      TUNE_BUDGET=900
      ;;
    lfp9)
      CFG="--equation=log-fokker-planck-isotropic --boundary_condition=gaussian \
--dim_Omega=9 --model=mlp-tanh-256-256-128-128 --N_Omega=3000 --N_dOmega=1000 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"
      BUDGET=6000
      TUNE_BUDGET=1200
      ;;
    *)
      echo "ERROR: unknown system '$1' (want: p5 p100 heat4 lfp9)" >&2
      return 1
      ;;
  esac
}

E5_SYSTEMS=(p5 p100 heat4 lfp9)
E5_ARMS=(engdw spring primesr ssspring)

# --- The paper's published hyperparameters -----------------------------------
#
# These are the FIXED-LEARNING-RATE winners, which is what Figure 3 plots: its
# legend lists "ENGD (Woodbury)" and "SPRING" separately from "ENGD-W (Line
# Search)", so the first two are the fixed-lr variants from A.x.1, not the
# line-search ones from A.x.
#
# We use these verbatim rather than re-tuning. That is the request ("our engd
# woodbury and spring runs should be the exact same as in this paper") and it
# is also the right call: re-tuning the baselines on our hardware and then
# reporting a win would be a weaker result than winning against the baselines
# exactly as their authors published them.
#
#   p5     paper A.2.1
#   p100   paper A.4.1  -- see the caveat in E5's block in PINN_EXPERIMENTS.md;
#                          A.4.1 misdirects the reader to Figure 12 (the 10d
#                          panel), but its values are the 100d ones. Confirmed
#                          three ways: SPRING damping 3.0116e-2 lies inside the
#                          100d sweep's damping range LU([1e-6, 1e0]) and
#                          outside the 10d one's LU([1e-10, 1e-3]); A.3.1 gives
#                          a different pair for 10d; and this repo's inherited
#                          anchor (lr 0.0924, damping 0.0301) is these numbers.
#   heat4  paper A.5.1  -- LOAD-BEARING CAVEAT, read E5's block before trusting
#                          the heat panel: A.5.1's figure is titled "5d Heat
#                          (D = 117121)", which is a 6-input network, not the
#                          4+1d one (D = 116865) that Figure 3 plots.
#   lfp9   paper A.6.1
e5_paper_hparams() {
  case "$1" in
    p5)
      PAPER_ENGDW_LR=5.2289e-2;    PAPER_ENGDW_DAMPING=6.804474e-8
      PAPER_SPRING_LR=6.3502e-2;   PAPER_SPRING_DAMPING=6.811585e-10
      PAPER_SPRING_MOMENTUM=8.26966e-1
      ;;
    p100)
      PAPER_ENGDW_LR=9.118e-2;     PAPER_ENGDW_DAMPING=6.233e-7
      PAPER_SPRING_LR=9.2362e-2;   PAPER_SPRING_DAMPING=3.0116e-2
      PAPER_SPRING_MOMENTUM=9.8386e-1
      ;;
    heat4)
      PAPER_ENGDW_LR=9.939225e-2;  PAPER_ENGDW_DAMPING=1.139970e-7
      PAPER_SPRING_LR=8.663887e-2; PAPER_SPRING_DAMPING=2.081775e-7
      PAPER_SPRING_MOMENTUM=9.078456e-1
      ;;
    lfp9)
      PAPER_ENGDW_LR=6.029401e-2;  PAPER_ENGDW_DAMPING=8.638985e-4
      PAPER_SPRING_LR=4.473188e-2; PAPER_SPRING_DAMPING=8.377655e-3
      PAPER_SPRING_MOMENTUM=9.760086e-1
      ;;
    *)
      echo "ERROR: unknown system '$1' for paper hyperparameters" >&2
      return 1
      ;;
  esac
}

# --- Arm command lines -------------------------------------------------------
#
# e5_arm_args <arm> <lr> <damping> <momentum> <norm_constraint>  -> sets ARM_ARGS
#
# ENGD-W AND SPRING ARE BOTH --optimizer=RNGD. That is how the paper's own code
# implements them: every ENGD_woodbury*.yaml and SPRING*.yaml in this repo sets
# --optimizer=RNGD --RNGD_approximation=exact, the two differing only in
# --RNGD_momentum. Do NOT substitute --optimizer=SPRING (optim/spring.py): that
# is a different class with a trust region and an adaptive-beta controller, and
# it is not what produced the paper's numbers.
#
# RNGD TAKES NO norm_constraint. optim/rngd.py's parser has no such flag and
# train.py calls check_all_args_parsed(), which RAISES on any leftover argv --
# so passing --RNGD_norm_constraint kills the run at startup. (Several archived
# yamls in this repo pass it and are dead for exactly that reason: see
# exp8_poisson5d_fixedlr/sweeps/SPRING.yaml.) The consequence is a real and
# unavoidable asymmetry: engdw and spring run with NO trust region, primesr and
# ssspring run with one. It is inherent to reproducing the paper's arms as
# published; state it in the write-up rather than hiding it.
e5_arm_args() {
  local arm="$1" lr="$2" damping="$3" momentum="$4" nc="$5"
  case "$arm" in
    engdw)
      ARM_ARGS="--optimizer=RNGD --RNGD_approximation=exact \
--RNGD_lr=${lr} --RNGD_damping=${damping} --RNGD_momentum=0.0"
      ;;
    spring)
      ARM_ARGS="--optimizer=RNGD --RNGD_approximation=exact \
--RNGD_lr=${lr} --RNGD_damping=${damping} --RNGD_momentum=${momentum}"
      ;;
    primesr)
      # PRIME-SR sets its momentum per step from the sampled Gram matrix, so it
      # has no momentum knob to tune -- that is the method, not an omission.
      ARM_ARGS="--optimizer=PRIMESR \
--PRIMESR_lr=${lr} --PRIMESR_damping=${damping} \
--PRIMESR_norm_constraint=${nc} --PRIMESR_print_every=100"
      ;;
    ssspring)
      # Flags follow the rules fixed in PINN_EXPERIMENTS.md section 2: the
      # Unified class only, a numeric lr (never grid_line_search, which bypasses
      # norm_constraint), beta_max pinned, and adaptive_eta + adaptive_probe
      # both on.
      ARM_ARGS="--optimizer=SameSampledSPRINGUnified \
--SameSampledSPRINGUnified_lr=${lr} \
--SameSampledSPRINGUnified_damping=${damping} \
--SameSampledSPRINGUnified_momentum=${momentum} \
--SameSampledSPRINGUnified_lb_window=30 \
--SameSampledSPRINGUnified_probe_lr=${lr} \
--SameSampledSPRINGUnified_probe_damping=${damping} \
--SameSampledSPRINGUnified_norm_constraint=${nc} \
--SameSampledSPRINGUnified_beta_max=0.99 \
--SameSampledSPRINGUnified_probe_seed=0 \
--SameSampledSPRINGUnified_adaptive_eta \
--SameSampledSPRINGUnified_adaptive_probe"
      ;;
    *)
      echo "ERROR: unknown arm '${arm}' (want: engdw spring primesr ssspring)" >&2
      return 1
      ;;
  esac
}

# --- Cluster preamble --------------------------------------------------------
# Same conda env and checkout path as e2_tune.sh / e3_final.sh.
e5_activate() {
  mkdir -p logs
  set +eu
  source ~/.bashrc
  conda activate rla_pinns
  local status=$?
  set -eu
  if [[ $status -ne 0 ]]; then
    echo "ERROR: conda activate rla_pinns failed (status $status)" >&2
    exit 1
  fi
  cd "${E5_REPO}/rla_pinns" || { echo "ERROR: cd ${E5_REPO}/rla_pinns failed" >&2; exit 1; }
}

# --- wandb entity ------------------------------------------------------------
# Never hard-code "rla-pinns": that is the upstream authors' team and a
# non-member gets 403 at wandb.init, killing the job ~11s in.
e5_entity_arg() {
  ENTITY_ARG=""
  if [[ -n "${PINN_WANDB_ENTITY:-}" ]]; then
    ENTITY_ARG="--wandb_entity=${PINN_WANDB_ENTITY}"
  fi
}
