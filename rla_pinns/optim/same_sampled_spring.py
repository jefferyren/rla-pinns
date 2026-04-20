"""SPRING with an interleaved same-sampled Kaczmarz++ probe for adaptive momentum.

Mirrors the JAX reference `get_Interleaved_SharedSample_optimizer`:
    - SPRING direction from the sampled Jacobian.
    - Kaczmarz++ probe uses the SAME sampled Jacobian A with a fixed random
      unit-norm target b = A @ x_star to measure a scale-free residual ratio.
    - The probe residual ratio feeds the same 2p-window adaptive-β schedule
      used in SPRING's adaptive-momentum variant, with an upper clamp β_max.
"""

from argparse import ArgumentParser, Namespace
from math import sqrt
from typing import List, Tuple

import torch
from torch import Tensor, arange, cat, cholesky_solve, randn_like, zeros, zeros_like
from torch.linalg import cholesky
from torch.nn import Module
from torch.optim import Optimizer

from rla_pinns import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from rla_pinns.parse_utils import parse_known_args_and_remove_from_argv
from rla_pinns.optim.utils import (
    evaluate_losses_with_layer_inputs_and_grad_outputs,
    apply_joint_J,
    apply_joint_JT,
    compute_joint_JJT,
)
from rla_pinns.pinn_utils import evaluate_boundary_loss
from rla_pinns.optim.line_search import (
    grid_line_search,
    parse_grid_line_search_args,
)


def parse_SameSampledSPRING_args(
    verbose: bool = False, prefix: str = "SameSampledSPRING_"
) -> Namespace:
    """Parse command-line arguments for `SameSampledSPRING`."""
    parser = ArgumentParser(
        description="Parse arguments for setting up SameSampledSPRING."
    )

    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate (float) or line-search strategy.",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping λ in (JJ^T + λI)^{-1} for the SPRING normal equation.",
        default=1e-3,
    )
    parser.add_argument(
        f"--{prefix}momentum",
        type=float,
        help="Initial β (decay factor) before the adaptive schedule kicks in.",
        default=0.9,
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=SameSampledSPRING.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    parser.add_argument(
        f"--{prefix}lb_window",
        type=int,
        help="Lookback window p for adaptive β. Buffer size is 2p.",
        default=30,
    )
    parser.add_argument(
        f"--{prefix}probe_lr",
        type=float,
        help="Momentum step size η_probe for the Kaczmarz++ probe. "
        "Defaults to `lr` when `lr` is numeric.",
        default=None,
    )
    parser.add_argument(
        f"--{prefix}probe_damping",
        type=float,
        help="Damping for the probe's normal equation. Defaults to `damping`.",
        default=None,
    )
    parser.add_argument(
        f"--{prefix}beta_max",
        type=float,
        help="Upper clamp on adaptive β.",
        default=0.99,
    )

    args = parse_known_args_and_remove_from_argv(parser)

    lr_key = f"{prefix}lr"
    lr_val = getattr(args, lr_key)
    if any(ch.isdigit() for ch in str(lr_val)):
        setattr(args, lr_key, float(lr_val))

    if getattr(args, lr_key) == "grid_line_search":
        grid = parse_grid_line_search_args(verbose=verbose)
        setattr(args, lr_key, (getattr(args, lr_key), grid))

    if verbose:
        print("Parsed arguments for SameSampledSPRING: ", args)

    return args


class SameSampledSPRING(Optimizer):
    """SPRING with an interleaved same-sampled Kaczmarz++ probe driving adaptive β."""

    LOSS_EVALUATORS = {
        "poisson": {
            "interior": poisson_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "heat": {
            "interior": heat_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "fokker-planck-isotropic": {
            "interior": fokker_planck_isotropic_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "log-fokker-planck-isotropic": {
            "interior": log_fokker_planck_isotropic_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
    }
    SUPPORTED_EQUATIONS = list(LOSS_EVALUATORS.keys())

    def __init__(
        self,
        layers: List[Module],
        lr,
        damping: float = 1e-3,
        momentum: float = 0.9,
        equation: str = "poisson",
        lb_window: int = 30,
        probe_lr: float = None,
        probe_damping: float = None,
        beta_max: float = 0.99,
    ):
        defaults = dict(lr=lr, damping=damping, decay_factor=momentum)
        params = sum((list(layer.parameters()) for layer in layers), [])
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "SameSampledSPRING does not support per-parameter options."
            )
        if equation not in self.SUPPORTED_EQUATIONS:
            raise ValueError(
                f"Equation {equation!r} not supported."
                f" Supported are: {self.SUPPORTED_EQUATIONS}."
            )
        if lb_window <= 0:
            raise ValueError("lb_window must be a positive integer.")

        self.equation = equation
        self.layers = layers
        self.steps = 0
        self.p = int(lb_window)
        self._beta_max = float(beta_max)

        # probe hyperparams
        if probe_lr is None:
            if isinstance(lr, float):
                probe_lr = float(lr)
            else:
                raise ValueError(
                    "probe_lr must be provided when lr is not numeric "
                    "(e.g. when using grid_line_search)."
                )
        self._probe_lr = float(probe_lr)
        self._probe_damping = float(
            damping if probe_damping is None else probe_damping
        )

        # per-parameter state: phi, probe iterate z, probe momentum, fixed x_star
        (group,) = self.param_groups
        for p in group["params"]:
            self.state[p]["phi"] = zeros_like(p)
            self.state[p]["z_probe"] = zeros_like(p)
            self.state[p]["mtm_probe"] = zeros_like(p)
            self.state[p]["x_star"] = randn_like(p)

        # globally normalize x_star to unit norm (matches JAX's flat-vector init)
        total = sum(
            (self.state[p]["x_star"] ** 2).sum() for p in group["params"]
        ).sqrt()
        for p in group["params"]:
            self.state[p]["x_star"].div_(total + 1e-12)

        # adaptive-β probe-residual buffer
        p0 = group["params"][0]
        dev, dt = p0.device, p0.dtype
        self._res_buffer = zeros(2 * self.p, device=dev, dtype=dt)
        self._buf_idx = 0
        self._r_hat = torch.tensor(1.0, device=dev, dtype=dt)
        self._checkpoint_idx = 1

        group["decay_factor"] = float(momentum)

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Take a SPRING + Kaczmarz++-probe step.

        Returns:
            (interior_loss, boundary_loss) evaluated before the parameter update.
        """
        (group,) = self.param_groups
        params = group["params"]
        lr = group["lr"]
        damping = group["damping"]
        decay_factor = group["decay_factor"]

        step_idx = self.steps  # matches JAX's step_idx (0-indexed)

        # forward + capture layer inputs / grad outputs for joint J
        (
            interior_loss,
            boundary_loss,
            interior_residual,
            boundary_residual,
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
            self.layers, X_Omega, y_Omega, X_dOmega, y_dOmega, self.equation
        )

        # JJ^T (√N-normalized) without damping — cloned per variant below
        OOT_raw = compute_joint_JJT(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        ).detach()
        N_total = OOT_raw.shape[0]
        diag_idx = arange(N_total, device=OOT_raw.device)

        # SPRING: OOT + λI
        OOT = OOT_raw.clone()
        OOT[diag_idx, diag_idx] = OOT.diag() + damping
        L_spring = cholesky(OOT)

        # √N-normalized concatenated residual (negated, following SPRING convention)
        N_Omega = X_Omega.shape[0]
        N_dOmega = X_dOmega.shape[0]
        interior_r = interior_residual.detach() / sqrt(N_Omega)
        boundary_r = boundary_residual.detach() / sqrt(N_dOmega)
        epsilon = -cat([interior_r, boundary_r]).flatten()

        # zeta = epsilon − β · (O · φ)
        O_phi = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["phi"].unsqueeze(-1) for p in params],
        ).squeeze(-1)
        zeta = epsilon - decay_factor * O_phi

        # step = J^T (JJ^T + λI)^{-1} zeta
        step_ds = cholesky_solve(zeta.unsqueeze(-1), L_spring)
        step_ps = [
            s.squeeze(-1)
            for s in apply_joint_JT(
                interior_inputs,
                interior_grad_outputs,
                boundary_inputs,
                boundary_grad_outputs,
                step_ds,
            )
        ]

        # φ ← β · φ + step
        for p, s in zip(params, step_ps):
            self.state[p]["phi"].mul_(decay_factor).add_(s)

        # parameter update — JAX does plain `params += η · φ`; keep line-search option
        if isinstance(lr, float):
            for p in params:
                p.data.add_(self.state[p]["phi"], alpha=lr)
        else:
            if lr[0] == "grid_line_search":
                directions = [self.state[p]["phi"] for p in params]

                def f() -> Tensor:
                    interior_loss_ls = self._eval_loss(X_Omega, y_Omega, "interior")
                    boundary_loss_ls = self._eval_loss(X_dOmega, y_dOmega, "boundary")
                    return interior_loss_ls + boundary_loss_ls

                grid = lr[1]
                grid_line_search(f, params, directions, grid)

        # --- Kaczmarz++ probe on the SAME sampled Jacobian ---
        probe_damping = self._probe_damping
        if probe_damping == damping:
            L_probe = L_spring
        else:
            OOT_probe = OOT_raw.clone()
            OOT_probe[diag_idx, diag_idx] = OOT_probe.diag() + probe_damping
            L_probe = cholesky(OOT_probe)

        # b = A · x_star  (A = √N-normalized joint Jacobian)
        b_tau = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["x_star"].unsqueeze(-1) for p in params],
        ).squeeze(-1)

        # A · z_probe  (pre-update)
        Az_old = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["z_probe"].unsqueeze(-1) for p in params],
        ).squeeze(-1)

        # v = (AA^T + λ_p I)^{-1} (A z − b)
        v = cholesky_solve((Az_old - b_tau).unsqueeze(-1), L_probe)

        # w = A^T v  (per-parameter)
        w_list = [
            s.squeeze(-1)
            for s in apply_joint_JT(
                interior_inputs,
                interior_grad_outputs,
                boundary_inputs,
                boundary_grad_outputs,
                v,
            )
        ]

        # mtm ← β · (mtm − w);   z ← z − w + η_probe · mtm
        probe_eta = self._probe_lr
        for p_obj, w in zip(params, w_list):
            s = self.state[p_obj]
            s["mtm_probe"].sub_(w).mul_(decay_factor)
            s["z_probe"].sub_(w).add_(s["mtm_probe"], alpha=probe_eta)

        # probe residual norm on the UPDATED z
        Az_new = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["z_probe"].unsqueeze(-1) for p in params],
        ).squeeze(-1)
        num = (Az_new - b_tau).norm()
        den = b_tau.norm() + 1e-12
        probe_res_norm = num / den
        one = torch.tensor(
            1.0, device=probe_res_norm.device, dtype=probe_res_norm.dtype
        )
        probe_res_norm = torch.where(
            torch.isfinite(probe_res_norm), probe_res_norm, one
        )

        # ring-buffer write (mirrors JAX `buffer.at[buffer_index % 2p].set(...)`)
        with torch.no_grad():
            self._res_buffer[self._buf_idx % (2 * self.p)] = probe_res_norm.detach()
        self._buf_idx += 1

        # β update — uses step_idx *before* self.steps is incremented (matches JAX)
        self._maybe_update_beta(step_idx)

        self.steps += 1
        return interior_loss, boundary_loss

    def _maybe_update_beta(self, step_idx: int) -> None:
        if not (step_idx % self.p == 0 and step_idx >= 2 * self.p):
            return

        with torch.no_grad():
            (group,) = self.param_groups
            p = self.p
            two_p = 2 * p
            buf_idx = self._buf_idx
            dev = self._res_buffer.device
            dt = self._res_buffer.dtype

            idxs_t = (buf_idx - torch.arange(1, p + 1, device=dev)) % two_p
            idxs_tp = (
                buf_idx - torch.arange(p + 1, 2 * p + 1, device=dev)
            ) % two_p

            eps_t = (self._res_buffer[idxs_t] ** 2).sum()
            eps_tp = (self._res_buffer[idxs_tp] ** 2).sum()
            r_ip = eps_t / (eps_tp + 1e-12)

            n_old_f = torch.tensor(float(self._checkpoint_idx), device=dev, dtype=dt)
            n_new_f = n_old_f + 1.0
            # JAX does n^{log n}; log(1)=0 → a_old = 1, matches exactly.
            a_old = torch.pow(n_old_f, torch.log(n_old_f))
            a_new = torch.pow(n_new_f, torch.log(n_new_f))
            alph = a_old / a_new

            one = torch.tensor(1.0, device=dev, dtype=dt)
            self._r_hat = alph * self._r_hat + (1.0 - alph) * torch.minimum(one, r_ip)

            rho = torch.clamp(
                1.0 - torch.pow(self._r_hat, 1.0 / float(p)), min=0.0
            )
            beta_new = (1.0 - rho) / (1.0 + rho)
            beta_new = torch.clamp(beta_new, 0.0, self._beta_max)

            group["decay_factor"] = float(beta_new.item())
            self._checkpoint_idx += 1

            print(
                f"SameSampledSPRING β update @ step {self.steps}: "
                f"β={beta_new.item():.6f}, r_hat={self._r_hat.item():.6f}, "
                f"ρ={rho.item():.6f}"
            )

    def _eval_loss(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        loss_evaluator = self.LOSS_EVALUATORS[self.equation][loss_type]
        loss, _, _ = loss_evaluator(self.layers, X, y)
        return loss
