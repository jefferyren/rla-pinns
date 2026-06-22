"""Same-Sampled SPRING with an interleaved same-sampled SPRING probe.

PyTorch port of the standalone reference
``interleaved_shared_sample_spring_probe_unified``:

    - SPRING direction from the sampled (joint) Jacobian, exactly as in
      :class:`SameSampledSPRING`.
    - A *SPRING* probe (rather than a Kaczmarz++ probe) runs on the SAME sampled
      Jacobian ``A`` with a fixed random unit-norm target ``b = A @ x_star``. It
      keeps its own SPRING momentum ``phi_probe`` and iterate ``z_probe``.
    - The probe residual ratio feeds the same 2p-window adaptive-β schedule used
      in the standalone reference (Appendix D).

Step sizes (matching the reference):

    - ``adaptive_eta=False`` : the main iterate uses the constant base step ``eta``
      (== ``interleaved_shared_sample_spring_probe``).
    - ``adaptive_eta=True``  : the main iterate uses
      ``eta_main = 1 - beta * (1 - eta)`` (== the ``adaptive_eta`` variant).

``eta`` is the base step size (here exposed as ``lr``). By default only the main
iterate's eta adapts; ``adaptive_probe=True`` makes the probe step adaptive too.
"""

from argparse import ArgumentParser, Namespace
from math import sqrt
from typing import List, Tuple

import torch
from torch import Tensor, arange, cat, cholesky_solve, randn_like, zeros_like
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


def parse_SameSampledSPRINGUnified_args(
    verbose: bool = False, prefix: str = "SameSampledSPRINGUnified_"
) -> Namespace:
    """Parse command-line arguments for `SameSampledSPRINGUnified`."""
    parser = ArgumentParser(
        description="Parse arguments for setting up SameSampledSPRINGUnified."
    )

    parser.add_argument(
        f"--{prefix}lr",
        help="Base step size η (float) or line-search strategy. With "
        "`adaptive_eta`/`adaptive_probe` a numeric value is required.",
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
        choices=SameSampledSPRINGUnified.SUPPORTED_EQUATIONS,
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
        help="Base step size η_probe for the SPRING probe. "
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
        f"--{prefix}adaptive_eta",
        action="store_true",
        help="Use the adaptive main step η_main = 1 - β·(1 - η) (Appendix D).",
        default=False,
    )
    parser.add_argument(
        f"--{prefix}adaptive_probe",
        action="store_true",
        help="Make the probe step adaptive as well (uses η_main).",
        default=False,
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
        print("Parsed arguments for SameSampledSPRINGUnified: ", args)

    return args


class SameSampledSPRINGUnified(Optimizer):
    """Same-Sampled SPRING with an interleaved same-sampled SPRING probe.

    Faithful port of ``interleaved_shared_sample_spring_probe_unified``.
    """

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
        adaptive_eta: bool = False,
        adaptive_probe: bool = False,
    ):
        defaults = dict(lr=lr, damping=damping, decay_factor=momentum)
        params = sum((list(layer.parameters()) for layer in layers), [])
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "SameSampledSPRINGUnified does not support per-parameter options."
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

        self._adaptive_eta = bool(adaptive_eta)
        self._adaptive_probe = bool(adaptive_probe)

        # The adaptive step sizes need a numeric base η to compute
        # η_main = 1 - β·(1 - η); they are incompatible with line search.
        if (self._adaptive_eta or self._adaptive_probe) and not isinstance(lr, float):
            raise ValueError(
                "adaptive_eta/adaptive_probe require a numeric `lr` (base η); "
                "they cannot be combined with grid_line_search."
            )

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

        # per-parameter state: phi, probe iterate z, probe momentum phi_probe,
        # fixed x_star
        (group,) = self.param_groups
        for p in group["params"]:
            self.state[p]["phi"] = zeros_like(p)
            self.state[p]["z_probe"] = zeros_like(p)
            self.state[p]["phi_probe"] = zeros_like(p)
            self.state[p]["x_star"] = randn_like(p)

        # globally normalize x_star to unit norm (matches the standalone
        # flat-vector init: x_star = x_star / ||x_star||)
        total = sum(
            (self.state[p]["x_star"] ** 2).sum() for p in group["params"]
        ).sqrt()
        for p in group["params"]:
            self.state[p]["x_star"].div_(total)

        # adaptive-beta probe-residual log (plain growing list; sliced
        # [t-p+1:t+1] / [t-2p+1:t-p+1] in _maybe_update_beta to match the
        # standalone reference's windows, which include the current step).
        p0 = group["params"][0]
        dev, dt = p0.device, p0.dtype
        self._probe_residuals: List[Tensor] = []
        self._r_hat = torch.tensor(1.0, device=dev, dtype=dt)
        self._checkpoint_idx = 1

        group["decay_factor"] = float(momentum)

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Take a SPRING + SPRING-probe step.

        Returns:
            (interior_loss, boundary_loss) evaluated before the parameter update.
        """
        (group,) = self.param_groups
        params = group["params"]
        lr = group["lr"]
        damping = group["damping"]
        decay_factor = group["decay_factor"]

        step_idx = self.steps

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

        for p, s in zip(params, step_ps):
            self.state[p]["phi"].mul_(decay_factor).add_(s)

        # ---- main step size η_main (Appendix D adaptive variant) ----
        eta_main = None
        if isinstance(lr, float):
            eta_main = (
                1.0 - decay_factor * (1.0 - lr) if self._adaptive_eta else lr
            )
            for p in params:
                p.data.add_(self.state[p]["phi"], alpha=eta_main)
        else:
            if lr[0] == "grid_line_search":
                directions = [self.state[p]["phi"] for p in params]

                def f() -> Tensor:
                    interior_loss_ls = self._eval_loss(X_Omega, y_Omega, "interior")
                    boundary_loss_ls = self._eval_loss(X_dOmega, y_dOmega, "boundary")
                    return interior_loss_ls + boundary_loss_ls

                grid = lr[1]
                grid_line_search(f, params, directions, grid)

        # --- SPRING probe on the SAME sampled Jacobian, synthetic target ---
        probe_damping = self._probe_damping
        if probe_damping == damping:
            L_probe = L_spring
        else:
            OOT_probe = OOT_raw.clone()
            OOT_probe[diag_idx, diag_idx] = OOT_probe.diag() + probe_damping
            L_probe = cholesky(OOT_probe)

        # probe step size η_pr (uses η_main when adaptive_probe, else base η_probe)
        eta_pr = eta_main if self._adaptive_probe else self._probe_lr

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

        # A · phi_probe  (probe SPRING momentum projected through A)
        O_phi_probe = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["phi_probe"].unsqueeze(-1) for p in params],
        ).squeeze(-1)

        # r_probe = b - A z_probe ;  zeta_probe = r_probe - β (A phi_probe)
        r_probe = b_tau - Az_old
        zeta_probe = r_probe - decay_factor * O_phi_probe

        # phi_probe <- J^T (JJ^T + λ_p I)^{-1} zeta_probe + β phi_probe
        v = cholesky_solve(zeta_probe.unsqueeze(-1), L_probe)
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
        for p_obj, w in zip(params, w_list):
            s = self.state[p_obj]
            s["phi_probe"].mul_(decay_factor).add_(w)
            s["z_probe"].add_(s["phi_probe"], alpha=eta_pr)

        # probe residual norm on the UPDATED z_probe: ||A z_probe - b||
        Az_new = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["z_probe"].unsqueeze(-1) for p in params],
        ).squeeze(-1)
        probe_res_norm = (Az_new - b_tau).norm()

        # append to the probe-residual log (no ring overwrite)
        self._probe_residuals.append(probe_res_norm.detach())

        # β update — uses step_idx *before* self.steps is incremented
        self._maybe_update_beta(step_idx)

        self.steps += 1
        return interior_loss, boundary_loss

    def _maybe_update_beta(self, step_idx: int) -> None:
        if not (step_idx % self.p == 0 and step_idx >= 2 * self.p):
            return

        with torch.no_grad():
            (group,) = self.param_groups
            p = self.p
            dev = self._r_hat.device
            dt = self._r_hat.dtype

            # Windows including the current step, matching the standalone
            # reference: residuals[t-p+1:t+1] / residuals[t-2p+1:t-p+1].
            window_t = torch.stack(
                self._probe_residuals[step_idx - p + 1 : step_idx + 1]
            )
            window_tp = torch.stack(
                self._probe_residuals[step_idx - 2 * p + 1 : step_idx - p + 1]
            )
            eps_t = (window_t ** 2).sum()
            eps_tp = (window_tp ** 2).sum()
            r_ip = eps_t / eps_tp

            n_old_f = torch.tensor(float(self._checkpoint_idx), device=dev, dtype=dt)
            n_new_f = n_old_f + 1.0

            a_old = torch.pow(n_old_f, torch.log(n_old_f))
            a_new = torch.pow(n_new_f, torch.log(n_new_f))
            alph = a_old / a_new

            one = torch.tensor(1.0, device=dev, dtype=dt)
            self._r_hat = alph * self._r_hat + (1.0 - alph) * torch.minimum(one, r_ip)

            rho = torch.clamp(
                1.0 - torch.pow(self._r_hat, 1.0 / float(p)), min=0.0
            )
            beta_new = (1.0 - rho) / (1.0 + rho)

            group["decay_factor"] = float(beta_new.item())
            self._checkpoint_idx += 1

            print(
                f"SameSampledSPRINGUnified β update @ step {self.steps}: "
                f"β={beta_new.item():.6f}, r_hat={self._r_hat.item():.6f}, "
                f"ρ={rho.item():.6f}"
            )

    def _eval_loss(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        loss_evaluator = self.LOSS_EVALUATORS[self.equation][loss_type]
        loss, _, _ = loss_evaluator(self.layers, X, y)
        return loss
