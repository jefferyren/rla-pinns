"""PRIME-SR (Principal Range Informed MomEntum SR) optimizer for PINNs.

Adaptive-momentum SR variant from "Momentum Stability and Adaptive Control in
Stochastic Reconfiguration" (Wang & Liu, 2026, arXiv:2604.18357), Algorithm 2.

The momentum factor mu_k is set per-step from two indicators of the sampled
SR Gram matrix T_k = O_k^T O_k:
  * effective spectral dimension alpha_k = (sum s^2)^2 / sum s^4
  * subspace overlap beta_tilde_k = || V_{k,a}^T V_{k-1,a} ||_F
where V_{k,a} are the top ceil(alpha_k) eigenvectors of T_k.
"""

from argparse import ArgumentParser, Namespace
from math import sqrt
from typing import List, Tuple

import torch
from torch import Tensor, cat, zeros_like
from torch.linalg import eigh
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


def parse_PRIMESR_args(verbose: bool = False, prefix: str = "PRIMESR_") -> Namespace:
    """Parse command-line arguments for `PRIMESR`."""
    parser = ArgumentParser(description="Parse arguments for setting up PRIME-SR.")

    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate or line search strategy for the optimizer.",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping factor lambda in (T_k + lambda I).",
        default=1e-3,
    )
    parser.add_argument(
        f"--{prefix}norm_constraint",
        type=float,
        help="Norm constraint C on the natural gradient step (paper Eq. 2.15).",
        default=1e-3,
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=PRIMESR.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    parser.add_argument(
        f"--{prefix}print_every",
        type=int,
        help="Print PRIME-SR diagnostics every N steps (0 = silent).",
        default=100,
    )

    args = parse_known_args_and_remove_from_argv(parser)

    lr_key = f"{prefix}lr"
    if any(ch.isdigit() for ch in str(getattr(args, lr_key))):
        setattr(args, lr_key, float(getattr(args, lr_key)))

    if getattr(args, lr_key) == "grid_line_search":
        grid = parse_grid_line_search_args(verbose=verbose)
        setattr(args, lr_key, (getattr(args, lr_key), grid))

    if verbose:
        print("Parsed arguments for PRIME-SR: ", args)

    return args


class PRIMESR(Optimizer):
    """PRIME-SR optimizer for PINN problems.

    See arXiv:2604.18357 (Wang & Liu, 2026), Algorithm 2.
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
        norm_constraint: float = 1e-3,
        equation: str = "poisson",
        print_every: int = 100,
    ):
        defaults = dict(lr=lr, damping=damping, norm_constraint=norm_constraint)
        params = sum((list(layer.parameters()) for layer in layers), [])
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("PRIMESR does not support per-parameter options.")
        if equation not in self.SUPPORTED_EQUATIONS:
            raise ValueError(
                f"Equation {equation!r} not supported."
                f" Supported are: {self.SUPPORTED_EQUATIONS}."
            )

        self.equation = equation
        self.layers = layers
        self.steps = 0
        self._print_every = int(print_every)

        (group,) = self.param_groups
        for p in group["params"]:
            self.state[p]["phi"] = zeros_like(p)

        # Cached principal range subspace V^R_{k-1, alpha} and its column count
        self._V_prev_alpha = None
        self._alpha_prev_ceil = None

        # Most recent diagnostics, exposed for external logging.
        self._last_mu = 0.0
        self._last_alpha = 0.0
        self._last_alpha_ceil = 0
        self._last_rank = 0
        self._last_beta_tilde = 0.0

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Take a PRIME-SR step.

        Args:
            X_Omega: Input for the interior loss.
            y_Omega: Target for the interior loss.
            X_dOmega: Input for the boundary loss.
            y_dOmega: Target for the boundary loss.

        Returns:
            Tuple of the interior and boundary loss before the step.
        """
        (group,) = self.param_groups
        params = group["params"]
        lr = group["lr"]
        damping = group["damping"]
        norm_constraint = group["norm_constraint"]

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

        # Sample-space Gram matrix T_k = O_k^T O_k  (paper Eq. 4.1)
        OOT_raw = compute_joint_JJT(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        ).detach()

        # Symmetrize to suppress floating-point asymmetry before eigh.
        OOT_raw = 0.5 * (OOT_raw + OOT_raw.T)

        # Eigendecomposition T_k = V diag(s^2) V^T; eigh returns ascending order.
        s2, V = eigh(OOT_raw)
        s2 = s2.flip(0).clamp_min(0.0)
        V = V.flip(1)

        # Numerical rank with MATLAB rank()'s default tolerance: N * eps * ||T_k||_2.
        # The paper specifies eps_r = N_s * eps_machine, applied as the rank tolerance.
        N_total = OOT_raw.shape[0]
        eps_m = torch.finfo(OOT_raw.dtype).eps
        max_s2 = s2[0]
        tol = float(N_total) * eps_m * max_s2
        r_k = int((s2 > tol).sum().item())

        use_adaptive = r_k > 0
        if use_adaptive:
            top_s2 = s2[:r_k]
            sum_s2 = top_s2.sum()
            sum_s4 = (top_s2 * top_s2).sum()
            alpha_k = (sum_s2 * sum_s2) / sum_s4  # Eq. 4.7
            alpha_ceil = max(1, min(int(torch.ceil(alpha_k).item()), r_k))
            V_alpha = V[:, :alpha_ceil]
        else:
            alpha_k = torch.zeros((), device=OOT_raw.device, dtype=OOT_raw.dtype)
            alpha_ceil = 0

        # Adaptive momentum mu_k via Eqs. 4.4 and 4.5.
        if use_adaptive and self._V_prev_alpha is not None:
            overlap = V_alpha.T @ self._V_prev_alpha
            beta_tilde = torch.linalg.matrix_norm(overlap, ord="fro")
            min_alpha = max(1, min(alpha_ceil, self._alpha_prev_ceil))
            upper = sqrt(float(min_alpha))
            ratio_b = (beta_tilde / upper).clamp(0.0, 1.0)
            inner1 = 1.0 - ratio_b.sqrt()
            ratio_a = (alpha_k / float(r_k)).clamp(0.0, 1.0)
            inner2 = 1.0 - ratio_a.pow(0.25)
            mu = (1.0 - inner1 * inner2).clamp(0.0, 1.0)
            mu_val = float(mu.item())
            beta_val = float(beta_tilde.item())
        else:
            # First step (or rank-deficient previous step): Delta theta_{-1} = 0,
            # so mu does not contribute to the update.
            mu_val = 0.0
            beta_val = 0.0

        # sqrt(N)-normalized residual (matches SPRING convention).
        N_Omega = X_Omega.shape[0]
        N_dOmega = X_dOmega.shape[0]
        interior_r = interior_residual.detach() / sqrt(N_Omega)
        boundary_r = boundary_residual.detach() / sqrt(N_dOmega)
        epsilon = -cat([interior_r, boundary_r]).flatten()

        # zeta_k = epsilon - mu_k * O_k phi
        O_phi = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["phi"].unsqueeze(-1) for p in params],
        ).squeeze(-1)
        zeta = epsilon - mu_val * O_phi

        # Solve (T_k + lambda I) y = zeta via the eigendecomp:
        # y = V diag(1 / (s^2 + lambda)) V^T zeta. Mathematically identical to the
        # Cholesky path SPRING uses; reusing the eigendecomp avoids a second factorization.
        Vt_zeta = V.T @ zeta
        step_dual = (V @ (Vt_zeta / (s2 + damping))).unsqueeze(-1)

        # Apply O_k^T to land back in parameter space (paper Eq. 4.6).
        step_primal = [
            s.squeeze(-1)
            for s in apply_joint_JT(
                interior_inputs,
                interior_grad_outputs,
                boundary_inputs,
                boundary_grad_outputs,
                step_dual,
            )
        ]

        # phi <- mu_k phi + step_primal
        for p, s in zip(params, step_primal):
            self.state[p]["phi"].mul_(mu_val).add_(s)

        # Norm-constrained update (paper Eq. 2.15).
        if isinstance(lr, float):
            norm_phi = sum(
                (self.state[p]["phi"] ** 2).sum() for p in params
            ).sqrt()
            scale = min(lr, (sqrt(norm_constraint) / norm_phi).item())
            for p in params:
                p.data.add_(self.state[p]["phi"], alpha=scale)
        else:
            if lr[0] == "grid_line_search":
                directions = [self.state[p]["phi"] for p in params]

                def f() -> Tensor:
                    interior_loss_ls = self._eval_loss(X_Omega, y_Omega, "interior")
                    boundary_loss_ls = self._eval_loss(X_dOmega, y_dOmega, "boundary")
                    return interior_loss_ls + boundary_loss_ls

                grid = lr[1]
                grid_line_search(f, params, directions, grid)

        # Cache V^R_{k, alpha} and ceil(alpha_k) for next iteration. If T_k was rank
        # deficient this step, keep the previous cache rather than overwriting with None.
        if use_adaptive:
            self._V_prev_alpha = V_alpha.detach()
            self._alpha_prev_ceil = alpha_ceil

        self._last_mu = mu_val
        self._last_alpha = float(alpha_k.item())
        self._last_alpha_ceil = alpha_ceil
        self._last_rank = r_k
        self._last_beta_tilde = beta_val

        if self._print_every > 0 and self.steps % self._print_every == 0:
            print(
                f"PRIME-SR step {self.steps}: "
                f"mu={mu_val:.6f}, alpha={float(alpha_k.item()):.4f}, "
                f"ceil(alpha)={alpha_ceil}, r={r_k}, beta_tilde={beta_val:.6f}"
            )

        self.steps += 1
        return interior_loss, boundary_loss

    def _eval_loss(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        loss_evaluator = self.LOSS_EVALUATORS[self.equation][loss_type]
        loss, _, _ = loss_evaluator(self.layers, X, y)
        return loss
