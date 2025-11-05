"""Implements the SPRING optimizer (https://arxiv.org/pdf/2401.10190v1) for PINNs."""

from argparse import ArgumentParser, Namespace
from math import sqrt
from typing import List, Tuple

import torch
from torch import Tensor, arange, cat, cholesky_solve, zeros_like
from torch.linalg import cholesky, inv
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


def parse_SPRING_args(verbose: bool = False, prefix="SPRING_") -> Namespace:
    """Parse command-line arguments for `SPRING`.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: The prefix for the arguments. Default: `'SPRING_'`.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="Parse arguments for setting up SPRING.")

    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate or line search strategy for the optimizer.",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping factor for the optimizer.",
        default=1e-3,
    )
    parser.add_argument(
        f"--{prefix}momentum",
        type=float,
        help="Decay factor of the previous step.",
        default=0.99,
    )
    parser.add_argument(
        f"--{prefix}norm_constraint",
        type=float,
        help="Norm constraint on the natural gradient.",
        default=1e-3,
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=SPRING.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )

    args = parse_known_args_and_remove_from_argv(parser)

    # overwrite the lr value
    lr = f"{prefix}lr"
    if any(char.isdigit() for char in getattr(args, lr)):
        setattr(args, lr, float(getattr(args, lr)))

    if getattr(args, lr) == "grid_line_search":
        # generate the grid from the command line arguments and overwrite the
        # `lr` entry with a tuple containing the grid
        grid = parse_grid_line_search_args(verbose=verbose)
        setattr(args, lr, (getattr(args, lr), grid))

    if verbose:
        print("Parsed arguments for SPRING: ", args)

    return args


class SPRING(Optimizer):
    """SPRING optimizer for PINN problems.

    See https://arxiv.org/pdf/2401.10190v1 for details.
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
        lr: float,
        damping: float = 1e-3,
        momentum: float = 0.99,
        norm_constraint: float = 1e-3,
        equation: str = "poisson",
        lb_window: int = 30,  # lookback window, 0 = no momentum
        beta0: float = 0.9,  # initial momentum factor
        **kwargs,  # Accept any additional keyword arguments
    ):
        print(f"SPRING __init__ method entered with lb_window={lb_window}, beta0={beta0}!", flush=True)
        
        # If lb_window or beta0 are not provided in kwargs, use defaults for adaptive momentum
        if 'lb_window' not in kwargs and lb_window == 30:
            # Enable adaptive momentum by default
            lb_window = 30
        if 'beta0' not in kwargs and beta0 == 0.9:
            # Use default initial momentum factor
            beta0 = 0.9
            
        print(f"SPRING using lb_window={lb_window}, beta0={beta0} for adaptive momentum", flush=True)
        """Set up the SPRING optimizer.

        Args:
            layers: The layers that form the neural network.
            lr: The learning rate.
            damping: The non-negative damping factor (λ in the paper).
                Default: `1e-3` (taken from Section 4 of the paper).
            decay_factor: The decay factor (μ in the paper). Must be in `[0; 1)`.
                Default: `0.99` (taken from Section 4 of the paper).
            norm_constraint: The positive norm constraint (C in the paper).
                Default: `1e-3` (taken from Section 4 of the paper).
            equation: Equation to solve. Currently supports `'poisson'`, `'heat'`, and
                `'fokker-planck-isotropic'`. Default: `'poisson'`.

        Raises:
            ValueError: If the optimizer is used with per-parameter options.
            ValueError: If the equation is not supported.
        """
        print(f"SPRING.__init__ called with lr={lr}, damping={damping}, momentum={momentum}", flush=True)
        defaults = dict(
            lr=lr,
            damping=damping,
            decay_factor=momentum,  # same as beta
            norm_constraint=norm_constraint,
        )
        params = sum((list(layer.parameters()) for layer in layers), [])
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SPRING does not support per-parameter options.")

        if equation not in self.SUPPORTED_EQUATIONS:
            raise ValueError(
                f"Equation {equation} not supported."
                f" Supported are: {self.SUPPORTED_EQUATIONS}."
            )
        self.equation = equation
        self.steps = 0
        self.layers = layers
        
        # DEBUG: Confirm SPRING optimizer is being used
        print(f"SPRING optimizer initialized with damping={damping}, momentum={momentum}", flush=True)

        # initialize phi
        (group,) = self.param_groups
        for p in group["params"]:
            self.state[p]["phi"] = zeros_like(p)

        # ADDED NEW PART FOR ADAPTIVE MOMENTUM
        self.p = int(lb_window)
        self._use_adaptive_beta = self.p > 0
        print(f"DEBUG: SPRING adaptive momentum setup: p={self.p}, _use_adaptive_beta={self._use_adaptive_beta}", flush=True)

        if self._use_adaptive_beta:
            (dev,) = {p.device for p in group["params"]}
            (dt,)  = {p.dtype  for p in group["params"]}

            self._res_buffer = torch.zeros(2 * self.p, device=dev, dtype=dt)
            self._buf_idx = 0
            self._r_hat = torch.tensor(1.0, device=dev, dtype=dt)
            self._checkpoint_idx = torch.tensor(1, device=dev)  # int-like tensor

            # seed beta (decay_factor) for the very first steps
            group["decay_factor"] = float(beta0)
            print(f"DEBUG: SPRING adaptive momentum initialized with beta0={beta0}, buffer size={2 * self.p}", flush=True)
        else:
            print(f"DEBUG: SPRING adaptive momentum disabled (p={self.p})", flush=True)

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        print("SPRING step method entered!", flush=True)
        
        # RUNTIME FIX: Initialize adaptive momentum if missing due to caching issues
        if not hasattr(self, '_use_adaptive_beta') or self._use_adaptive_beta is None:
            print("RUNTIME FIX: Initializing missing adaptive momentum attributes", flush=True)
            self.p = 30  # lookback window
            self._use_adaptive_beta = True
            
            # Get device and dtype from parameters
            (group,) = self.param_groups
            (dev,) = {p.device for p in group["params"]}
            (dt,)  = {p.dtype  for p in group["params"]}
            
            # Initialize adaptive momentum buffers using already imported torch
            import torch  # This should work in the method context
            self._res_buffer = torch.zeros(2 * self.p, device=dev, dtype=dt)
            self._buf_idx = 0
            self._r_hat = torch.tensor(1.0, device=dev, dtype=dt)
            self._checkpoint_idx = torch.tensor(1, device=dev)
            
            # Set initial momentum factor
            group["decay_factor"] = 0.9
            print(f"RUNTIME FIX: Adaptive momentum initialized - p={self.p}, use_adaptive={self._use_adaptive_beta}")
        
        """Take a step.

        Args:
            X_Omega: Input for the interior loss.
            y_Omega: Target for the interior loss.
            X_dOmega: Input for the boundary loss.
            y_dOmega: Target for the boundary loss.

        Returns:
            Tuple of the interior and boundary loss before taking the step.
        """
        try:
            print(f"SPRING step() method called - step {self.steps}", flush=True)
            
            (group,) = self.param_groups
            params = group["params"]
            lr = group["lr"]
            damping = group["damping"]
            decay_factor = group["decay_factor"]
            norm_constraint = group["norm_constraint"]

            #DEBUG:
            print(f"SPRING step {self.steps}: decay_factor={decay_factor}, damping={damping}", flush=True)
        except Exception as e:
            print(f"SPRING step() EXCEPTION: {e}", flush=True)
            raise
        # compute OOT
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
        OOT = compute_joint_JJT(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        ).detach()
        # apply damping
        idx = arange(OOT.shape[0], device=OOT.device)
        OOT[idx, idx] = OOT.diag() + damping

        # update zeta
        # compute the residual
        N_dOmega = X_dOmega.shape[0]
        boundary_residual = boundary_residual.detach() / sqrt(N_dOmega)

        N_Omega = X_Omega.shape[0]
        interior_residual = interior_residual.detach() / sqrt(N_Omega)

        epsilon = -cat([interior_residual, boundary_residual]).flatten()

        # ADAPTIVE PART 
        if self._use_adaptive_beta:
            res_norm = epsilon.norm()  # scalar
            self._res_buffer[self._buf_idx % (2 * self.p)] = res_norm
            self._buf_idx += 1

        O_phi = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["phi"].unsqueeze(-1) for p in params],
        ).squeeze(-1)
        zeta = epsilon - O_phi.mul_(decay_factor)

        # apply inverse of damped OOT to zeta
        step = cholesky_solve(zeta.unsqueeze(-1), cholesky(OOT))

        # apply OT
        step = [
            s.squeeze(-1)
            for s in apply_joint_JT(
                interior_inputs,
                interior_grad_outputs,
                boundary_inputs,
                boundary_grad_outputs,
                step,
            )
        ]

        # update phi
        for p, s in zip(params, step):
            self.state[p]["phi"].mul_(decay_factor).add_(s)

        if isinstance(lr, float):
            # compute effective learning rate
            norm_phi = sum([(self.state[p]["phi"] ** 2).sum() for p in params]).sqrt()
            scale = min(lr, (sqrt(norm_constraint) / norm_phi).item())

            # update parameters
            for p in params:
                p.data.add_(self.state[p]["phi"], alpha=scale)
        else:
            if lr[0] == "grid_line_search":

                directions = [self.state[p]["phi"] for p in params]

                def f() -> Tensor:
                    """Closure to evaluate the loss.

                    Returns:
                        Loss value.
                    """
                    interior_loss = self._eval_loss(X_Omega, y_Omega, "interior")
                    boundary_loss = self._eval_loss(X_dOmega, y_dOmega, "boundary")
                    return interior_loss + boundary_loss

                grid = lr[1]
                grid_line_search(f, params, directions, grid)

        self.steps += 1
        # NEW: maybe update beta (= decay_factor) for the next step
        if self._use_adaptive_beta:
            self._maybe_update_beta()

        return interior_loss, boundary_loss

    def _maybe_update_beta(self):
        with torch.no_grad():
            """Update group['decay_factor'] (beta) every p steps after a 2p warm-up"""
            if (self.steps % self.p != 0) or (self._buf_idx < 2 * self.p):
                return

            (group,) = self.param_groups
            p = self.p
            two_p = 2 * p
            idx = self._buf_idx
            dev = self._res_buffer.device
            dt  = self._res_buffer.dtype

            # last p entries (t) vs previous p entries (t-p)
            t_idx  = torch.arange(1, p + 1, device=dev)
            tp_idx = torch.arange(p + 1, 2 * p + 1, device=dev)
            idxs_t  = (idx - t_idx)  % two_p
            idxs_tp = (idx - tp_idx) % two_p

            eps_t  = (self._res_buffer.index_select(0, idxs_t)  ** 2).sum()
            eps_tp = (self._res_buffer.index_select(0, idxs_tp) ** 2).sum()
            # numerical guard
            eps_tp = torch.clamp(eps_tp, min=torch.finfo(dt).eps)

            r_ip = eps_t / eps_tp
            r_ip = torch.minimum(torch.tensor(1.0, device=dev, dtype=dt), r_ip)

            # a_n = n^{log n}
            n_old = self._checkpoint_idx
            n_new = self._checkpoint_idx + 1
            n_old_f = n_old.to(dt)
            n_new_f = n_new.to(dt)

            # guard log(1)
            a_old = torch.pow(n_old_f, torch.log(n_old_f + 1e-12))
            a_new = torch.pow(n_new_f, torch.log(n_new_f))
            alph  = a_old / a_new

            # r_hat ← α r_hat + (1-α) * min(1, r_ip)
            self._r_hat = alph * self._r_hat + (1.0 - alph) * r_ip

            # ρ = max(0, 1 - r_hat^{1/p});  β = (1-ρ)/(1+ρ)
            rho = torch.clamp(1.0 - torch.pow(self._r_hat, 1.0 / float(p)), min=0.0, max=1.0)
            beta_new = (1.0 - rho) / (1.0 + rho)  # in [0,1)

            # Optional safety clamp (uncomment if desired):
            # beta_new = torch.clamp(beta_new, 0.0, 0.999)

            group["decay_factor"] = float(beta_new.item())   # used next step
            self._checkpoint_idx = n_new

    def _eval_loss(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        """Evaluate the loss.

        Args:
            X: Input data.
            y: Target data.
            loss_type: Type of the loss function. Can be `'interior'` or `'boundary'`.

        Returns:
            The differentiable loss.
        """
        loss_evaluator = self.LOSS_EVALUATORS[self.equation][loss_type]
        loss, _, _ = loss_evaluator(self.layers, X, y)
        return loss
