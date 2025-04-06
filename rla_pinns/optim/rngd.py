from typing import List, Dict, Callable

import torch
from functools import partial
from math import sqrt
from argparse import ArgumentParser, Namespace
from torch import Tensor, arange, zeros_like, cholesky_solve
from typing import List, Tuple
from torch.nn import Module
from torch.optim import Optimizer
from torch.linalg import qr, cholesky
from rla_pinns import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from rla_pinns.optim.line_search import (
    grid_line_search,
    parse_grid_line_search_args,
)
from rla_pinns.parse_utils import parse_known_args_and_remove_from_argv
from rla_pinns.pinn_utils import evaluate_boundary_loss
from rla_pinns.optim.utils import (
    evaluate_losses_with_layer_inputs_and_grad_outputs,
    apply_joint_J,
    apply_joint_JT,
    apply_joint_JJT,
)


def parse_randomized_args(verbose: bool = False, prefix="RNGD_") -> Namespace:
    parser = ArgumentParser(
        description="Parse arguments for setting up the randomized optimizer."
    )
    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate or line search strategy for the optimizer.",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=RNGD.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    parser.add_argument(
        f"--{prefix}rank_val",
        type=int,
        help="Low-rank approximation parameter.",
        default=0,
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping parameter.",
        default=1e-8,
    )
    parser.add_argument(
        f"--{prefix}approximation",
        type=str,
        choices=["nystrom", "exact"],
        help="Randomized method to approximate the range of a low-rank matrix.",
        default="exact",
    )
    parser.add_argument(
        f"--{prefix}momentum",
        type=float,
        default=0.0,
        help="Momentum parameter for the optimizer.",
    )
    parser.add_argument(
        f"--{prefix}norm_constraint",
        type=float,
        default=0.0,
        help="Norm constraint parameter for the SPRING step.",
    )

    args = parse_known_args_and_remove_from_argv(parser)

    # overwrite the lr value
    lr = f"{prefix}lr"
    if any(char.isdigit() for char in getattr(args, lr)):
        setattr(args, lr, float(getattr(args, lr)))

    if getattr(args, lr) == "grid_line_search":
        assert (
            args.RNGD_norm_constraint == 0.0
        ), "Norm constraint is not supported with grid line search."
        # generate the grid from the command line arguments and overwrite the
        # `lr` entry with a tuple containing the grid
        grid = parse_grid_line_search_args(verbose=verbose)
        setattr(args, lr, (getattr(args, lr), grid))

    if verbose:
        print("Parsed arguments for randomized optimizer: ", args)

    return args


class RNGD(Optimizer):

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
        lr: float = 1e-3,
        equation: str = "poisson",
        damping: float = 0.0,
        approximation: str = "exact",
        rank_val: int = 0,
        momentum: float = 0.0,
        norm_constraint: float = 1e-3,
        *,
        maximize: bool = False,
    ):

        params = sum((list(layer.parameters()) for layer in layers), [])
        defaults = dict(
            lr=lr,
            damping=damping,
            maximize=maximize,
            momentum=momentum,
            norm_constraint=norm_constraint,
            approximation=approximation,
            rank_val=rank_val,
            equation=equation,
        )
        super().__init__(params, defaults)

        if equation not in self.LOSS_EVALUATORS:
            raise ValueError(
                f"Equation {equation} not supported. "
                f"Supported equations are: {list(self.LOSS_EVALUATORS.keys())}."
            )

        self.equation = equation
        self.layers = layers
        self.layers_idx = [
            idx for idx, layer in enumerate(self.layers) if list(layer.parameters())
        ]
        self.steps = 0

        self.Ds = sum(
            [
                p.numel()
                for layer in self.layers
                for p in layer.parameters()
                if p.requires_grad
            ]
        )

        self.l = rank_val
        self._approximation = approximation

        if approximation == "exact":
            assert (
                rank_val == 0 or rank_val is None
            ), "Rank value is not used with exact approximation. Has to be zero or None."
            self._get_inverse = self._inv_exact
        elif approximation == "nystrom":
            assert rank_val > 0, "Rank value must be a positive integer."
            self._get_inverse = self._inv_sketch
        else:
            raise ValueError(f"Randomization method {approximation} not supported.")

        if momentum != 0.0:
            self._step_fn = self._spring_step
            # initialize phi
            (group,) = self.param_groups
            for p in group["params"]:
                self.state[p]["phi"] = zeros_like(p)
        else:
            self._step_fn = self._engd_step

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:

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

        N_dOmega = X_dOmega.shape[0]
        boundary_residual = boundary_residual.detach() / sqrt(N_dOmega)

        N_Omega = X_Omega.shape[0]
        interior_residual = interior_residual.detach() / sqrt(N_Omega)

        epsilon = -torch.cat([interior_residual, boundary_residual]).flatten()

        step = self._step_fn(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            epsilon,
        )
        self._update_parameters(step, X_Omega, y_Omega, X_dOmega, y_dOmega)
        self.steps += 1

        return interior_loss, boundary_loss

    def _update_parameters(self, directions, X_Omega, y_Omega, X_dOmega, y_dOmega):
        (group,) = self.param_groups
        lr = group["lr"]
        params = group["params"]
        momentum = group["momentum"]

        if isinstance(lr, float):
            if momentum != 0.0:
                norm_constraint = group["norm_constraint"]
                norm_phi = sum([(d**2).sum() for d in directions]).sqrt()
                scale = min(lr, (sqrt(norm_constraint) / norm_phi).item())
            else:
                scale = lr

            for p, d in zip(params, directions):
                p.data.add_(d, alpha=scale)
        else:
            if lr[0] == "grid_line_search":

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

            else:
                raise ValueError(f"Unsupported line search: {lr[0]}.")

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

    def _engd_step(
        self,
        interior_inputs: Dict[int, Tensor],
        interior_grad_outputs: Dict[int, Tensor],
        boundary_inputs: Dict[int, Tensor],
        boundary_grad_outputs: Dict[int, Tensor],
        residuals,
    ):
        (group,) = self.param_groups
        params = group["params"]
        damping = group["damping"]
        (dev,) = {p.device for p in params}
        (dt,) = {p.dtype for p in params}
        (N_Omega,) = {
            t.shape[0]
            for t in list(interior_inputs.values())
            + list(interior_grad_outputs.values())
        }
        (N_dOmega,) = {
            t.shape[0]
            for t in list(boundary_inputs.values())
            + list(boundary_grad_outputs.values())
        }

        fn = partial(
            apply_joint_JJT,
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        )
        inv = self._get_inverse(fn, N_Omega + N_dOmega, dt, dev, self.l, damping)

        step = inv @ residuals.unsqueeze(-1)
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

        step = [s.view(p.shape) for s, p in zip(step, params)]
        return step

    def _spring_step(
        self,
        interior_inputs: Dict[int, Tensor],
        interior_grad_outputs: Dict[int, Tensor],
        boundary_inputs: Dict[int, Tensor],
        boundary_grad_outputs: Dict[int, Tensor],
        residuals,
    ):
        (group,) = self.param_groups
        params = group["params"]
        damping = group["damping"]
        momentum = group["momentum"]
        (dev,) = {p.device for p in params}
        (dt,) = {p.dtype for p in params}
        (N_Omega,) = {
            t.shape[0]
            for t in list(interior_inputs.values())
            + list(interior_grad_outputs.values())
        }
        (N_dOmega,) = {
            t.shape[0]
            for t in list(boundary_inputs.values())
            + list(boundary_grad_outputs.values())
        }

        fn = partial(
            apply_joint_JJT,
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        )
        inv = self._get_inverse(fn, N_Omega + N_dOmega, dt, dev, self.l, damping)
        J_phi = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["phi"].unsqueeze(-1) for p in params],
        ).squeeze(-1)

        zeta = residuals - J_phi.mul_(momentum)
        step = inv @ zeta.unsqueeze(-1)

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

        for p, s in zip(params, step):
            self.state[p]["phi"].mul_(momentum).add_(s)

        step = [self.state[p]["phi"] for p in params]
        return step

    @staticmethod
    def _inv_sketch(
        fn: Callable[[Tensor], Tensor], N: int, dt, dev, l: int, damping: float
    ):
        U, S = nystrom_stable(fn, N, l, dt, dev)
        lhs = U @ torch.diag(1 / (S + damping)) @ U.T
        idx = torch.arange(0, N, device=dev)
        arg = -(U @ U.T)
        arg[idx, idx] = arg.diag() + 1
        rhs = arg / damping
        out = lhs + rhs

        return out

    @staticmethod
    def _inv_exact(
        fn: Callable[[Tensor], Tensor], N: int, dt, dev, l: int, damping: float
    ):
        JJT = exact(fn, N, 0, dt, dev)
        I = torch.eye(N, dtype=dt, device=dev)
        out = torch.linalg.solve(JJT + I * damping, I)
        return out


def exact(apply_A, N, _, dt, dev) -> Tuple[Tensor, Tensor]:
    I = torch.eye(N, dtype=dt, device=dev)
    A = apply_A(I)
    ####
    return A


def nystrom_naive(apply_A, dim, sketch_size, dt, dev) -> Tuple[Tensor, Tensor]:
    Omega = torch.randn(dim, sketch_size, dtype=dt, device=dev)
    Omega, _ = qr(Omega)
    AO = apply_A(Omega)
    OAO = Omega.T @ AO
    B = cholesky_solve(AO.T, cholesky(OAO))
    A_hat = AO @ B
    return A_hat


def nystrom_stable(
    A: Callable[[Tensor], Tensor],
    dim: int,
    sketch_size,
    dt,
    dev,
) -> Tensor:
    """Compute a stable Nystrom approximation."""
    O = torch.randn(dim, sketch_size, device=dev, dtype=dt)
    O, _ = qr(O)
    Y = A(O)

    nu = 1e-7

    Y.add_(O, alpha=nu)
    C = cholesky(O.T @ Y, upper=True)

    B = torch.linalg.solve_triangular(C, Y, upper=True, left=False)
    U, Sigma, _ = torch.linalg.svd(B, full_matrices=False)
    Lambda = (Sigma**2 - nu).clamp(min=0.0)

    return U, Lambda
