from typing import List, Dict

import torch
import numpy as np

from math import sqrt
from argparse import ArgumentParser, Namespace
from torch import Tensor, cholesky_solve, arange
from typing import List, Tuple
from torch.nn import Module
from torch.optim import Optimizer
from torch.linalg import cholesky
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
        compute_joint_JJT
)

def parse_randomized_args(verbose: bool = False, prefix="randomized_") -> Namespace:
    parser = ArgumentParser(description="Parse arguments for setting up the randomized optimizer.")
    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate or line search strategy for the optimizer.",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=Randomized.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    parser.add_argument(
        f"--{prefix}rank_val",
        type=int,
        help="Low-rank approximation parameter.",
        default=500,
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping parameter.",
        default=0.0,
    )
    parser.add_argument(
        f"--{prefix}eps",
        type=float,
        help="Floating point representation error.",
        default=1e-7,
    )
    parser.add_argument(
        f"--{prefix}approximation",
        type=str,
        choices=["random_naive", "random_nystrom"],
        help="Randomized method to approximate the range of a low-rank matrix.",
        default="random_naive",
    )
    parser.add_argument(
        f"--{prefix}woodbury",
        type=bool,
        help="Whether to use the kernel matrix the optimization.",
    )
    parser.add_argument(
        f"--{prefix}spring",
        type=bool,
        help="Whether to use the spring step.",
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

    if getattr(args, lr) == "auto":
        # use a small learning rate for the first step
        lr_init = 1e-6
        setattr(args, lr, (getattr(args, lr), lr_init))

    if verbose:
        print("Parsed arguments for randomized optimizer: ", args)

    return args

class Randomized(Optimizer):
    
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
        rank_val: int = 300,
        damping: float = 0.0,
        eps: float = 1e-10,
        approximation: str = 'random_naive',
        woodbury: bool = False,
        spring: bool = False,
        *,
        maximize: bool = False
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr is None:
            raise ValueError(f"Invalid learning rate: {lr}")
    
        params = sum((list(layer.parameters()) for layer in layers), [])
        defaults = dict(
            lr=lr,
            dampening=damping,
            maximize=maximize,
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

        self.Ds = sum([p.numel() for layer in self.layers for p in layer.parameters() if p.requires_grad])

        self.l = rank_val
        self.eps = eps
        
        if woodbury:
            self._matrix_fn = apply_kernel
        else:
            self._matrix_fn = apply_gramian

        if spring:
            self._get_step = spring_step
        else:
            self._get_step = normal_step

        if approximation == 'random_naive':
            self._randomization = rsvd
        elif approximation == 'random_nystrom':
            self._randomization = nystrom_approx
        else:
            raise ValueError(f"Randomization method {approximation} not supported.")

        # initialize phi
        (group,) = self.param_groups
        for p in group["params"]:
            self.state[p]["phi"] = zeros_like(p)

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        (group, ) = self.param_groups
        lr = group["lr"]
        params = group["params"]
        damping = group["damping"]
        dev = params[0].device

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
        
        self._update_preconditioner(
            interior_inputs, interior_grad_outputs, 
            boundary_inputs, boundary_grad_outputs,
            dev
        )

        step = self._get_step(epsilon, damping, dev)

        step = step.split([p.numel() for p in params])
        step = [-s.view(p.shape) for s, p in zip(step, params)]

        if isinstance(lr, float):
            for p, s in zip(params, step):
                p.data.add_(s, alpha=lr)
        else:

            def f() -> Tensor:
                interior_loss = self._eval_loss(X_Omega, y_Omega, "interior")
                boundary_loss = self._eval_loss(X_dOmega, y_dOmega, "boundary")
                return interior_loss + boundary_loss

            grid = np.logspace(-3, 0, 10)
            best, _ = grid_line_search(f, params, step, grid)
    
        self.steps += 1

        return interior_loss, boundary_loss

    def _eval_loss(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        """Evaluate the loss.

        Args:
            X: Input data.
            y: Target data.
            loss_type: Type of the loss function. Can be `'interior'` or `'boundary'`.

        Returns:
            The differentiable loss.
        """
        loss_evaluator = self.LOSS_EVALUATORS[loss_type][self.equation]
        loss, _, _ = loss_evaluator(self.model, X, y)
        return loss
    
    def _update_preconditioner(self, 
        interior_inputs,
        interior_grad_outputs, 
        boundary_inputs,
        boundary_grad_outputs, 
        damping, dev
    ) -> None:

        U, S, V = self._randomization(
            interior_inputs, interior_grad_outputs, 
            boundary_inputs, boundary_grad_outputs,
            self.woodbury, self.l, self.Ds, self.eps, dev)

        self.U = U
        self.S = S
        self.V = V

        I = torch.eye(self.Ds, dtype=torch.float64, device=dev)
        lhs = U @ torch.linalg.inv(torch.diag(damping/S) + V.T @ U) @ V.T
        out = (1 / damping) * (I - lhs)
        return out


def normal_step(self, precond, residuals) -> Tensor:
    return precond @ residuals


def spring_step():
    pass


def rsvd(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    matrix_fn,
    l: int, Ds: int, *kwargs, dev: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """ Compute the randomized SVD of the the Gram matrix.
    
        Args:
            interior_inputs: The layer inputs for the interior loss.
            interior_grad_outputs: The layer gradient outputs for the interior loss.
            boundary_inputs: The layer inputs for the boundary loss.
            boundary_grad_outputs: The layer gradient outputs for the boundary loss.
            l: rank of the sketch matrix.
            Ds: number of parameters.
            dev: device to use.

        Returns:
            U: left singular vectors of shape [Ds, l].
            S: singular values of shape [l, 1].
            V: right singular vectors of shape [l, Ds].
    """
    Omega = torch.randn(Ds, l, dtype=torch.float64, device=dev)
    Y = matrix_fn(
        interior_inputs, interior_grad_outputs, 
        boundary_inputs, boundary_grad_outputs, 
        Omega
        )

    Q, _ = torch.linalg.qr(Y)
    B = matrix_fn(
        interior_inputs, interior_grad_outputs, 
        boundary_inputs, boundary_grad_outputs, 
        Q
        ).T

    U_tilde, S, V = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U, S, V


def nystrom_approx(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    matrix_fn, l: int, Ds: int, eps: float, dev: str,
) -> Tuple[Tensor, Tensor]:

    Omega = torch.randn(Ds, l, device=dev)
    Omega, _ = torch.linalg.qr(Omega)

    Y = matrix_fn(
        interior_inputs, interior_grad_outputs, 
        boundary_inputs, boundary_grad_outputs, 
        Omega
        )

    Y_v = Y + eps * Omega
    C = cholesky(Omega.T @ Y_v)
    B = cholesky_solve(Y_v, C)

    U, Sigma, _ = torch.linalg.svd(B)
    I = torch.ones(Sigma.shape[0], device=dev)
    S = max(0, Sigma @ Sigma - eps * I)

    return U, S, U


def woodbury_step(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    epsilon: Tensor,
    damping: float,
    *kwargs
) -> Tuple[Tensor, Tensor]:

    OOT = compute_joint_JJT(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ).detach()
    
    I = arange(OOT.shape[0], device=OOT.device)
    OOT[idx, idx] = OOT.diag() + damping

    out = cholesky_solve(epsilon.unsqueeze(-1), cholesky(OOT)).squeeze(-1)

    step = apply_joint_JT(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        out,
    )

    return step


def apply_gramian(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    v: Tensor,
):
    JTv = apply_joint_J(
        interior_inputs, interior_grad_outputs, 
        boundary_inputs, boundary_grad_outputs, 
        v
        )
    JJTv = apply_joint_JT(
        interior_inputs, interior_grad_outputs, 
        boundary_inputs, boundary_grad_outputs, 
        JTv
        )

    return JJTv


def apply_kernel(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    v: Tensor
):
    JTv = apply_joint_JT(
        interior_inputs, interior_grad_outputs, 
        boundary_inputs, boundary_grad_outputs, 
        v
        )
    JJTv = apply_joint_J(
        interior_inputs, interior_grad_outputs, 
        boundary_inputs, boundary_grad_outputs, 
        JTv
        )

    return JJTv

