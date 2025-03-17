from typing import List, Dict

import torch
import numpy as np

from math import sqrt
from argparse import ArgumentParser, Namespace
from torch import Tensor, cholesky_solve, arange, zeros_like, diag
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

from rla_pinns.parse_utils import parse_known_args_and_remove_from_argv
from rla_pinns.pinn_utils import evaluate_boundary_loss
from rla_pinns.optim.utils import (
    evaluate_losses_with_layer_inputs_and_grad_outputs,
    apply_joint_J,
    apply_joint_JT,
    compute_joint_JJT,
)


def parse_randomized_args(verbose: bool = False, prefix="randomized_") -> Namespace:
    parser = ArgumentParser(
        description="Parse arguments for setting up the randomized optimizer."
    )
    parser.add_argument(
        f"--{prefix}lr",
        type=float,
        help="Learning rate for the optimizer.",
        default=0.001,
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
        default=500,
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping parameter.",
        default=0.1,
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
        choices=["sketch_naive", "sketch_nystrom", "exact"],
        help="Randomized method to approximate the range of a low-rank matrix.",
        default="sketch_nystrom",
    )
    parser.add_argument(
        f"--{prefix}use_woodbury",
        action="store_false",
        help="Whether to use the kernel matrix the optimization.",
    )
    parser.add_argument(
        f"--{prefix}momentum",
        type=float,
        default=0.0,
        help="Momentum parameter for the optimizer.",
    )

    args = parse_known_args_and_remove_from_argv(parser)
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
        rank_val: int = 300,
        damping: float = 0.0,
        eps: float = 1e-10,
        approximation: str = "random_naive",
        use_woodbury: bool = True,
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
            use_woodbury=use_woodbury,
            approximation=approximation,
            rank_val=rank_val,
            eps=eps,
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
        self.eps = eps

        if not use_woodbury:
            raise ValueError("Only Woodbury is supported.")

        if approximation == "exact":
            self._get_inv = self._exact_inv_damped_kernel
        elif approximation == "sketch_naive":
            self._randomization = rsvd
            self._get_inv = self._sketch_inv_damped_kernel
        elif approximation == "sketch_nystrom":
            self._randomization = nystrom_approx
            self._get_inv = self._sketch_inv_damped_kernel
        else:
            raise ValueError(f"Randomization method {approximation} not supported.")

        if momentum != 0.0:
            self._step_fn = self._spring_step
            # initialize phi
            (group,) = self.param_groups
            for p in group["params"]:
                self.state[p]["phi"] = zeros_like(p)
        else:
            self._step_fn = self._normal_step

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        (group,) = self.param_groups
        lr = group["lr"]
        params = group["params"]
        damping = group["damping"]
        momentum = group["momentum"]
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

        N_dOmega = X_dOmega.shape[0]
        boundary_residual = boundary_residual.detach() / sqrt(N_dOmega)

        N_Omega = X_Omega.shape[0]
        interior_residual = interior_residual.detach() / sqrt(N_Omega)

        epsilon = -torch.cat([interior_residual, boundary_residual]).flatten()

        inputs = (interior_inputs, interior_grad_outputs, boundary_inputs, boundary_grad_outputs)
        hps = (damping, momentum, norm_constraint, lr)
        self._step_fn(params, inputs, epsilon, hps)
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

    def _update_preconditioner(
        self,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs
    ) -> None:

        U, S, V = self._randomization(
            interior_inputs, interior_grad_outputs,
            boundary_inputs, boundary_grad_outputs,
            self.l, self.Ds, self.eps,
        )

        self.U = U
        self.S = S
        self.V = V

    @staticmethod
    def _exact_inv_damped_kernel(
            interior_inputs, interior_grad_outputs,
            boundary_inputs, boundary_grad_outputs, 
            damping
            ):
        
        JJT = compute_joint_JJT(
            interior_inputs, interior_grad_outputs,
            boundary_inputs, boundary_grad_outputs,
        ).detach()

        # apply damping
        idx = arange(JJT.shape[0], device=JJT.device)
        JJT[idx, idx] = JJT.diag() + damping
        inv = cholesky_solve(torch.eye(JJT.shape[0], device=JJT.device), cholesky(JJT))
        return inv
        
    def _sketch_inv_damped_kernel(self,
            interior_inputs, interior_grad_outputs,
            boundary_inputs, boundary_grad_outputs, 
            damping,
            ):
        
        self._update_preconditioner(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs
        )

        I = torch.eye(self.U.sahpe[0], device=self.U.device)
        arg = diag(inv(damping / self.S)) + self.V.T @ self.U
        rhs = cholesky_solve(self.V.T, cholesky(arg))
        inv = I - self.U @ rhs
        inv.mul_(1 / damping)
        return inv
        
    def _normal_step(self, params, inputs, residuals, hps) -> Tensor:
        (
            interior_inputs, 
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs
        ) = inputs

        damping, _, _, lr = hps

        inv = self._get_inv(
            interior_inputs, interior_grad_outputs, 
            boundary_inputs, boundary_grad_outputs,
            damping
            )
        invR = inv @ residuals.unsqueeze(-1)

        # apply JT
        step = [s.squeeze(-1) for s in apply_joint_JT(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            invR,
        )]

        step = step.split([p.numel() for p in params])
        step = [-s.view(p.shape) for s, p in zip(step, params)]

        for p, s in zip(params, step):
            p.data.add_(s, alpha=lr)

    def _spring_step(self, params, inputs, residuals, hps):
        (
            interior_inputs, 
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs
        ) = inputs

        damping, momentum, norm_constraint, lr = hps

        inv = self._get_inv(
            interior_inputs, interior_grad_outputs,
            boundary_inputs, boundary_grad_outputs,
            damping
        )

        J_phi = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["phi"].unsqueeze(-1) for p in params],
        ).squeeze(-1)
        
        zeta = residuals - J_phi.mul_(momentum)

        # apply inverse of damped OOT to zeta
        step = inv @ zeta.unsqueeze(-1)

        # apply OT
        step = [s.squeeze(-1) for s in apply_joint_JT(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            step,
        )]

        # update phi
        for p, s in zip(params, step):
            self.state[p]["phi"].mul_(momentum).add_(s)

        # compute effective learning rate
        norm_phi = sum([(self.state[p]["phi"] ** 2).sum() for p in params]).sqrt()
        scale = min(lr, (sqrt(norm_constraint) / norm_phi).item())

        # update parameters
        for p in params:
            p.data.add_(self.state[p]["phi"], alpha=scale)

def rsvd(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    l: int, Ds: int, *kwargs
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the randomized SVD of the the Gram matrix.

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
    Omega = torch.randn(Ds, l, dtype=torch.float64, device=interior_inputs[0].device)
    Y = apply_kernel(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        Omega,
    )

    Q, _ = torch.linalg.qr(Y)
    B = apply_kernel(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        Q,
    ).T

    U_tilde, S, V = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U, S, V


def nystrom_approx(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    l: int,
    Ds: int,
    eps: float,
    *kwargs
) -> Tuple[Tensor, Tensor]:

    Omega = torch.randn(Ds, l, device=interior_inputs[0].device)
    Omega, _ = torch.linalg.qr(Omega)

    Y = apply_kernel(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        Omega,
    )

    Y_v = Y + eps * Omega
    C = cholesky(Omega.T @ Y_v)
    B = cholesky_solve(Y_v, C)

    U, Sigma, _ = torch.linalg.svd(B)
    I = torch.ones(Sigma.shape[0], device=interior_inputs[0].device)
    S = max(0, Sigma @ Sigma - eps * I)

    return U, S, U

def apply_kernel(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    v: Tensor,
):
    JTv = apply_joint_JT(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        v,
    )
    JJTv = apply_joint_J(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        JTv,
    )

    return JJTv