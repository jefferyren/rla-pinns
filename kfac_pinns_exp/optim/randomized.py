from typing import List

import torch
import numpy as np

from argparse import ArgumentParser, Namespace
from torch import Tensor
from typing import List, Tuple
from torch.nn import Module
from torch.optim import Optimizer
from kfac_pinns_exp import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from kfac_pinns_exp.pinn_utils import (
    evaluate_boundary_loss,
    evaluate_boundary_loss_and_kfac,
)
from kfac_pinns_exp.optim.line_search import (
    grid_line_search,
    parse_grid_line_search_args,
)

from kfac_pinns_exp.linops import GramianLinearOperator, SumLinearOperator
from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv


def parse_randomized_args(verbose: bool = False, prefix="KFAC_") -> Namespace:
    parser = ArgumentParser(description="Parse arguments for setting up KFAC.")
    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate or line search strategy for the optimizer.",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=RandomizedOptimizer.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    parser.add_argument(
        f"--{prefix}rank_val",
        type=int,
        help="Low-rank approximation parameter.",
        default=500,
    )

    args = parse_known_args_and_remove_from_argv(parser)

    # overwrite the lr value
    lr = f"{prefix}lr"
    if any(char.isdigit() for char in getattr(args, lr)):
        setattr(args, lr, float(getattr(args, lr)))

    if getattr(args, lr) == "auto":
        # use a small learning rate for the first step
        lr_init = 1e-6
        setattr(args, lr, (getattr(args, lr), lr_init))

    return args

class RandomizedOptimizer(Optimizer):
    
    LOSS_AND_KFAC_EVALUATORS = {
        "poisson": {
            "interior": poisson_equation.evaluate_interior_loss_and_kfac,
            "boundary": evaluate_boundary_loss_and_kfac,
        },
        "heat": {
            "interior": heat_equation.evaluate_interior_loss_and_kfac,
            "boundary": evaluate_boundary_loss_and_kfac,
        },
        "fokker-planck-isotropic": {
            "interior": fokker_planck_isotropic_equation.evaluate_interior_loss_and_kfac,  # noqa: B950
            "boundary": evaluate_boundary_loss_and_kfac,
        },
        "log-fokker-planck-isotropic": {
            "interior": log_fokker_planck_isotropic_equation.evaluate_interior_loss_and_kfac,  # noqa: B950
            "boundary": evaluate_boundary_loss_and_kfac,
        },
    }
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
        ema_factor: float = 0.95,
        rank_val: int = 300,
        dampening: float = 0.0,
        tol_eigvals: float = 1e-10,
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
            dampening=dampening,
            maximize=maximize,
            ema_factor=ema_factor,
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

        # Hyperparameters
        self.l = rank_val
        self.tol = tol_eigvals

        # Preconditioning matrices
        self.U = torch.zeros(self.Ds, self.l, dtype=torch.float64, device=layers[0].weight.device)
        self.S = torch.zeros(self.l, dtype=torch.float64, device=layers[0].weight.device)
        self.V = torch.zeros(self.Ds, self.l, dtype=torch.float64, device=layers[0].weight.device)

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:

        loss_interior = self.eval_loss(X_Omega, y_Omega, "interior")
        loss_interior.backward()
        loss_boundary = self.eval_loss(X_dOmega, y_dOmega, "boundary")
        loss_boundary.backward()

        operator, grad = self._get_operator_and_grad(X_Omega, y_Omega, X_dOmega, y_dOmega)
        self._update_preconditioner(operator, X_Omega.device)
        self._update_parameters(grad, X_Omega, y_Omega, X_dOmega, y_dOmega)

        self.steps += 1

        return loss_interior, loss_boundary
    
    def _get_operator_and_grad(self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor):
        G_interior = GramianLinearOperator(self.equation, self.layers, X_Omega, y_Omega, "interior")
        G_boundary = GramianLinearOperator(self.equation, self.layers, X_dOmega, y_dOmega, "boundary")
            
        G = SumLinearOperator(G_interior, G_boundary)
        grad = torch.cat(
            [
                (g_int + g_bnd).flatten()
                for g_int, g_bnd in zip(G_interior.grad, G_boundary.grad)
            ]
        )

        del G_interior, G_boundary
        return G, grad

    def _RSVD(self, operator, l: int, dev: str) -> Tuple[Tensor, Tensor, Tensor]:
        (group, ) = self.param_groups
        ema_factor = group["ema_factor"]

        Omega = torch.randn(self.Ds, l, dtype=torch.float64, device=dev)

        G_t = operator @ Omega        
        G_svd_prev = self.U @ (torch.diag(self.S) @ (self.V.T @ Omega))
        Y = ema_factor * G_svd_prev + (1 - ema_factor) * G_t

        Q, _ = torch.linalg.qr(Y)
        B =  Q.T @ operator

        U_tilde, S, V = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde

        return U, S, V

    def _update_preconditioner(self, operator, dev: str) -> None:        
        U, S, V = self._RSVD(operator, self.l, dev)
        valid_eig = min(sum(S > self.tol), self.l)

        self.S = S[:valid_eig]
        self.U = U[:, :valid_eig]
        self.V = V[:valid_eig, :]
     
    def _update_parameters(self, grad: Tensor, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor) -> None:
        (group, ) = self.param_groups
        lr = group["lr"]
        params = group["params"]

        inv = self._add_damping_and_invert()
        grad_l = inv @ grad

        def f() -> Tensor:
            interior_loss = self.eval_loss(X_Omega, y_Omega, "interior")
            boundary_loss = self.eval_loss(X_dOmega, y_dOmega, "boundary")
            return interior_loss + boundary_loss

        grad_l_list = grad_l.split([p.numel() for p in params])
        grad_l_list = [-g.view(p.shape) for g, p in zip(grad_l_list, params)]

        if isinstance(lr, float):
            for param, param_grad in zip(params, grad_l_list):
                param.data.add_(param_grad, alpha=lr)
        else:
            grid = np.logspace(-3, 0, 10)
            best, _ = grid_line_search(f, params, grad_l_list, grid)
        
    def eval_loss(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        loss_evaluator = self.LOSS_EVALUATORS[self.equation][loss_type]
        loss, _, _ = loss_evaluator(self.layers, X, y)
        return loss

    def _add_damping_and_invert(self):
        (group, ) = self.param_groups
        dampening = group["dampening"]

        if dampening == 0.0:
            inv = self.V @ (torch.diag(1 / self.S) @ self.U.T)
        else:
            I = torch.ones(self.S.shape[0], self.S.shape[0], device=self.S.device)
            USVT = self.U @ (torch.diag(self.S) @ self.V.T)
            USVT.data.mul_(dampening)
            inv = torch.linalg.pinv(I + USVT)
            inv.data.mul_(1 / dampening)

        return inv

