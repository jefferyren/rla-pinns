"""Implements linear operators."""

from math import sqrt
from typing import List, Callable

from torch import as_tensor, float32, float64, Tensor
import numpy as np

from torch import Tensor, cat, zeros
from einops import einsum
from torch.nn import Linear, Module
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator

from rla_pinns import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from rla_pinns.pinn_utils import (
    evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
)


class GramianLinearOperator:
    """Class for linear operators representing a Gramian.

    Pre-computes the information required to multiply with the Gramian in one
    backward pass. This saves computation when doing multiple Gramian-vector
    products, compared to matrix-free multiplication with the Gramian based on
    nested autodiff.

    Attributes:
        SUPPORTED_APPROXIMATIONS: The supported Gramian approximations.
        SUPPORTED_GGN_TYPES: The supported GGN types.
        SUPPORTED_LOSS_TYPES: The supported loss types.
        EVAL_FNS: The functions to evaluate the loss, inputs and output gradients
            for each equation type.
    """

    SUPPORTED_APPROXIMATIONS = {"full", "per_layer"}
    SUPPORTED_GGN_TYPES = {"type-2"}
    SUPPORTED_LOSS_TYPES = {"interior", "boundary"}
    EVAL_FNS = {
        "poisson": {
            "interior": poisson_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
            "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
        },
        "heat": {
            "interior": heat_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
            "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
        },
        "fokker-planck-isotropic": {
            "interior": fokker_planck_isotropic_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
            "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
        },
        "log-fokker-planck-isotropic": {
            "interior": log_fokker_planck_isotropic_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
            "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
        },
    }

    def __init__(
        self,
        equation: str,
        layers: List[Module],
        X: Tensor,
        y: Tensor,
        loss_type: str,
        ggn_type: str = "type-2",
        approximation: str = "full",
    ):
        """Pre-compute the information for the Gramian-vector product.

        Args:
            equation: The type of equation to solve.
            layers: The neural network's layers.
            X: The input data tensor.
            y: The target data tensor.
            loss_type: The type of loss to use. Can be `'interior'` or `'boundary'`.
            ggn_type: The type of GGN to use. Default: `'type-2'`.
            approximation: The Gramian approximation. Can be `'full'` or `'per_layer'`.

        Raises:
            NotImplementedError: If there are trainable parameters in unsupported
                layers.
            ValueError: For unsupported values of `approximation`.
            ValueError: For unsupported values of `ggn_type`.
        """
        self.layers = layers
        self.batch_size = X.shape[0]
        self.ggn_type = ggn_type
        self.equation = equation
        self.loss_type = loss_type

        if ggn_type not in self.SUPPORTED_GGN_TYPES:
            raise ValueError(
                f"GGN type {ggn_type!r} not supported. "
                f"Choose from {self.SUPPORTED_GGN_TYPES}."
            )

        if approximation not in self.SUPPORTED_APPROXIMATIONS:
            raise ValueError(
                f"Approximation {approximation!r} not supported. "
                f"Choose from {self.SUPPORTED_APPROXIMATIONS}."
            )
        self.approximation = approximation

        self.layer_idxs = []
        for idx, layer in enumerate(layers):
            if isinstance(layer, Linear):
                if (
                    layer.weight.requires_grad
                    and layer.bias is not None
                    and layer.bias.requires_grad
                ):
                    self.layer_idxs.append(idx)
                elif any(p.requires_grad for p in layer.parameters()):
                    raise NotImplementedError(
                        "Trainable linear layers must have differentiable weight+bias."
                    )
            elif any(p.requires_grad for p in layer.parameters()):
                raise NotImplementedError(
                    "Trainable parameters must be in linear layers."
                )

        self.params = sum(
            (list(layers[idx].parameters()) for idx in self.layer_idxs), []
        )
        # compute quantities required for Gramian-vector products
        eval_fn = self.EVAL_FNS[equation][loss_type]
        loss, self.layer_inputs, self.layer_grad_outputs = eval_fn(layers, X, y, ggn_type)

        # `grad_outputs` have scaling `1/N`, but we need `1/sqrt(N)` for the matvec
        for g_out in self.layer_grad_outputs.values():
            g_out.mul_(sqrt(self.batch_size))

        self.grad = grad(loss, self.params, allow_unused=True, materialize_grads=True)
        print(f"grad: {sum([val.size(dim=-1) for val in self.grad])}")


    def __matmul__(self, v: Tensor) -> Tensor:
        """Multiply the Gramian onto a vector or matrix.

        Args:
            v: The vector or matrix to multiply with the Gramian. Has shape `[D]` or
                `[D, N]` where `D` is the total number of parameters in the network
                and `N` is the number of vectors to multiply.

        Returns:
            The result of the Gramian-vector product. Has shape `[D]` or `[D, N]`.
        """
        is_vector = v.ndim == 1
        num_vectors = 1 if is_vector else v.shape[1]

        # split into parameters
        v = [
            v_p.reshape(*p.shape, num_vectors)
            for v_p, p in zip(v.split([p.numel() for p in self.params]), self.params)
        ]

        # matrix-vector product in list format
        matmul_func = {"full": self._matmul_full, "per_layer": self._matmul_per_layer}[
            self.approximation
        ]
        Gv = matmul_func(v)

        # flatten and concatenate
        Gv = cat([Gv_p.flatten(end_dim=-2) for Gv_p in Gv])
        return Gv.squeeze(-1) if is_vector else Gv

    def _matmul_full(self, v_list: List[Tensor]) -> List[Tensor]:
        """Multiply the full Gramian onto a matrix.

        Args:
            v_list: The matrix to multiply with the Gramian in list format.
                Each entry has shape `[*p.shape, N]` where `p` is a parameter tensor
                and `N` is the number of vectors to multiply.

        Returns:
            The result of the Gramian-matrix product in list format. Has same shape
            as `v_list`.
        """
        assert self.loss_type != "both", "Full Gramian not supported for both losses."

        (dev,) = {v.device for v in v_list}
        (dt,) = {v.dtype for v in v_list}
        (num_vectors,) = {v.shape[-1] for v in v_list}
        JT_v = zeros(self.batch_size, num_vectors, device=dev, dtype=dt)

        # multiply with the transpose Jacobian
        for i, layer_idx in enumerate(self.layer_idxs):
            z = self.layer_inputs[layer_idx]
            g = self.layer_grad_outputs[layer_idx]
            # combine weight and bias
            v_idx = cat([v_list[2 * i], v_list[2 * i + 1].unsqueeze(1)], dim=1)
            JT_v.add_(
                einsum(z, g, v_idx, "n ... d_in, n ... d_out, d_out d_in v -> n v")
            )

        result = []

        # multiply with the Jacobian
        for layer_idx in self.layer_idxs:
            z = self.layer_inputs[layer_idx]
            g = self.layer_grad_outputs[layer_idx]
            v_idx = einsum(z, g, JT_v, "n ... d_in, n ... d_out, n v -> d_out d_in v")
            # un-combine weight and bias
            result.extend([v_idx[:, :-1], v_idx[:, -1]])

        return result

    def _matmul_per_layer(self, v_list: List[Tensor]) -> List[Tensor]:
        """Multiply the per-layer Gramian onto a matrix.

        Args:
            v_list: The matrix to multiply with the Gramian in list format.
                Each entry has shape `[*p.shape, N]` where `p` is a parameter tensor
                and `N` is the number of vectors to multiply.

        Returns:
            The result of the Gramian-matrix product in list format. Has same shape
            as `v_list`.
        """
        result = []

        # multiply with the per-layer Gramian
        for i, layer_idx in enumerate(self.layer_idxs):
            z = self.layer_inputs[layer_idx]
            g = self.layer_grad_outputs[layer_idx]
            # combine weight and bias
            v_idx = cat([v_list[2 * i], v_list[2 * i + 1].unsqueeze(1)], dim=1)
            # multiply with JJT
            JT_v = einsum(z, g, v_idx, "n ... d_in, n ... d_out, d_out d_in v -> n v")
            v_idx = einsum(z, g, JT_v, "n ... d_in, n ... d_out, n v -> d_out d_in v")
            # un-combine weight and bias
            result.extend([v_idx[:, :-1], v_idx[:, -1]])

        return result

    def __rmatmul__(self, v: Tensor) -> Tensor:

        """Multiply a vector or matrix from the left with the Gramian.

        Args:
            v: The vector or matrix to multiply from the left with the Gramian. Has shape `[N]` or `[N, D]`,
            where `D` is the total number of parameters in the network, and `N` is the number of vectors
            to multiply.

        Returns:
            The result of the left Gramian-vector or Gramian-matrix product.
        """

        is_vector = v.ndim == 1
        Gv_result = self.__matmul__(v=v if is_vector else v.T)
        Gv_result = Gv_result if is_vector else Gv_result.T

        return Gv_result.squeeze(0) if is_vector else Gv_result


class SumLinearOperator:
    def __init__(self, g1: GramianLinearOperator, g2: GramianLinearOperator):
        self.g1 = g1
        self.g2 = g2                

    def __matmul__(self, other):
        return self.g1 @ other + self.g2 @ other

    def __rmatmul__(self, other):
        assert other.ndim == 2, "Expected a 2D tensor"
        return (self @ other.T).T


class GramianScipyOperator(LinearOperator):
    """Re-implementation of a Gramian operator as a Scipy LinearOperator.

    Attributes:
        layers: List of neural network layers (should be instances of torch.nn.Module).
        X: The input tensor.
        y: The target tensor.
        eval_fn: The function to evaluate the loss, layer inputs, and layer gradient outputs.
        batch_size: The batch size from the input tensor.
        params: List of trainable parameters in the network.
        layer_inputs: Cached layer inputs needed for the Gramian computation.
        layer_grad_outputs: Cached gradient outputs for each layer.
    """

    DATA_MAP = {
        float32: np.float32,
        float64: np.float64,
    }

    def __init__(
        self,
        equation: str,
        layers: List[Callable],
        X: Tensor,
        y: Tensor,
        dX: Tensor,
        dy: Tensor,
        loss_type: str
    ):
        """Initialize the Gramian operator."""
        if loss_type == "both":
            assert dX is not None and dy is not None

            self.G = SumLinearOperator(
                GramianLinearOperator(equation, layers, X, y, "interior"),
                GramianLinearOperator(equation, layers, dX, dy, "boundary"),
            )

            shape = (sum(p.numel() for p in self.G.g1.params), ) * 2
            (self.torch_dtype, ) = {p.dtype for p in self.G.g1.params}
            self.device = self.G.g1.params[0].device
        else:
            self.G = GramianLinearOperator(equation, layers, X, y, loss_type)
            shape = (sum(p.numel() for p in self.G.params), ) * 2
            (self.torch_dtype, ) = {p.dtype for p in self.G.params}
            self.device = self.G.params[0].device
            
        super().__init__(dtype=self.DATA_MAP[self.torch_dtype], shape=shape)

    def _matmat(self, v):
        """Computes Gramian @ v (right-side multiplication)."""
        v = as_tensor(v, dtype=self.torch_dtype, device=self.device)
        Gv = self.G @ v
        return Gv.cpu().numpy().astype(self.dtype)

    def _rmatmat(self, v):
        """Computes v @ Gramian (left-side multiplication)."""
        v = as_tensor(v, dtype=self.torch_dtype, device=self.device)
        vG = v @ self.G
        return vG.cpu().numpy().astype(self.dtype)
