from torch import cat, randn, zeros_like, allclose, manual_seed
from einops import einsum
from math import sqrt

Dim = 5
N_Omega = 10
N_dOmega = 5
k = 3
list_len = 2

manual_seed(0)

interior_inputs = [randn(N_Omega, Dim) for _ in range(list_len)]
boundary_inputs = [randn(N_dOmega, Dim) for _ in range(list_len)]
interior_grad_outputs = [randn(N_Omega, Dim) for _ in range(list_len)]
boundary_grad_outputs = [randn(N_dOmega, Dim) for _ in range(list_len)]


M = randn(N_Omega + N_dOmega, k)
M_out1 = zeros_like(M)

for idx in range(len(interior_inputs)):

    J_boundary = einsum(
        boundary_grad_outputs[idx] * sqrt(N_dOmega),
        boundary_inputs[idx],
        "n d_out, n d_in -> n d_out d_in",
    )
    J_interior = einsum(
        interior_grad_outputs[idx] * sqrt(N_Omega),
        interior_inputs[idx],
        "n ... d_out, n ... d_in -> n d_out d_in",
    )
    J = cat([J_interior, J_boundary], dim=0).flatten(start_dim=1)

    M_out1.add_(J @ (J.T @ M))


M_interior, M_boundary = M.split([N_Omega, N_dOmega])
M_out2 = zeros_like(M)

for idx in range(len(interior_inputs)):

    scaled_boundary_grad_outputs = boundary_grad_outputs[idx] * sqrt(N_dOmega)
    scaled_interior_grad_outputs = interior_grad_outputs[idx] * sqrt(N_Omega)

    # NOTE: the following operations are done in two einsums for performance reasons

    JTM_boundary = einsum(
        scaled_boundary_grad_outputs,
        M_boundary,
        "n ... d_out, n k -> n ... d_out k",
    )

    JTM_boundary = einsum(
        JTM_boundary,
        boundary_inputs[idx],
        "n ... d_out k, n ... d_in-> d_out d_in k",
    )

    JTM_interior = einsum(
        scaled_interior_grad_outputs,
        M_interior,
        "n ... d_out, n k -> n ... d_out k",
    )

    JTM_interior = einsum(
        JTM_interior,
        interior_inputs[idx],
        "n ... d_out k, n ... d_in-> d_out d_in k",
    )

    JTM = JTM_boundary + JTM_interior

    JJTM_boundary = einsum(
        scaled_boundary_grad_outputs,
        JTM,
        "n ... d_out, d_out d_in k -> n ... d_in k",
    )

    JJTM_boundary = einsum(
        JJTM_boundary,
        boundary_inputs[idx],
        "n ... d_in k, n ... d_in -> n k",
    )

    JJTM_interior = einsum(
        scaled_interior_grad_outputs,
        JTM,
        "n ... d_out, d_out d_in k -> n ... d_in k",
    )

    JJTM_interior = einsum(
        JJTM_interior,
        interior_inputs[idx],
        "n ... d_in k, n ... d_in -> n k",
    )

    JJTM = cat([JJTM_interior, JJTM_boundary], dim=0)

    M_out2.add_(JJTM)

assert allclose(M_out1, M_out2), "Outputs differ"
