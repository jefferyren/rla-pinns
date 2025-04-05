"""Test SPRING-related functionality."""

from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import List, Tuple

from pytest import mark
from torch import cat, device, dtype, eye, float64, manual_seed, zeros
from torch.linalg import eigvalsh

from rla_pinns.linops import GramianLinearOperator
from rla_pinns.optim.utils import (
    _apply_individual_J,
    _apply_individual_JT,
    apply_joint_J,
    apply_joint_JT,
    compute_individual_JJT,
    compute_joint_JJT,
    apply_joint_JJT,
    evaluate_losses_with_layer_inputs_and_grad_outputs,
)
from rla_pinns.train import (
    create_condition_data,
    create_interior_data,
    set_up_layers,
)

PDE_CASES: List[Tuple[str, str]] = [
    ("poisson", "sin_product"),
    ("poisson", "cos_sum"),
    ("heat", "sin_product"),
    ("fokker-planck-isotropic", "gaussian"),
    ("log-fokker-planck-isotropic", "gaussian"),
]
PDE_IDS = [f"{equation}-{condition}" for equation, condition in PDE_CASES]
MODEL_CASES = ["mlp-tanh-64"]


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("model", MODEL_CASES, ids=MODEL_CASES)
@mark.parametrize("equation, condition", PDE_CASES, ids=PDE_IDS)
def test_leading_eigenvalues_G_and_JJT(
    equation: str,
    condition: str,
    model: str,
    device: device,
    dtype: dtype = float64,
    N_Omega: int = 64,
    N_dOmega: int = 32,
    dim_Omega: int = 3,
):
    """Make sure the leading Gramian and Jacobian outer product eigenvalues match.

    Args:
        equation: String specifying the PDE.
        condition: String specifying the conditions.
        model: String specifying the model.
        device: The device to run the test on.
        dtype: The data type to run the test in. Defaults to `float64`.
        N_Omega: The number of interior data points. Defaults to `64`.
        N_dOmega: The number of boundary data points. Defaults to `32`.
        dim_Omega: The dimension of the domain. Defaults to `3`.
    """
    manual_seed(0)  # make deterministic

    # generate neural network and data
    layers = [
        layer.to(device, dtype) for layer in set_up_layers(model, equation, dim_Omega)
    ]
    num_params = sum(sum(p.numel() for p in layer.parameters()) for layer in layers)

    X_Omega, y_Omega = [
        t.to(device, dtype)
        for t in create_interior_data(equation, condition, dim_Omega, N_Omega)
    ]
    X_dOmega, y_dOmega = [
        t.to(device, dtype)
        for t in create_condition_data(equation, condition, dim_Omega, N_dOmega)
    ]

    # ground truth: Eigenvalues of the Gramians
    identity = eye(num_params, device=device, dtype=dtype)
    G_interior = (
        GramianLinearOperator(equation, layers, X_Omega, y_Omega, "interior") @ identity
    )
    G_boundary = (
        GramianLinearOperator(equation, layers, X_dOmega, y_dOmega, "boundary")
        @ identity
    )
    G = G_interior + G_boundary

    G_interior_evals = eigvalsh(G_interior)
    G_boundary_evals = eigvalsh(G_boundary)
    G_evals = eigvalsh(G)

    # compare with: Eigenvalues of the Jacobian outer product
    (
        _,
        _,
        _,
        _,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
        layers, X_Omega, y_Omega, X_dOmega, y_dOmega, equation
    )
    JJT_interior = compute_individual_JJT(interior_inputs, interior_grad_outputs)
    JJT_boundary = compute_individual_JJT(boundary_inputs, boundary_grad_outputs)

    JJT = compute_joint_JJT(
        interior_inputs, interior_grad_outputs, boundary_inputs, boundary_grad_outputs
    )
    assert JJT_interior.shape == (N_Omega, N_Omega)
    assert JJT_boundary.shape == (N_dOmega, N_dOmega)
    assert JJT.shape == (N_Omega + N_dOmega, N_Omega + N_dOmega)

    JJT_interior_evals = eigvalsh(JJT_interior)
    JJT_boundary_evals = eigvalsh(JJT_boundary)
    JJT_evals = eigvalsh(JJT)

    # clip to same length, sort descendingly, and compare
    for jjt_evals, n, g_evals in zip(
        [JJT_interior_evals, JJT_boundary_evals, JJT_evals],
        [N_Omega, N_dOmega, N_Omega + N_dOmega],
        [G_interior_evals, G_boundary_evals, G_evals],
    ):
        effective_evals = min(num_params, n)
        g_evals = g_evals.flip(0)[:effective_evals]
        jjt_evals = jjt_evals.flip(0)[:effective_evals]
        report_nonclose(g_evals, jjt_evals)

    # Test the application joint JJT to a matrix
    JJT_via_matvec = apply_joint_JJT(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        eye(N_dOmega + N_Omega, device=device, dtype=dtype),
    )
    report_nonclose(JJT, JJT_via_matvec)


LOSS_TYPE_CASES = ["interior", "boundary"]


@mark.parametrize("loss_type", LOSS_TYPE_CASES, ids=LOSS_TYPE_CASES)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("model", MODEL_CASES, ids=MODEL_CASES)
@mark.parametrize("equation, condition", PDE_CASES, ids=PDE_IDS)
def test_apply_individual_J(
    equation: str,
    condition: str,
    model: str,
    device: device,
    loss_type: str,
    dtype: dtype = float64,
    N: int = 64,
    dim_Omega: int = 3,
):
    """Test multiplication with the Jacobian of one loss.

    Args:
        equation: String specifying the PDE.
        condition: String specifying the conditions.
        model: String specifying the model.
        device: The device to run the test on.
        loss_type: String specifying the loss type.
        dtype: The data type to run the test in. Defaults to `float64`.
        N: The number of data points to use. Defaults to `64`.
        dim_Omega: The dimension of the domain. Defaults to `3`.
    """
    assert loss_type in {"interior", "boundary"}
    manual_seed(0)  # make deterministic

    # generate neural network and data
    layers = [
        layer.to(device, dtype) for layer in set_up_layers(model, equation, dim_Omega)
    ]
    params = sum((list(layer.parameters()) for layer in layers), [])
    param_sizes = [p.numel() for p in params]
    num_params = sum(param_sizes)

    X_Omega, y_Omega = [
        t.to(device, dtype)
        for t in create_interior_data(equation, condition, dim_Omega, N)
    ]
    X_dOmega, y_dOmega = [
        t.to(device, dtype)
        for t in create_condition_data(equation, condition, dim_Omega, N)
    ]

    # ground truth: Gramian
    identity = eye(num_params, device=device, dtype=dtype)
    if loss_type == "interior":
        G = (
            GramianLinearOperator(equation, layers, X_Omega, y_Omega, loss_type)
            @ identity
        )
    else:
        G = (
            GramianLinearOperator(equation, layers, X_dOmega, y_dOmega, loss_type)
            @ identity
        )

    # compute the Jacobian through applications of J
    (
        _,
        _,
        _,
        _,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
        layers, X_Omega, y_Omega, X_dOmega, y_dOmega, equation
    )

    inputs = interior_inputs if loss_type == "interior" else boundary_inputs
    grad_outputs = (
        interior_grad_outputs if loss_type == "interior" else boundary_grad_outputs
    )

    # Apply J to matrix M
    Id = eye(num_params, device=device, dtype=dtype)
    Id = [
        m_p.reshape(*p.shape, num_params)
        for m_p, p in zip(Id.split(param_sizes), params)
    ]
    J = _apply_individual_J(inputs, grad_outputs, Id)

    report_nonclose(G, J.T @ J)


@mark.parametrize("loss_type", LOSS_TYPE_CASES, ids=LOSS_TYPE_CASES)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("model", MODEL_CASES, ids=MODEL_CASES)
@mark.parametrize("equation, condition", PDE_CASES, ids=PDE_IDS)
def test_apply_individual_JT(
    equation: str,
    condition: str,
    model: str,
    device: device,
    loss_type: str,
    dtype: dtype = float64,
    N: int = 64,
    dim_Omega: int = 3,
):
    """Test multiplication with the transpose Jacobian of one loss.

    Args:
        equation: String specifying the PDE.
        condition: String specifying the conditions.
        model: String specifying the model.
        device: The device to run the test on.
        loss_type: String specifying the loss type.
        dtype: The data type to run the test in. Defaults to `float64`.
        N: The number of data points to use. Defaults to `64`.
        dim_Omega: The dimension of the domain. Defaults to `3`.
    """
    assert loss_type in {"interior", "boundary"}
    manual_seed(0)  # make deterministic

    # generate neural network and data
    layers = [
        layer.to(device, dtype) for layer in set_up_layers(model, equation, dim_Omega)
    ]
    params = sum((list(layer.parameters()) for layer in layers), [])
    param_sizes = [p.numel() for p in params]
    num_params = sum(param_sizes)

    X_Omega, y_Omega = [
        t.to(device, dtype)
        for t in create_interior_data(equation, condition, dim_Omega, N)
    ]
    X_dOmega, y_dOmega = [
        t.to(device, dtype)
        for t in create_condition_data(equation, condition, dim_Omega, N)
    ]

    # ground truth: Gramian
    identity = eye(num_params, device=device, dtype=dtype)
    if loss_type == "interior":
        G = (
            GramianLinearOperator(equation, layers, X_Omega, y_Omega, loss_type)
            @ identity
        )
    else:
        G = (
            GramianLinearOperator(equation, layers, X_dOmega, y_dOmega, loss_type)
            @ identity
        )

    # compute the transpose Jacobian through applications of J^T
    (
        _,
        _,
        _,
        _,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
        layers, X_Omega, y_Omega, X_dOmega, y_dOmega, equation
    )

    inputs = interior_inputs if loss_type == "interior" else boundary_inputs
    grad_outputs = (
        interior_grad_outputs if loss_type == "interior" else boundary_grad_outputs
    )

    Id = eye(N, device=device, dtype=dtype)
    JT = _apply_individual_JT(inputs, grad_outputs, Id)
    JT = cat([m.flatten(end_dim=-2) for m in JT])

    # make sure that J.T @ J equals the Gramian
    report_nonclose(G, JT @ JT.T)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("model", MODEL_CASES, ids=MODEL_CASES)
@mark.parametrize("equation, condition", PDE_CASES, ids=PDE_IDS)
def test_apply_joint_J(
    equation: str,
    condition: str,
    model: str,
    device: device,
    dtype: dtype = float64,
    N_Omega: int = 64,
    N_dOmega: int = 32,
    dim_Omega: int = 3,
):
    """Test multiplication with the Jacobian of both losses.

    Args:
        equation: String specifying the PDE.
        condition: String specifying the conditions.
        model: String specifying the model.
        device: The device to run the test on.
        dtype: The data type to run the test in. Defaults to `float64`.
        N_Omega: The number of interior data points to use. Defaults to `64`.
        N_dOmega: The number of boundary data points to use. Defaults to `32`.
        dim_Omega: The dimension of the domain. Defaults to `3`.
    """
    manual_seed(0)  # make deterministic

    # generate neural network and data
    layers = [
        layer.to(device, dtype) for layer in set_up_layers(model, equation, dim_Omega)
    ]
    params = sum((list(layer.parameters()) for layer in layers), [])
    param_sizes = [p.numel() for p in params]
    num_params = sum(param_sizes)

    X_Omega, y_Omega = [
        t.to(device, dtype)
        for t in create_interior_data(equation, condition, dim_Omega, N_Omega)
    ]
    X_dOmega, y_dOmega = [
        t.to(device, dtype)
        for t in create_condition_data(equation, condition, dim_Omega, N_dOmega)
    ]

    # ground truth: Gramian
    identity = eye(num_params, device=device, dtype=dtype)
    G = (
        GramianLinearOperator(equation, layers, X_Omega, y_Omega, "interior") @ identity
        + GramianLinearOperator(equation, layers, X_dOmega, y_dOmega, "boundary")
        @ identity
    )

    # compute the Jacobian through applications of J
    (
        _,
        _,
        _,
        _,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
        layers, X_Omega, y_Omega, X_dOmega, y_dOmega, equation
    )

    # Apply J to matrix M
    Id = eye(num_params, device=device, dtype=dtype)
    Id = [
        m_p.reshape(*p.shape, num_params)
        for m_p, p in zip(Id.split(param_sizes), params)
    ]
    J = apply_joint_J(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        Id,
    )

    report_nonclose(G, J.T @ J)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("model", MODEL_CASES, ids=MODEL_CASES)
@mark.parametrize("equation, condition", PDE_CASES, ids=PDE_IDS)
def test_apply_joint_JT(
    equation: str,
    condition: str,
    model: str,
    device: device,
    dtype: dtype = float64,
    N_Omega: int = 64,
    N_dOmega: int = 32,
    dim_Omega: int = 3,
):
    """Test multiplication with the transpose Jacobian of both losses.

    Args:
        equation: String specifying the PDE.
        condition: String specifying the conditions.
        model: String specifying the model.
        device: The device to run the test on.
        dtype: The data type to run the test in. Defaults to `float64`.
        N_Omega: The number of interior data points to use. Defaults to `64`.
        N_dOmega: The number of boundary data points to use. Defaults to `32`.
        dim_Omega: The dimension of the domain. Defaults to `3`.
    """
    manual_seed(0)  # make deterministic

    # generate neural network and data
    layers = [
        layer.to(device, dtype) for layer in set_up_layers(model, equation, dim_Omega)
    ]
    params = sum((list(layer.parameters()) for layer in layers), [])
    param_sizes = [p.numel() for p in params]
    num_params = sum(param_sizes)

    X_Omega, y_Omega = [
        t.to(device, dtype)
        for t in create_interior_data(equation, condition, dim_Omega, N_Omega)
    ]
    X_dOmega, y_dOmega = [
        t.to(device, dtype)
        for t in create_condition_data(equation, condition, dim_Omega, N_dOmega)
    ]

    # ground truth: Gramian
    identity = eye(num_params, device=device, dtype=dtype)
    G = (
        GramianLinearOperator(equation, layers, X_Omega, y_Omega, "interior") @ identity
        + GramianLinearOperator(equation, layers, X_dOmega, y_dOmega, "boundary")
        @ identity
    )

    # compute the transpose Jacobian through applications of J^T
    (
        _,
        _,
        _,
        _,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
        layers, X_Omega, y_Omega, X_dOmega, y_dOmega, equation
    )

    JT = zeros(num_params, N_Omega + N_dOmega, device=device, dtype=dtype)
    Id = eye(N_dOmega + N_Omega, device=device, dtype=dtype)
    JT = apply_joint_JT(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        Id,
    )
    JT = cat([m.flatten(end_dim=-2) for m in JT])

    # make sure that J.T @ J equals the Gramian
    report_nonclose(G, JT @ JT.T)
