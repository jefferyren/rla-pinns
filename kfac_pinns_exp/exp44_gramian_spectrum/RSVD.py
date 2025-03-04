from time import time
from torch import allclose, cat, device, eye, float64, zeros, randn
from torch.linalg import eigvalsh, svd, qr, matrix_rank
from torch.nn import Sequential
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from kfac_pinns_exp.linops import GramianLinearOperator
from kfac_pinns_exp.train import (
    INTERIOR_LOSS_EVALUATORS,
    create_condition_data,
    create_interior_data,
    evaluate_boundary_loss,
    set_up_layers,
)


def RSVD(G, l=None):
    if l is None:
        l = int(matrix_rank(G).item() + 5)

    m, n = G.shape
    Omega = randn(n, l, dtype=G.dtype, device=G.device)
    Y = G @ Omega
    Q, R = qr(Y)
    B = Q.T @ G
    U_hat, S, V = svd(B, full_matrices=False)
    U = Q @ U_hat
    return U, S, V

def GramianManual(net, loss, residual, dt, dev):
    params = [p for p in net.parameters() if p.requires_grad]

    Ds = [p.numel() for p in params]
    D = sum(Ds)
    G = zeros(D, D, dtype=dt, device=dev)

    for d in range(D):
        e_d = zeros(D, dtype=dt, device=dev)
        e_d[d] = 1.0

        e_d = e_d.split(Ds)
        e_d = [e.reshape_as(p) for e, p in zip(e_d, params)]

        Ge_d = ggn_vector_product_from_plist(loss, residual, params, e_d)
        Ge_d = cat([e.flatten() for e in Ge_d])

        G[:, d] = Ge_d

    start = time()
    U, S, Vh = RSVD(G)
    print(f"SVD computation with functorch took {time() - start:.3f} s.")
    return U, S, Vh, G


def GramianLinops(net, equation, layers, X, y, loss_type, dev, dt):
    params = [p for p in net.parameters() if p.requires_grad]

    Ds = [p.numel() for p in params]
    D = sum(Ds)

    G_operator = GramianLinearOperator(equation, layers, X, y, loss_type)
    G = G_operator @ eye(D, dtype=dt, device=dev)

    start = time()
    U, S, Vh = RSVD(G)
    print(f"SVD computation with linops took {time() - start:.3f} s.")
    return U, S, Vh, G


def check_rank_and_eigenspectrum(G):
    rank = matrix_rank(G)
    eigenvalues = eigvalsh(G)
    return rank, eigenvalues


def main():
    model_name = "mlp-tanh-64"
    equation = "poisson"
    condition = "sin_product"
    dim_Omega = 2
    num_data = 128
    loss_type = "interior"
    dt, dev = float64, device("cpu")


    layers = [layer.to(dev, dt) for layer in set_up_layers(model_name, equation, dim_Omega)]
    net = Sequential(*layers).to(dev, dt)

    data_generator = {"interior": create_interior_data, "boundary": create_condition_data}[loss_type]
    X, y = data_generator(equation, condition, dim_Omega, num_data)
    X, y = X.to(dev, dt), y.to(dev, dt)

    loss_evaluator = {
        "interior": INTERIOR_LOSS_EVALUATORS[equation],
        "boundary": evaluate_boundary_loss,
    }[loss_type]

    loss, residual, _ = loss_evaluator(layers, X, y)
    
    _, S1, _, G1 = GramianManual(net, loss, residual, dt, dev)
    _, S2, _, G2 = GramianLinops(net, equation, layers, X, y, loss_type, dev, dt)

    rank1, eigenvalues1 = check_rank_and_eigenspectrum(G1)
    rank2, eigenvalues2 = check_rank_and_eigenspectrum(G2)
    assert rank1 == rank2
    assert allclose(eigenvalues1, eigenvalues2)
    assert allclose(S1, S2)


if __name__ == "__main__":
    main()
