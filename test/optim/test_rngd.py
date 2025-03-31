from typing import List
from pytest import mark
from rla_pinns.utils import run_verbose

from torch import norm, randn, float64, manual_seed
from test.exp1_poisson5d import rngd
from rla_pinns.optim.rngd import nystrom_naive

ARGS = [
    # train with SPRING and RNGD on different equations
    *[
        [
            "--optimizer=RNGD", # NOTE: this is a placeholder, the actual optimizer is set in the script
            f"--equation={equation}",
            f"--boundary_condition={condition}",
            "--SPRING_decay_factor=0.9",
            "--SPRING_damping=1e-10",
            "--SPRING_lr=0.001",
            "--RNGD_approximation=exact",
            "--RNGD_momentum=0.9",
            "--RNGD_damping=1e-10",
            "--RNGD_lr=0.001",
            "--model=mlp-tanh-64",
            "--N_Omega=1000",
            "--N_dOmega=200",
            "--batch_frequency=10000",
            "--dim_Omega=2",
        ]
        for (equation, condition) in list(
            [
                ("poisson", "cos_sum"),
                ("heat", "cos_sum"),
                ("fokker-planck-isotropic", "gaussian"),
                ("log-fokker-planck-isotropic", "gaussian"),
            ]
        )
    ],
]


ARG_IDS = ["_".join(cmd) for cmd in ARGS]


@mark.parametrize("args", ARGS, ids=ARG_IDS)
def test_rngd(args: List[str]) -> None:
    """Test the training script (integration test)."""
    # Run the training script with the provided arguments
    run_verbose(["python", rngd.__file__] + args)


def check_approx(A, A_hat):
    diff = A - A_hat
    fro_norm_diff = norm(diff, p='fro')
    fro_norm_A = norm(A, p='fro')
    error = (fro_norm_diff / fro_norm_A).item()
    return error


def test_nystrom():
    manual_seed(0)
    A = randn(50, 100, dtype=float64)
    B = A.T @ A

    r = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    errors_naive = []
    for val in r:
        manual_seed(1)
        errors_naive.append(check_approx(B, nystrom_naive(B.matmul, B.shape[0], float64, "cpu", val)))

    for i in range(1, len(r)):
        assert errors_naive[i - 1] > errors_naive[i], f"Error increases for larger sketch values in naive version."

    assert errors_naive[-1] < 1e-9, f"Error is too large for the largest sketch value in naive version."
