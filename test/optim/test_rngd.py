from time import time
from typing import List
from pytest import mark
from rla_pinns.utils import run_verbose
from rla_pinns.optim.rand_utils import (
    nystrom_naive,
    nystrom_stable,
    nystrom_stable_fast,
)
from torch.linalg import cholesky, solve_triangular
from torch import norm, randn, float64, manual_seed, diag

ARGS = [
    # train with SPRING and RNGD on different equations
    *[
        [
            "--optimizer=RNGD",  # NOTE: this is a placeholder, the actual optimizer is set in the script
            f"--equation=poisson",
            f"--boundary_condition=cos_sum",
            "--SPRING_decay_factor=0.9",
            "--SPRING_damping=1e-10",
            "--SPRING_lr=0.001",
            "--SPRING_norm_constraint=1e-3",
            "--RNGD_momentum=0.9",
            "--RNGD_damping=1e-10",
            "--RNGD_lr=0.001",
            "--RNGD_norm_constraint=1e-3",
            "--model=mlp-tanh-64",
            "--N_Omega=1000",
            "--N_dOmega=200",
            "--batch_frequency=10000",
            "--dim_Omega=2",
        ]
    ],
]


ARG_IDS = ["_".join(cmd) for cmd in ARGS]


# NOTE: This test is no longer valid as the new SPRING rescales the steps for bias corrections, and it doens't use the norm constraint
# @mark.parametrize("args", ARGS, ids=ARG_IDS)
# def test_rngd(args: List[str]) -> None:
#     """Test the training script (integration test)."""
#     # Run the training script with the provided arguments
#     run_verbose(["python", rngd.__file__] + args)


def check_approx(A, A_hat, is_B):
    if isinstance(A_hat, tuple):
        U, S = A_hat
        A_hat = U @ diag(S) @ U.T
    elif is_B:
        A_hat = A_hat @ A_hat.T

    diff = A - A_hat
    fro_norm_diff = norm(diff, p="fro")
    fro_norm_A = norm(A, p="fro")
    error = (fro_norm_diff / fro_norm_A).item()
    return error


def test_nystrom():
    manual_seed(0)
    A = randn(500, 1000, dtype=float64)
    B = A.T @ A

    r = [50, 100, 200, 500]

    errors_naive = []
    errors_stable = []
    errors_fast = []
    for val in r:
        manual_seed(1)
        errors_naive.append(
            check_approx(
                B, nystrom_naive(B.matmul, B.shape[0], val, float64, "cpu"), False
            )
        )
        manual_seed(1)
        errors_stable.append(
            check_approx(
                B, nystrom_stable(B.matmul, B.shape[0], val, float64, "cpu"), False
            )
        )
        manual_seed(1)
        errors_fast.append(
            check_approx(
                B, nystrom_stable_fast(B.matmul, B.shape[0], val, float64, "cpu"), True
            )
        )

    for i in range(1, len(r)):
        assert (
            errors_naive[i - 1] > errors_naive[i]
        ), f"Error increases for larger sketch values in naive version."
        assert (
            errors_stable[i - 1] > errors_stable[i]
        ), f"Error increases for larger sketch values in stable version."
        assert (
            errors_fast[i - 1] > errors_fast[i]
        ), f"Error increases for larger sketch values in fast version."

    assert (
        errors_naive[-1] < 1e-5
    ), f"Error is too large for the largest sketch value in naive version."
    assert (
        errors_stable[-1] < 1e-5
    ), f"Error is too large for the largest sketch value in stable version."
    assert (
        errors_fast[-1] < 1e-5
    ), f"Error is too large for the largest sketch value in fast version."

    start_naive = time()
    manual_seed(1)
    nystrom_naive(B.matmul, B.shape[0], 25, float64, "cpu")
    end_naive = time()

    start_stable = time()
    manual_seed(1)
    nystrom_stable(B.matmul, B.shape[0], 25, float64, "cpu")
    end_stable = time()

    start_fast = time()
    manual_seed(1)
    nystrom_stable_fast(B.matmul, B.shape[0], 25, float64, "cpu")
    end_fast = time()

    print("Time for naive version:", end_naive - start_naive)
    print("Time for stable version:", end_stable - start_stable)
    print("Time for fast version:", end_fast - start_fast)

    # assert (
    #     end_naive - start_naive > end_stable - start_stable
    # ), f"Stable version is slower than naive version."
