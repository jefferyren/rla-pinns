"""Tests for `damped_cholesky`.

The rank-deficient case reproduces the failure that killed ENGD-Woodbury and
SS-SPRING on the 5d Poisson benchmark: a kernel whose numerical rank is well
below N, plus an absolute damping that is negligible against the kernel's scale.
"""

from pytest import raises
from torch import allclose, arange, diag, eye, float64, manual_seed, randn

from rla_pinns.optim.linalg_utils import damped_cholesky


def _damped(M, lam):
    """`M + lam * I`, without touching `M`."""
    return M + lam * eye(M.shape[0], dtype=M.dtype)


def test_well_conditioned_uses_nominal_damping():
    """A comfortably PD matrix must be untouched: same lambda, no escalation."""
    manual_seed(0)
    A = randn(40, 12, dtype=float64)
    M = A @ A.T + 1e-2 * eye(40, dtype=float64)
    raw = M.clone()

    L, lam, n = damped_cholesky(M, 1e-3, site="test")

    assert n == 0
    assert lam == 1e-3
    assert allclose(L @ L.T, _damped(raw, 1e-3))


def test_rank_deficient_escalates_and_succeeds():
    """The 5d-Poisson failure mode: rank << N, damping negligible vs the scale."""
    manual_seed(0)
    # rank 30 of 200, with a kernel scale large enough that an absolute damping
    # of 1e-8 sits below the rounding noise of the product itself -- which is
    # exactly the regime p5 is in. Verified to raise without the escalation:
    # "the leading minor of order 31 is not positive-definite".
    A = randn(200, 30, dtype=float64) * 1e4
    M = A @ A.T
    raw = M.clone()

    # 6.8e-8 is ENGD-W's published 5d Poisson damping; it cannot rescue this.
    L, lam, n = damped_cholesky(M, 6.804474e-8, site="test")

    assert n >= 1, "should have escalated"
    assert lam > 6.804474e-8
    assert allclose(L @ L.T, _damped(raw, lam))


def test_zero_damping_escalates_from_a_relative_floor():
    """damping=0 must still have somewhere to go: growth alone would stay at 0."""
    manual_seed(0)
    A = randn(60, 8, dtype=float64)
    M = A @ A.T
    raw = M.clone()

    L, lam, n = damped_cholesky(M, 0.0, site="test")

    assert lam > 0.0
    assert n >= 1
    assert allclose(L @ L.T, _damped(raw, lam))


def test_diagonal_is_restored_from_raw_each_attempt(monkeypatch):
    """Escalation must not compound: lambda is applied to the RAW diagonal.

    The failures are forced rather than provoked with an ill-conditioned matrix.
    Whether a given matrix is *numerically* positive definite is platform
    dependent -- the same matrix and damping factorize under one LAPACK build
    and not another, which is the entire reason `damped_cholesky` exists. An
    earlier version of this test relied on that luck and passed on macOS /
    torch 1.12 while going vacuous (n == 0) on Savio / torch 2.2.0.
    """
    import rla_pinns.optim.linalg_utils as mod

    real_cholesky = mod.cholesky
    calls = {"n": 0}

    def fail_twice(M):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError(
                "linalg.cholesky: The factorization could not be completed "
                "because the input is not positive-definite (the leading minor "
                "of order 7 is not positive-definite)."
            )
        return real_cholesky(M)

    monkeypatch.setattr(mod, "cholesky", fail_twice)

    manual_seed(0)
    A = randn(80, 10, dtype=float64) * 10.0
    M = A @ A.T
    raw_diag = M.diag().clone()

    _, lam, n = mod.damped_cholesky(M, 1e-12, site="test")

    assert n == 2
    assert lam == 1e-12 * 10.0 * 10.0  # two x10 steps, not compounded
    # Had each retry added lambda on top of the previous, the final diagonal
    # would be raw + 1e-12 + 1e-11 + 1e-10 rather than raw + 1e-10.
    assert allclose(M.diag(), raw_diag + lam)


def test_gives_up_rather_than_escalating_forever():
    """An indefinite matrix is not a damping problem and must not be masked."""
    M = diag(arange(-5.0, 5.0, dtype=float64))  # genuinely indefinite, scale ~1

    with raises(RuntimeError, match="still not positive definite"):
        damped_cholesky(M, 0.0, site="test", max_escalations=2)


def test_non_definiteness_errors_are_reraised(monkeypatch):
    """An OOM must propagate immediately, never be retried as if it were damping."""
    import rla_pinns.optim.linalg_utils as mod

    def boom(_):
        raise RuntimeError("CUDA out of memory. Tried to allocate 3.80 GiB")

    monkeypatch.setattr(mod, "cholesky", boom)
    M = eye(4, dtype=float64)

    with raises(RuntimeError, match="out of memory"):
        mod.damped_cholesky(M, 1e-8, site="test")
