"""Numerically robust linear algebra shared by the kernel-based optimizers.

Deliberately depends on nothing but `torch`, so it can be unit-tested without
importing the rest of the package (which pulls in Taylor-mode autodiff and
therefore `torch.func`).
"""

from typing import Tuple

from torch import Tensor, arange, finfo
from torch.linalg import cholesky

# How many times a single call has escalated, keyed by call site. Used only to
# rate-limit the warning below; not part of any optimizer's state.
_ESCALATION_COUNTS = {}


def damped_cholesky(
    M: Tensor,
    damping: float,
    site: str = "",
    max_escalations: int = 8,
    growth: float = 10.0,
) -> Tuple[Tensor, float, int]:
    """Cholesky factor of `M + damping * I`, raising the damping until it exists.

    WHY THIS EXISTS. `torch.linalg.cholesky` *raises* on a matrix that is not
    numerically positive definite, and the damped kernel `JJ^T + lambda I` goes
    indefinite in float64 whenever the kernel's effective rank is below N and
    `lambda` is small *relative to the kernel's scale*. The optimizers add
    `damping` as an ABSOLUTE value, so the same nominal lambda is relatively
    tiny on a problem whose residuals are large. On the 5d Poisson benchmark
    (right-hand side of order 50, batch 3500) this killed ENGD-Woodbury after
    277 steps and SS-SPRING after 790 -- both while their L2 error was still
    falling monotonically. The failure is in the solve, not the optimization.

    WHAT IT CHANGES. At every step where the nominal `damping` factorizes, the
    result is bit-for-bit what the caller would have got before: lambda is used
    as given and `n_escalations` is 0. Only a step that would otherwise have
    thrown is affected, and then only by using a larger lambda for that one
    step. Report the escalation count -- it is the honest measure of how far a
    run departed from its nominal hyperparameters.

    Args:
        M: The undamped matrix. **Modified in place**: its diagonal is
            overwritten with `raw_diagonal + lambda`. Pass a clone to preserve it.
        damping: The nominal damping. May be 0.
        site: Short label for the warning, e.g. `"ENGD-W"`.
        max_escalations: How many times to grow lambda before giving up.
        growth: Multiplier applied to lambda on each retry.

    Returns:
        `(L, lambda_used, n_escalations)`.

    Raises:
        RuntimeError: If the factorization still fails after `max_escalations`
            retries, or if it failed for a reason other than definiteness (an
            out-of-memory error, say, which must not be silently retried).
    """
    idx = arange(M.shape[0], device=M.device)
    raw_diag = M.diag().clone()  # a vector: cheap next to the matrix itself

    # Relative floor, so that damping=0 -- or a damping so small it is lost in
    # the kernel's own rounding -- still has somewhere to escalate to.
    scale = raw_diag.abs().mean().item()
    floor = finfo(M.dtype).eps * max(scale, 1.0)

    lam = float(damping)
    last_error = None
    for n_escalations in range(max_escalations + 1):
        M[idx, idx] = raw_diag + lam
        try:
            L = cholesky(M)
        except RuntimeError as e:  # torch._C._LinAlgError subclasses RuntimeError
            if "positive-definite" not in str(e) and "positive definite" not in str(e):
                raise  # not a definiteness failure (e.g. OOM) -- never retry it
            last_error = e
            lam = max(lam * growth, floor)
            continue

        if n_escalations:
            _warn_escalation(site, damping, lam, n_escalations)
        return L, lam, n_escalations

    raise RuntimeError(
        f"damped_cholesky[{site}]: still not positive definite after "
        f"{max_escalations} escalations (damping {damping:.3e} -> {lam:.3e}). "
        f"Last error: {last_error}"
    )


def _warn_escalation(site: str, nominal: float, used: float, n: int) -> None:
    """Print on the 1st, 10th, 100th ... escalation, so long runs stay readable."""
    count = _ESCALATION_COUNTS.get(site, 0) + 1
    _ESCALATION_COUNTS[site] = count
    if count & (count - 1) == 0 or count % 1000 == 0:  # powers of two, then 1000s
        print(
            f"[damped_cholesky:{site}] escalated damping {nominal:.3e} -> "
            f"{used:.3e} ({n} step(s)); {count} escalation(s) so far",
            flush=True,
        )


def escalation_count(site: str = "") -> int:
    """Total escalations recorded for `site` in this process."""
    return _ESCALATION_COUNTS.get(site, 0)


def escalation_total() -> int:
    """Escalations across all call sites in this process.

    Logged per step by `train.py` as `cholesky_escalations`. This is the number
    to quote when reporting how far a run departed from its nominal damping:
    0 means the run is bit-for-bit what the unmodified code would have produced.
    Read it from the metric rather than from the stdout warnings, which are
    rate-limited to powers of two and therefore undercount.
    """
    return sum(_ESCALATION_COUNTS.values())
