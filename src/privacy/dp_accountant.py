"""
Renyi Differential Privacy (RDP) accountant.

Implements subsampled Gaussian mechanism RDP accounting
and optimal RDP-to-(eps,delta)-DP conversion.
"""

import math
from typing import Sequence, Tuple


# Default RDP orders — denser grid for tighter bounds
DEFAULT_ORDERS = (
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 16.0, 20.0, 32.0, 64.0, 128.0, 256.0,
)


def _rdp_gaussian(alpha: float, sigma: float) -> float:
    """RDP of the Gaussian mechanism at order alpha."""
    return alpha / (2.0 * sigma ** 2)


def _rdp_subsampled_gaussian(alpha: float, sigma: float, q: float) -> float:
    """RDP of the subsampled Gaussian mechanism (Poisson sampling)."""
    if q >= 1.0:
        return _rdp_gaussian(alpha, sigma)
    if q == 0.0:
        return 0.0
    if alpha <= 1:
        return 0.0
    rdp_full = _rdp_gaussian(alpha, sigma)
    exponent = (alpha - 1) * rdp_full
    # For large exponents, use the asymptotic bound: log(q * exp(x)) / (a-1)
    # which avoids math.exp overflow while remaining a valid upper bound.
    if exponent > 500:
        return (math.log(q) + exponent) / (alpha - 1)
    return math.log1p(q * (math.exp(exponent) - 1)) / (alpha - 1)


def compute_epsilon(sigma: float, q: float, steps: int, delta: float,
                    orders: Sequence[float] = DEFAULT_ORDERS) -> float:
    """Convert RDP guarantees to (eps, delta)-DP.

    Uses the optimal conversion: eps = min_alpha { steps * rdp(alpha) + log(1/delta)/(alpha-1) }
    """
    best_eps = float("inf")
    for a in orders:
        if a <= 1:
            continue
        rdp = steps * _rdp_subsampled_gaussian(a, sigma, q)
        eps = rdp + math.log(1.0 / delta) / (a - 1.0)
        best_eps = min(best_eps, eps)
    return best_eps


def find_sigma(target_eps: float, delta: float, q: float, steps: int,
               orders: Sequence[float] = DEFAULT_ORDERS) -> float:
    """Binary search for the noise multiplier sigma that achieves target_eps."""
    lo, hi = 0.01, 500.0
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if compute_epsilon(mid, q, steps, delta, orders) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def compute_epsilon_per_epoch(sigma: float, q: float, steps_per_epoch: int,
                              n_epochs: int, delta: float,
                              orders: Sequence[float] = DEFAULT_ORDERS):
    """Return a list of cumulative epsilon after each epoch."""
    epsilons = []
    for ep in range(1, n_epochs + 1):
        total_steps = ep * steps_per_epoch
        epsilons.append(compute_epsilon(sigma, q, total_steps, delta, orders))
    return epsilons
