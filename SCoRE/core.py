"""Risk-adjusted e-values for the cost-based SCoRE construction.

Costs normalize selected loss; rho weights correct covariate shift. They play
different roles and must not be multiplied together in the cost denominator.
"""

import numpy as np


def _vector(name, values, length=None, positive=False, nonnegative=False):
    # convert to numpy array with length/sign checks.
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or (length is not None and len(values) != length):
        raise ValueError(f"{name} must be a one-dimensional array of length {length}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values.")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must contain strictly positive values.")
    if nonnegative and np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative values.")
    return values


def _optional_vector(name, values, length):
    # supply a vector of ones if values is None.
    return np.ones(length) if values is None else _vector(name, values, length, positive=True)


def _loss_support(Lbar_t, finite_t, m):
    # Normalize the possible positive test losses into (bounds, values).
    # Return (bounds, None) for interval support (0, bound_j], or (None, values) if the support is discrete.
    if Lbar_t is not None and finite_t is not None:
        raise ValueError("Specify only one of Lbar_t and finite_t.")
    if finite_t is not None:
        values = np.asarray(finite_t, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != m:
            raise ValueError("finite_t must have shape (m,) or (m, K).")
        present = ~np.isnan(values)
        if np.any(~np.isfinite(values[present])) or np.any(values[present] <= 0):
            raise ValueError("finite_t must contain positive finite losses or NaN padding.")
        return None, values
    bounds = np.asarray(1.0 if Lbar_t is None else Lbar_t, dtype=float)
    if bounds.ndim == 0:
        bounds = np.full(m, bounds.item())
    return _vector("Lbar_t", bounds, m, nonnegative=True), None


def _prepare(Lc, Sc, St, rho_c, rho_t, Lbar_t, finite_t):
    Sc = _vector("Sc", Sc)
    St = _vector("St", St)
    Lc = _vector("Lc", Lc, len(Sc), nonnegative=True)
    if not len(Sc):
        raise ValueError("At least one calibration unit is required.")
    rho_c = _optional_vector("rho_c", rho_c, len(Sc))
    rho_t = _optional_vector("rho_t", rho_t, len(St))
    bounds, values = _loss_support(Lbar_t, finite_t, len(St))
    return Lc, Sc, St, rho_c, rho_t, bounds, values


def _prefix_on_grid(scores, weights, grid):
    # Compute sum_i weights_i 1{scores_i <= t}, including every score tie.
    order = np.argsort(scores, kind="stable")
    cumulative = np.concatenate(([0.0], np.cumsum(weights[order])))
    return cumulative[np.searchsorted(scores[order], grid, side="right")]


def _last_true(mask):
    # Last feasible grid index per row, or -1 when no cutoff is feasible.
    last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    return np.where(mask.any(axis=1), last, -1)


def _evalues(prepared, gamma, cc=None, ct=None, block=512, selective=True):
    # explicitly computes the SCoRE evalue (for either the selective WSR case or MSR case) via grid search.
    Lc, Sc, St, rho_c, rho_t, bounds, values = prepared
    gamma = float(gamma)
    if isinstance(block, (bool, np.bool_)) or not isinstance(block, (int, np.integer)) or block <= 0:
        raise ValueError("block must be a positive integer.")
    n, m = len(Sc), len(St)
    if selective:
        cc = _optional_vector("cc", cc, n)
        ct = _optional_vector("ct", ct, m)
    if m == 0:
        return np.empty(0, dtype=float)

    grid = np.unique(np.concatenate((Sc, St)))
    loss_sum = _prefix_on_grid(Sc, rho_c * Lc, grid)
    if selective:
        cost_sum = _prefix_on_grid(St, ct, grid)
    total_weight = rho_c.sum() + rho_t
    result = np.zeros(m)

    # Only the batch loop and (for finite supports) support-column loop remain.
    # Grid calculations are vectorized, with O(block * (n + m)) working memory.
    for start in range(0, m, block):
        stop = min(start + block, m)
        rows = np.arange(stop - start)
        included = St[start:stop, None] <= grid
        weight = rho_t[start:stop]
        total = total_weight[start:stop]
        if selective:
            baseline = np.minimum(cc.min(), ct[start:stop])
            denominator = (baseline[:, None] + cost_sum
                           - ct[start:stop, None] * included)
            budget = (gamma * total / m)[:, None] * denominator
        else:
            budget = np.broadcast_to((gamma * total)[:, None], included.shape)

        def at_loss(loss):
            feasible = loss_sum + (weight * loss)[:, None] * included <= budget
            index = _last_true(feasible)
            accepted = (index >= 0) & included[rows, index]
            return index, accepted

        if bounds is not None:
            upper = bounds[start:stop]
            last_upper, accepted = at_loss(upper)
            # Strict inequality: the interval is (0, upper], excluding zero.
            last_positive = _last_true(loss_sum < budget)
            denom = weight * upper + loss_sum[last_upper]
            # If positive losses reach a later cutoff than the upper bound, that last cutoff has capacity in (0, upper). 
            # At its capacity, the e-value denominator equals budget. Beyond the test score, selected cost (and hence budget) is nondecreasing, 
            # so this is the largest attained denominator. Otherwise use the upper bound.
            changed = last_positive != last_upper
            denom = np.where(changed, budget[rows, last_positive], denom)
            evalues = np.divide(total, denom, out=np.zeros_like(total),
                                where=accepted & (upper > 0) & (denom > 0))
            evalues[upper == 0] = np.inf  # empty positive attainable set
        else:
            evalues = np.full(stop - start, np.inf)
            for loss in values[start:stop].T:
                present = ~np.isnan(loss)
                loss = np.where(present, loss, 0.0)
                index, accepted = at_loss(loss)
                denom = weight * loss + loss_sum[index]
                current = np.divide(total, denom, out=np.zeros_like(total),
                                    where=present & accepted & (denom > 0))
                current[~present] = np.inf
                evalues = np.minimum(evalues, current)
        result[start:stop] = evalues
    return result


def score_evalues(Lc, Sc, St, gamma, cc=None, ct=None, rho_c=None, rho_t=None,
                  Lbar_t=None, finite_t=None, block=512):
    """Compute cost-aware SCoRE e-values, optionally under covariate shift.

    Lc and Sc are calibration losses and scores (length n); St contains m test
    scores. Smaller scores favor selection. Losses must be nonnegative; costs
    cc/ct and shift weights rho_c/rho_t must be positive and default to ones.
    gamma is a finite positive tuning level (zero is accepted for compatibility).

    Use Lbar_t (scalar or length m) for attainable losses (0, Lbar_t[j]], or
    finite_t of shape (m, K) for positive attainable losses, padded with NaN.
    A length-m finite_t specifies one positive loss per unit. The default is
    (0, 1]; binary losses require finite_t=np.ones(m) explicitly. A zero bound
    or an all-NaN/empty support row denotes a surely zero loss and returns inf.
    Bounds/supports must cover all possible test losses, not observed labels.

    The cutoff uses h_j=min(cc, ct[j]) and the cost of other test units below
    it. Returned e-values do not yet include randomized boosting. Computation
    uses batches of at most block test units to bound working memory.
    """
    prepared = _prepare(Lc, Sc, St, rho_c, rho_t, Lbar_t, finite_t)
    return _evalues(prepared, gamma, cc, ct, block)


def cost_ebh(E, c, alpha):
    """Largest decreasing-e-value prefix with E[k] >= m/(alpha * sum(c[:k])).

    Costs are positive and default to one when c is None. Returns indices in
    decreasing e-value order, with stable tie breaking, as in the original eBH.
    """
    E = np.asarray(E, dtype=float)
    if E.ndim != 1 or np.any(np.isnan(E)) or np.any(E < 0):
        raise ValueError("E must be a one-dimensional array of nonnegative e-values.")
    c = _optional_vector("c", c, len(E))
    alpha = float(alpha)
    order = np.argsort(-E, kind="stable")
    thresholds = len(E) / (alpha * np.cumsum(c[order]))
    feasible = np.flatnonzero(E[order] >= thresholds)
    return order[:feasible[-1] + 1] if feasible.size else np.empty(0, dtype=int)
