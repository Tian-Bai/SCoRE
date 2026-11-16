"""SCoRE procedures for marginal, selective, and weighted selection risk."""

import numpy as np

from .core import _evalues, _optional_vector, _prepare, _prefix_on_grid, score_evalues
from .utility import BH, eBH


def _uniform_random(random_state, size=None):
    if random_state is None:
        return np.random.uniform(0, 1, size)
    if isinstance(random_state, np.random.Generator):
        return random_state.uniform(0, 1, size)
    return np.random.default_rng(random_state).uniform(0, 1, size)


def _as_1d_array(name, values):
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return arr


def _split_calib(D_calib):
    if not isinstance(D_calib, (tuple, list)) or len(D_calib) not in (2, 3):
        raise ValueError("D_calib must be (L, S) or (L, S, c).")
    losses = _as_1d_array("Calibration losses", D_calib[0])
    scores = _as_1d_array("Calibration scores", D_calib[1])
    if len(losses) != len(scores):
        raise ValueError("Calibration losses and scores must have the same length.")
    costs = _optional_vector("Calibration costs", D_calib[2] if len(D_calib) == 3 else None,
                             len(scores))
    return losses, scores, costs


def _split_test(D_test):
    # Plain lists/tuples of two or three scalar scores remain score vectors.
    # The loss field is deliberately never read: test outcomes are unknown.
    costs = None
    if isinstance(D_test, (tuple, list)) and len(D_test) in (2, 3) and np.ndim(D_test[1]) > 0:
        costs = D_test[2] if len(D_test) == 3 else None
        D_test = D_test[1]
    scores = _as_1d_array("Test scores", D_test)
    return scores, _optional_vector("Test costs", costs, len(scores))


def CS(D_calib, D_test, alpha, mult_test=True, return_pvals=False):
    """Conformal selection for binary losses, with BH for SDR or testing for MSR.

    Both datasets accept (L, S) or (L, S, c); costs and test losses are unused.
    D_test may also be a score vector. Returns selected indices and, if
    requested, the p-values.
    """
    alpha = float(alpha)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    Lcalib, Scalib, _ = _split_calib(D_calib)
    if not np.all(np.isin(Lcalib, [0, 1])):
        raise ValueError("Conformal selection requires binary calibration losses in {0, 1}.")
    Stest, _ = _split_test(D_test)
    # Excluding zero-loss units implements infinite clipped scores without the old arbitrary +1000 offset, and handles any finite score scale.
    bad_scores = np.sort(Scalib[Lcalib == 1])
    pvals = (1 + np.searchsorted(bad_scores, Stest, side="right")) / (len(Scalib) + 1)
    selected = BH(pvals, alpha) if mult_test else np.flatnonzero(pvals <= alpha)
    return (selected, pvals) if return_pvals else selected


def _msr(D_calib, D_test, alpha, gamma, wcalib=None, wtest=None,
         Lbar_t=None, finite_t=None, return_evals=False, block=512):
    alpha = float(alpha)
    gamma = float(gamma)
    Lcalib, Scalib, _ = _split_calib(D_calib)
    Stest, _ = _split_test(D_test)
    prepared = _prepare(Lcalib, Scalib, Stest, wcalib, wtest,
                        Lbar_t, finite_t)
    Lc, Sc, St, wc, wt, bounds, values = prepared
    if gamma <= alpha and not return_evals:
        # Marginal-risk shortcut: only the largest attainable loss and the
        # calibration loss prefix at each test score are needed.
        if bounds is None:
            bounds = np.max(np.where(np.isnan(values), 0.0, values), axis=1, initial=0.0)
        loss_sum = _prefix_on_grid(Sc, wc * Lc, St)
        feasible = wt * bounds + loss_sum <= gamma * (wc.sum() + wt)
        return np.flatnonzero(feasible | (bounds == 0))
    evalues = _evalues(prepared, gamma, block=block, selective=False)
    selected = np.flatnonzero(evalues >= 1 / alpha)
    return (selected, evalues) if return_evals else selected


def SCoRE_MSR(D_calib, D_test, alpha, gamma, *, Lbar_t=None, finite_t=None):
    """Control marginal selection risk: expected selected loss divided by m.

    D_calib and D_test accept (L, S) or (L, S, c). Test losses and costs are
    unused; D_test also accepts a score vector. Lbar_t and finite_t specify
    attainable losses as in score_evalues;
    the default is (0, 1]. alpha must be positive and gamma nonnegative.
    For gamma <= alpha, uses the sorted-prefix computational shortcut.
    Costs do not normalize marginal risk; cost-scaled losses can be supplied
    directly with their corresponding bounds or finite supports.
    """
    return _msr(D_calib, D_test, alpha, gamma, Lbar_t=Lbar_t, finite_t=finite_t)


def SCoRE_MSR_bf(D_calib, D_test, alpha, gamma, return_evals=False, *,
                 Lbar_t=None, finite_t=None, block=512):
    """Compute marginal SCoRE e-values explicitly.

    Uses the exact positive attainable-loss infimum rather than the former
    endpoint approximation. See SCoRE_MSR for inputs. If return_evals is True,
    returns (selected indices, e-values); otherwise returns selected indices.
    """
    selected, evalues = _msr(D_calib, D_test, alpha, gamma, Lbar_t=Lbar_t,
                            finite_t=finite_t, return_evals=True, block=block)
    return (selected, evalues) if return_evals else selected


def SCoRE_MSR_w(D_calib, D_test, wcalib, wtest, alpha, gamma, *,
                Lbar_t=None, finite_t=None):
    """SCoRE_MSR with positive calibration/test covariate-shift weights.

    wcalib and wtest have the respective sample lengths. Attainable-loss
    options and return conventions are the same as SCoRE_MSR.
    """
    return _msr(D_calib, D_test, alpha, gamma, wcalib, wtest, Lbar_t, finite_t)


def _selective(D_calib, D_test, alpha, gamma, prune, return_evals, random_state,
               Lbar_t, finite_t, block, wcalib=None, wtest=None, use_costs=True):
    alpha = float(alpha)
    if prune not in (None, "hete", "homo"):
        raise ValueError("prune must be one of None, 'hete', or 'homo'")
    Lcalib, Scalib, ccalib = _split_calib(D_calib)
    Stest, ctest = _split_test(D_test)
    if not use_costs:
        ccalib = ctest = None
    evalues = score_evalues(Lcalib, Scalib, Stest, gamma,
                           cc=ccalib, ct=ctest, rho_c=wcalib, rho_t=wtest,
                           Lbar_t=Lbar_t, finite_t=finite_t, block=block)
    if prune is not None:
        size = len(evalues) if prune == "hete" else None
        uniforms = _uniform_random(random_state, size)
        # Preserve zero e-values even in the possible RNG endpoint draw U=0.
        with np.errstate(divide="ignore"):
            evalues = np.divide(evalues, uniforms, out=np.zeros_like(evalues),
                                where=evalues > 0)
    selected = eBH(evalues, alpha, costs=ctest)
    return (selected, evalues) if return_evals else selected


def SCoRE_WSR(D_calib, D_test, alpha, gamma, prune=None, return_evals=False,
              random_state=None, *, Lbar_t=None, finite_t=None, block=512):
    """Control weighted selection risk using the supplied per-unit costs.

    D_calib and D_test accept (L, S, c) or (L, S), with omitted costs set to
    ones independently for each dataset. Costs must be positive vectors of
    the corresponding sample length. Test losses are ignored: use None when
    unavailable. D_test also accepts a score vector with unit costs.
    The controlled risk is E[sum(selected losses) / sum(selected costs)],
    with empty risk 0. Unit costs recover SDR.
    Costs do not automatically multiply the losses: supply cost-scaled losses
    yourself when appropriate. alpha > 0 and gamma >= 0 need not be <= 1.

    Lbar_t is a scalar or length-m bound for (0, Lbar_t[j]]; finite_t is an
    (m,) or (m, K) array of positive attainable losses with NaN padding. Pass
    only one; the default is (0, 1]. See score_evalues for empty support rules.
    block controls the maximum number of test units processed together.

    prune='hete' divides by independent uniforms; 'homo' uses one uniform.
    random_state accepts a seed or NumPy Generator (None uses np.random).
    Returns integer selected indices, or (indices, e-values) if return_evals.
    As before, returned e-values include boosting when prune is enabled.
    """
    return _selective(D_calib, D_test, alpha, gamma, prune, return_evals, random_state,
                      Lbar_t, finite_t, block)


def SCoRE_WSR_w(D_calib, D_test, wcalib, wtest, alpha, gamma, prune=None,
                return_evals=False, random_state=None, *, Lbar_t=None,
                finite_t=None, block=512):
    """SCoRE_WSR under covariate shift.

    Positive wcalib/wtest are density-ratio weights with the corresponding
    sample lengths. They reweight calibration losses and the e-value
    normalization, whereas the costs in the data tuples normalize selected
    cost. All other options and return conventions are those of SCoRE_WSR.
    """
    return _selective(D_calib, D_test, alpha, gamma, prune, return_evals, random_state,
                      Lbar_t, finite_t, block, wcalib, wtest)


def SCoRE_SDR(D_calib, D_test, alpha, gamma, prune=None, return_evals=False,
              random_state=None, *, Lbar_t=None, finite_t=None, block=512):
    """Control SDR: expected average loss among selected units.

    This is the unit-cost specialization of SCoRE_WSR. Both datasets accept
    (L, S) or (L, S, c), but supplied costs are ignored in this procedure.
    Test losses are ignored as well; D_test may also be a score vector.
    Use SCoRE_WSR to normalize by supplied costs. Loss-support, boosting,
    batching, and return options are documented in SCoRE_WSR.
    """
    return _selective(D_calib, D_test, alpha, gamma, prune, return_evals, random_state,
                      Lbar_t, finite_t, block, use_costs=False)


def SCoRE_SDR_w(D_calib, D_test, wcalib, wtest, alpha, gamma, prune=None,
                return_evals=False, random_state=None, *, Lbar_t=None,
                finite_t=None, block=512):
    """SCoRE_SDR with positive calibration/test covariate-shift weights.

    Unit costs are used regardless of costs in the data tuples. wcalib and
    wtest are density-ratio weights, not costs. See SCoRE_SDR for other inputs.
    """
    return _selective(D_calib, D_test, alpha, gamma, prune, return_evals, random_state,
                      Lbar_t, finite_t, block, wcalib, wtest, use_costs=False)


# Compatibility aliases for existing experiment imports. Prefer MSR names.
SCoRE_MDR = SCoRE_MSR
SCoRE_MDR_bf = SCoRE_MSR_bf
SCoRE_MDR_w = SCoRE_MSR_w
