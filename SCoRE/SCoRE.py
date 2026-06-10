import numpy as np
from .utility import BH, eBH

# implementation of SCoRE procedures

def _uniform_random(random_state, size=None):
    if random_state is None:
        return np.random.uniform(0, 1, size)
    if isinstance(random_state, np.random.Generator):
        return random_state.uniform(0, 1, size)
    return np.random.default_rng(random_state).uniform(0, 1, size)


def _as_index_array(sel):
    return np.asarray(sel, dtype=int)


def _as_1d_array(name, values):
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return arr


def _split_calib(Dcalib):
    if not isinstance(Dcalib, (tuple, list)) or len(Dcalib) != 2:
        raise ValueError("Dcalib must be a tuple or list of losses and scores (Lcalib, Scalib).")

    Lcalib = _as_1d_array("Lcalib", Dcalib[0])
    Scalib = _as_1d_array("Scalib", Dcalib[1])
    if len(Lcalib) != len(Scalib):
        raise ValueError("The losses and scores (Lcalib, Scalib) must have the same length.")
    return Lcalib, Scalib


def _is_legacy_dtest(Dtest):
    if not isinstance(Dtest, (tuple, list)) or len(Dtest) != 2:
        return False
    if np.ndim(Dtest[1]) == 0:
        return False
    return Dtest[0] is None or np.ndim(Dtest[0]) > 0


def _get_stest(Dtest):
    if _is_legacy_dtest(Dtest):
        Dtest = Dtest[1]
    return _as_1d_array("Dtest", Dtest)


def _validate_binary_loss(Lcalib):
    if not np.all(np.isin(Lcalib, [0, 1])):
        raise ValueError("Conformal selection requires binary calibration losses in {0, 1}.")


def _validate_alpha(alpha):
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    return alpha


def _validate_gamma(gamma):
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma < 0 or gamma > 1:
        raise ValueError("gamma must be in [0, 1]")
    return gamma


def _validate_prune(prune):
    if prune not in (None, "hete", "homo"):
        raise ValueError("prune must be one of None, 'hete', or 'homo'")
    return prune


def CS(Dcalib, Dtest, alpha, mult_test=True, return_pvals=False):
    """Conformal Selection (CS) procedure for binary losses that controls the marginal deployment risk (MDR) or selective deployment risk (SDR).
    Here, MDR reduces to the average type-I error and SDR reduces to the usual false discovery rate (FDR).
    
    The function applies only when the loss function evaluates strictly to {0,1}.
    
    Args:
        Dcalib (tuple): A tuple containing losses and scores (Lcalib, Scalib) for the calibration set.
        Dtest (array-like): Test scores Stest. A legacy tuple/list (ignored, Stest) is also accepted.
        alpha (float): The target error margin.
        mult_test (bool): Whether to perform multiple testing correction using the Benjamini-Hochberg (BH) procedure. If False, MDR is controlled; otherwise SDR is controlled.
        return_pvals (bool): If True, returns the calculated p-values alongside the selected indices.
        
    Returns:
        Union[np.ndarray, tuple]: Selected indices, or (selected indices, p-values) if return_pvals is True.
    """
    alpha = _validate_alpha(alpha)
    Lcalib, Scalib = _split_calib(Dcalib)
    _validate_binary_loss(Lcalib)
    Stest = _get_stest(Dtest)
    Ncalib, Ntest = len(Scalib), len(Stest)

    calib_scores = 1000 * (Lcalib == 0) + Scalib
    test_scores = Stest
    
    pvals = np.zeros(Ntest)
    for j in range(Ntest):
        pvals[j] = (1 + np.sum(calib_scores <= test_scores[j])) / (Ncalib + 1)

    if mult_test:
        sel = BH(pvals, alpha)
    else:
        sel = np.flatnonzero(pvals <= alpha)

    if not return_pvals:
        return _as_index_array(sel)
    return sel, pvals

def SCoRE_MDR_bf(Dcalib, Dtest, alpha, gamma, return_evals=False):
    """Brute-force algorithm for SCoRE testing with Marginal Deployment Risk (MDR) control. The algorithm manually search for a suitable cutoff t.
    Compared to SCoRE_MDR, this brute-force computation enables computing the SCoRE e-values explicitly.
    
    Args:
        Dcalib (tuple): A tuple containing losses and scores (Lcalib, Scalib) for the calibration set.
        Dtest (array-like): Test scores Stest. A legacy tuple/list (ignored, Stest) is also accepted.
        alpha (float): The target error margin.
        gamma (float): A tuning parameter spanning [0, 1]. Recommended value is gamma=alpha.
        return_evals (bool): Whether to output the computed e-values. Defaults to False.
        
    Returns:
        Union[np.ndarray, tuple]: Selected indices, or (selected indices, e-values) if return_evals is True.
    """
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    Lcalib, Scalib = _split_calib(Dcalib)
    Stest = _get_stest(Dtest)
    Ncalib, Ntest = len(Scalib), len(Stest)

    M = list(np.concatenate([Scalib, Stest]))

    def F(j, t, l):
        return (np.sum(Lcalib * (Scalib <= t)) + l * (Stest[j] <= t)) / (Ncalib + 1)
    
    def t_gamma(j, l):
        max_t = -np.inf
        for cur_t in M:
            if F(j, cur_t, l) <= gamma:
                max_t = max(max_t, cur_t)
        return max_t
    
    sel = []
    evalues = np.zeros(Ntest)

    for i_itr in range(Ntest):
        evalue = np.inf
        for l in [0, 1]:
            t_l = t_gamma(i_itr, l)
            num = (Ncalib + 1) * (Stest[i_itr] <= t_l)
            denom = np.sum(Lcalib * (Scalib <= t_l)) + l * (Stest[i_itr] <= t_l)

            evalue = min(evalue, num / denom)
        evalues[i_itr] = evalue

        phi = (evalue >= (1 / alpha))
        if phi == 1:
            sel.append(i_itr)

    if not return_evals:
        return _as_index_array(sel)
    return _as_index_array(sel), evalues

def SCoRE_MDR(Dcalib, Dtest, alpha, gamma):
    """SCoRE testing procedure with Marginal Deployment Risk (MDR) control, implemented using the computational shortcut. Note the e-values are not directly available with this shortcut.
    
    Args:
        Dcalib (tuple): A tuple containing losses and scores (Lcalib, Scalib) for the calibration set.
        Dtest (array-like): Test scores Stest. A legacy tuple/list (ignored, Stest) is also accepted.
        alpha (float): The target error margin. 
        gamma (float): A tuning parameter spanning [0, 1]. Recommended value is gamma=alpha.
        
    Returns:
        list: A list of selected instances with low risk and deemed safe to deploy.
    """
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    Lcalib, Scalib = _split_calib(Dcalib)
    Stest = _get_stest(Dtest)
    Ncalib, Ntest = len(Scalib), len(Stest)

    sel = []

    for i_itr in range(Ntest):
        phi = (1 + np.sum(Lcalib * (Scalib <= Stest[i_itr]))) / (Ncalib + 1) <= gamma

        if gamma > alpha and phi == 1: # need to check the 2nd condition
            M = list(np.concatenate([Scalib, Stest]))
            for t in M:
                upp = (1 + np.sum(Lcalib * (Scalib <= t))) / (Ncalib + 1)
                low = upp - 1 / (Ncalib + 1)

                # check whether (alpha, gamma] and [low, upp] overlap
                if not ((upp <= alpha) or (low > gamma)): # overlap
                    phi = 0
                    break

        if phi == 1: # selected
            sel.append(i_itr)

    return _as_index_array(sel)

def SCoRE_MDR_w(Dcalib, Dtest, wcalib, wtest, alpha, gamma):
    """SCoRE testing procedure with Marginal Deployment Risk (MDR) control under the covariate shift case, implemented using the computational shortcut.
    
    Args:
        Dcalib (tuple): A tuple containing losses and scores (Lcalib, Scalib) for the calibration set.
        Dtest (array-like): Test scores Stest. A legacy tuple/list (ignored, Stest) is also accepted.
        wcalib (np.ndarray): The covariate shift weights for the calibration data.
        wtest (np.ndarray): The covariate shift weights for the test data.
        alpha (float): The target error margin.
        gamma (float): A tuning parameter spanning [0, 1]. Recommended value is gamma=alpha.
        
    Returns:
        list: A list of selected instances with low risk and deemed safe to deploy.
    """
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    Lcalib, Scalib = _split_calib(Dcalib)
    Stest = _get_stest(Dtest)
    wcalib = _as_1d_array("wcalib", wcalib)
    wtest = _as_1d_array("wtest", wtest)
    Ncalib, Ntest = len(Scalib), len(Stest)
    if len(wcalib) != Ncalib:
        raise ValueError("wcalib must have the same length as Lcalib and Scalib.")
    if len(wtest) != Ntest:
        raise ValueError("wtest must have the same length as Stest.")

    sel = []

    calib_w_sum = np.sum(wcalib)
    for i_itr in range(Ntest):
        phi = (wtest[i_itr] + np.sum(wcalib * Lcalib * (Scalib <= Stest[i_itr]))) / (wtest[i_itr] + calib_w_sum) <= gamma

        if gamma > alpha and phi == 1: # need to check the 2nd condition
            M = list(np.concatenate([Scalib, Stest]))
            for t in M:
                upp = (wtest[i_itr] + np.sum(wcalib * Lcalib * (Scalib <= t))) / (wtest[i_itr] + calib_w_sum)
                low = upp - wtest[i_itr] / (wtest[i_itr] + calib_w_sum)

                # check whether (alpha, gamma] and [low, upp] overlap
                if not ((upp <= alpha) or (low > gamma)): # overlap
                    phi = 0
                    break

        if phi == 1: # selected
            sel.append(i_itr)

    return _as_index_array(sel)

######## SDR ########

def SCoRE_SDR(Dcalib, Dtest, alpha, gamma, prune=None, return_evals=False, random_state=None):
    """SCoRE testing procedure for Selective Deployment Risk (SDR) control. Optimized implementation with time complexity $O(m(n+m) + (n+m)\\log(n+m))$.
    
    Args:
        Dcalib (tuple): losses and scores (Lcalib, Scalib) for the calibration set.
        Dtest (array-like): Test scores Stest. A legacy tuple/list (ignored, Stest) is also accepted.
        alpha (float): The target error margin.
        gamma (float): A tuning parameter spanning [0, 1]. Recommended value is gamma=alpha.
        prune (str, optional): Optional boosting strategy (either 'hete' or 'homo'). Use of 'homo' is generally recommended.
        return_evals (bool, optional): Returns computed e-values if True.
        random_state (int or np.random.Generator, optional): Random seed or generator used when pruning is enabled. Randomization is only needed for the boosting strategies.
        
    Returns:
        Union[list, tuple]: Selection set indices, or combined tuple depending on `return_evals`.
    """
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    prune = _validate_prune(prune)
    Lcalib, Scalib = _split_calib(Dcalib)
    Stest = _get_stest(Dtest)
    Ncalib, Ntest = len(Scalib), len(Stest)

    Scalib_tagged = [(lp, l, 'calib') for lp, l in zip(Scalib, Lcalib)]
    Stest_tagged = [(lp, 0, 'test') for lp in Stest] # 0 is dummy value

    M_tagged = Scalib_tagged + Stest_tagged
    M_tagged.sort()

    M = np.array([a[0] for a in M_tagged])

    evalues = np.zeros(Ntest)

    # some intermediate prefix sums
    NUMER = np.zeros(Ncalib + Ntest) # for np.sum(Lcalib * (Scalib <= t)) with t being the i-th ranked value in M
    DENOM = np.zeros(Ncalib + Ntest) # for 1 + np.sum(Stest <= t).
    for i, (t, L, l_type) in enumerate(M_tagged):
        NUMER[i] = (NUMER[i-1] if i != 0 else 0)
        DENOM[i] = (DENOM[i-1] if i != 0 else 1)
        if l_type == 'calib':
            NUMER[i] += L
        else:
            DENOM[i] += 1
            
    # above will have a bug when there are ties in M_tagged. 
    # for example, if M_tagged = [(0.5, 0, 'calib'), (0.5, 1, 'calib')], then NUMER[0] = 0, NUMER[1] = 1.
    # But for t = 0.5, we should have NUMER = 1. So we need to correct for ties.
    for i in range(len(M_tagged) - 2, -1, -1):
        if M_tagged[i][0] == M_tagged[i+1][0]:
            NUMER[i] = NUMER[i+1]
            DENOM[i] = DENOM[i+1]

    for j in range(Ntest):
        # we precompute all FR, t_gamma, and ell
        FR_0 = np.zeros(Ncalib + Ntest)
        FR_1 = np.zeros(Ncalib + Ntest)

        ELL = np.zeros(Ncalib + Ntest)

        # pairs of (i, t)
        t_0, t_1 = (-1, -np.inf), (-1, -np.inf)

        # compute FR and ell
        for i, (t, _, _) in enumerate(M_tagged):
            FR_0[i] = NUMER[i] / (DENOM[i] - (Stest[j] <= t)) / (Ncalib + 1) * Ntest
            FR_1[i] = (NUMER[i] + (Stest[j] <= t)) / (DENOM[i] - (Stest[j] <= t)) / (Ncalib + 1) * Ntest

            ELL[i] = (Ncalib + 1) * gamma / Ntest * (DENOM[i] - (Stest[j] <= t)) - NUMER[i]

        # compute t_gamma. Also store the original ranking i
        for i, t in enumerate(M):
            if FR_0[i] <= gamma:
                t_0 = (i, t)
            if FR_1[i] <= gamma:
                t_1 = (i, t)

        if Stest[j] > t_1[1]:
            continue # e-value is zero

        if t_1[1] == t_0[1]:
            evalues[j] = (Ncalib + 1) / (1 + NUMER[t_1[0]])
            continue # same upper/lower bound case

        max_ell = np.zeros(Ntest + Ncalib) # max_ell[rank(t)]: max of l(t') with t' > t, t' in M, and FR(t', 0) <= gamma. 
                                           # max_ell[0] correspond to the smallest t in M, max_ell[-1] correspond to the largest t in M.
        last_max = -np.inf
        for i, t in zip(range(Ntest + Ncalib - 1, -1, -1), reversed(M)): # n+m iterations
            max_ell[i] = last_max

            if FR_0[i] <= gamma:
                last_max = max(last_max, ELL[i]) # both O(n+m)

        M_star = [] # store pairs of (i, t)
        for i, t in enumerate(M):
            if t < max(Stest[j], t_1[1]):
                continue # this is to keep the index i
            if t > t_0[1]:
                break

            if FR_0[i] <= gamma and ELL[i] > max_ell[i]:
                M_star.append((i, t))

        evalue = np.inf
        for i, t in M_star:
            cur_val = (Ncalib + 1) / (ELL[i] + NUMER[i])
            evalue = min(evalue, cur_val)
        
        evalues[j] = evalue
    
    if prune == 'hete':
        evalues /= _uniform_random(random_state, len(evalues))
    if prune == 'homo':
        evalues /= _uniform_random(random_state)
    sel = eBH(evalues, alpha)
    
    if not return_evals:
        return _as_index_array(sel)
    return sel, evalues

def SCoRE_SDR_w(Dcalib, Dtest, wcalib, wtest, alpha, gamma, prune=None, return_evals=False, random_state=None):
    """SCoRE testing procedure for Selective Deployment Risk (SDR) control under the covariate shift case. Optimized implementation with time complexity $O(m(n+m) + (n+m)\\log(n+m))$.
    
    Args:
        Dcalib (tuple): losses and scores (Lcalib, Scalib) for the calibration set.
        Dtest (array-like): Test scores Stest. A legacy tuple/list (ignored, Stest) is also accepted.
        wcalib (np.ndarray): The covariate shift weights for the calibration data.
        wtest (np.ndarray): The covariate shift weights for the test data.
        alpha (float): The target error margin.
        gamma (float): A tuning parameter spanning [0, 1]. Recommended value is gamma=alpha.
        prune (str, optional): Optional boosting strategy (either 'hete' or 'homo'). Use of 'homo' is generally recommended.
        return_evals (bool, optional): Returns computed e-values if True.
        random_state (int or np.random.Generator, optional): Random seed or generator used when pruning is enabled.
        
    Returns:
        Union[list, tuple]: Selection set indices, or combined tuple depending on `return_evals`.
    """
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    prune = _validate_prune(prune)
    Lcalib, Scalib = _split_calib(Dcalib)
    Stest = _get_stest(Dtest)
    wcalib = _as_1d_array("wcalib", wcalib)
    wtest = _as_1d_array("wtest", wtest)
    Ncalib, Ntest = len(Scalib), len(Stest)
    if len(wcalib) != Ncalib:
        raise ValueError("wcalib must have the same length as Lcalib and Scalib.")
    if len(wtest) != Ntest:
        raise ValueError("wtest must have the same length as Stest.")

    Scalib_tagged = [(lp, l, w, 'calib') for lp, l, w in zip(Scalib, Lcalib, wcalib)]
    Stest_tagged = [(lp, 0, w, 'test') for lp, w in zip(Stest, wtest)] # 0 is dummy value

    M_tagged = Scalib_tagged + Stest_tagged
    M_tagged.sort()

    M = np.array([a[0] for a in M_tagged])

    evalues = np.zeros(Ntest)

    calib_w_sum = np.sum(wcalib)

    # some intermediate prefix sums
    NUMER = np.zeros(Ncalib + Ntest) # for np.sum(wcalib * Lcalib * (Scalib <= t)) with t being the i-th ranked value in M
    DENOM = np.zeros(Ncalib + Ntest) # for 1 + np.sum(Stest <= t).
    for i, (t, L, w, l_type) in enumerate(M_tagged):
        NUMER[i] = (NUMER[i-1] if i != 0 else 0)
        DENOM[i] = (DENOM[i-1] if i != 0 else 1)
        if l_type == 'calib':
            NUMER[i] += w * L
        else:
            DENOM[i] += 1

    # Correction for ties
    for i in range(len(M_tagged) - 2, -1, -1):
        if M_tagged[i][0] == M_tagged[i+1][0]:
            NUMER[i] = NUMER[i+1]
            DENOM[i] = DENOM[i+1]

    for j in range(Ntest):
        # we precompute all FR, t_gamma, and ell
        FR_0 = np.zeros(Ncalib + Ntest)
        FR_1 = np.zeros(Ncalib + Ntest)

        ELL = np.zeros(Ncalib + Ntest)

        # pairs of (i, t)
        t_0, t_1 = (-1, -np.inf), (-1, -np.inf)

        # compute FR and ell
        for i, (t, _, _, _) in enumerate(M_tagged):
            FR_0[i] = NUMER[i] / (DENOM[i] - (Stest[j] <= t)) / (calib_w_sum + wtest[j]) * Ntest
            FR_1[i] = (NUMER[i] + wtest[j] * (Stest[j] <= t)) / (DENOM[i] - (Stest[j] <= t)) / (calib_w_sum + wtest[j]) * Ntest

            ELL[i] = (calib_w_sum + wtest[j]) / wtest[j] * gamma / Ntest * (DENOM[i] - (Stest[j] <= t)) - NUMER[i] / wtest[j]

        # compute t_gamma. Also store the original ranking i
        for i, t in enumerate(M):
            if FR_0[i] <= gamma:
                t_0 = (i, t)
            if FR_1[i] <= gamma:
                t_1 = (i, t)

        if Stest[j] > t_1[1]:
            continue # e-value is zero

        if t_1[1] == t_0[1]:
            evalues[j] = (calib_w_sum + wtest[j]) / (wtest[j] + NUMER[t_1[0]])
            continue # same upper/lower bound case

        max_ell = np.zeros(Ntest + Ncalib) # max_ell[rank(t)]: max of l(t') with t' > t, t' in M, and FR(t', 0) <= gamma. 
                                           # max_ell[0] correspond to the smallest t in M, max_ell[-1] correspond to the largest t in M.
        last_max = -np.inf
        for i, t in zip(range(Ntest + Ncalib - 1, -1, -1), reversed(M)): # n+m iterations
            max_ell[i] = last_max

            if FR_0[i] <= gamma:
                last_max = max(last_max, ELL[i]) # both O(n+m)

        M_star = [] # store pairs of (i, t)
        for i, t in enumerate(M):
            if t < max(Stest[j], t_1[1]):
                continue # this is to keep the index i
            if t > t_0[1]:
                break

            if FR_0[i] <= gamma and ELL[i] > max_ell[i]:
                M_star.append((i, t))

        evalue = np.inf
        for i, t in M_star:
            cur_val = (calib_w_sum + wtest[j]) / (wtest[j] * ELL[i] + NUMER[i])
            evalue = min(evalue, cur_val)
        
        evalues[j] = evalue
    
    if prune == 'hete':
        evalues /= _uniform_random(random_state, len(evalues))
    if prune == 'homo':
        evalues /= _uniform_random(random_state)
    sel = eBH(evalues, alpha)
    
    if not return_evals:
        return _as_index_array(sel)
    return sel, evalues
