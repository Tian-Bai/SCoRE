import numpy as np
import pandas as pd
from .utility import gen_data_Jin2023, BH, eBH, eval_MDR, eval_SDR

# implementation of SCoRE procedures


def CS(Dcalib, Dtest, alpha, mult_test=True, return_pvals=False): # only when L in {0,1}
    Lcalib, Lcalib_pred = Dcalib
    Ltest, Ltest_pred = Dtest # True Ltest should not be used
    Ncalib, Ntest = len(Lcalib_pred), len(Ltest_pred)

    calib_scores = 1000 * (Lcalib == 0) + Lcalib_pred
    test_scores = Ltest_pred
    
    pvals = np.zeros(Ntest)
    for j in range(Ntest):
        pvals[j] = (1 + np.sum(calib_scores <= test_scores[j])) / (Ncalib + 1)

    if mult_test:
        sel = BH(pvals, alpha)
    else:
        sel = [j for j in range(Ntest) if pvals[j] <= alpha]

    if not return_pvals:
        return sel
    return sel, pvals

def SCoRE_MDR_bf(Dcalib, Dtest, alpha, gamma, return_evals=False): # MDR case, brute force
    Lcalib, Lcalib_pred = Dcalib
    Ltest, Ltest_pred = Dtest # True Ltest should not be used
    Ncalib, Ntest = len(Lcalib_pred), len(Ltest_pred)

    M = list(np.concatenate([Lcalib_pred, Ltest_pred]))

    def F(j, t, l):
        return (np.sum(Lcalib * (Lcalib_pred <= t)) + l * (Ltest_pred[j] <= t)) / (Ncalib + 1)
    
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
            num = (Ncalib + 1) * (Ltest_pred[i_itr] <= t_l)
            denom = np.sum(Lcalib * (Lcalib_pred <= t_l)) + l * (Ltest_pred[i_itr] <= t_l)

            evalue = min(evalue, num / denom)
        evalues[i_itr] = evalue

        phi = (evalue >= (1 / alpha))
        if phi == 1:
            sel.append(i_itr)

    if not return_evals:
        return sel
    return sel, evalues

def SCoRE_MDR(Dcalib, Dtest, alpha, gamma):
    Lcalib, Lcalib_pred = Dcalib
    Ltest, Ltest_pred = Dtest # True Ltest should not be used
    Ncalib, Ntest = len(Lcalib_pred), len(Ltest_pred)

    sel = []

    for i_itr in range(Ntest):
        phi = (1 + np.sum(Lcalib * (Lcalib_pred <= Ltest_pred[i_itr]))) / (Ncalib + 1) <= gamma

        if gamma > alpha and phi == 1: # need to check the 2nd condition
            M = list(np.concatenate([Lcalib_pred, Ltest_pred]))
            for t in M:
                upp = (1 + np.sum(Lcalib * (Lcalib_pred <= t))) / (Ncalib + 1)
                low = upp - 1 / (Ncalib + 1)

                # check whether (alpha, gamma] and [low, upp] overlap
                if not ((upp <= alpha) or (low > gamma)): # overlap
                    phi = 0
                    break

        if phi == 1: # selected
            sel.append(i_itr)

    return sel

def SCoRE_MDR_w(Dcalib, Dtest, wcalib, wtest, alpha, gamma):
    Lcalib, Lcalib_pred = Dcalib
    Ltest, Ltest_pred = Dtest # True Ltest should not be used
    Ncalib, Ntest = len(Lcalib_pred), len(Ltest_pred)

    sel = []

    calib_w_sum = np.sum(wcalib)
    for i_itr in range(Ntest):
        phi = (wtest[i_itr] + np.sum(wcalib * Lcalib * (Lcalib_pred <= Ltest_pred[i_itr]))) / (wtest[i_itr] + calib_w_sum) <= gamma

        if gamma > alpha and phi == 1: # need to check the 2nd condition
            M = list(np.concatenate([Lcalib_pred, Ltest_pred]))
            for t in M:
                upp = (wtest[i_itr] + np.sum(wcalib * Lcalib * (Lcalib_pred <= t))) / (wtest[i_itr] + calib_w_sum)
                low = upp - wtest[i_itr] / (wtest[i_itr] + calib_w_sum)

                # check whether (alpha, gamma] and [low, upp] overlap
                if not ((upp <= alpha) or (low > gamma)): # overlap
                    phi = 0
                    break

        if phi == 1: # selected
            sel.append(i_itr)

    return sel

######## SDR ########

def SCoRE_SDR_bin(Dcalib, Dtest, alpha, gamma, prune=None, oracle=False, return_evals=False): # only when L in {0,1}. Equivalent to CS
    Lcalib, Lcalib_pred = Dcalib
    Ltest, Ltest_pred = Dtest # True Ltest should not be used
    Ncalib, Ntest = len(Lcalib_pred), len(Ltest_pred)

    M = list(np.concatenate([Lcalib_pred, Ltest_pred]))

    def FR(j, t, l):
        calib_avg_loss = (l * (Ltest_pred[j] <= t) + np.sum(Lcalib * (Lcalib_pred <= t))) / (Ncalib + 1)
        test_avg_sel = (1 + np.sum(Ltest_pred <= t) - (Ltest_pred[j] <= t)) / Ntest
        return calib_avg_loss / test_avg_sel

    def FR_hat(j, t, l):
        calib_avg_loss = (l + np.sum(Lcalib * (Lcalib_pred <= t))) / (Ncalib + 1)
        test_avg_sel = (1 + np.sum(Ltest_pred <= t) - (Ltest_pred[j] <= t)) / Ntest
        return calib_avg_loss / test_avg_sel

    def t_gamma(j, l):
        max_t = -np.inf
        for cur_t in M:
            if FR(j, cur_t, l) <= gamma:
                max_t = max(max_t, cur_t)
        return max_t

    def ell(j, t):
        return (Ncalib + 1) * gamma / Ntest * (1 + np.sum(Ltest_pred <= t) - (Ltest_pred[j] <= t)) - np.sum(Lcalib * (Lcalib_pred <= t))

    evalues = np.zeros(Ntest)

    for j in range(Ntest):
        if not oracle:
            t_1 = t_gamma(j, 1)
        else:
            t_1 = t_gamma(j, Ltest[j])

        evalue_prime = (Ncalib + 1) * (Ltest_pred[j] <= t_1) / ((Ltest_pred[j] <= t_1) + np.sum(Lcalib * (Lcalib_pred <= t_1)))
        evalues[j] = evalue_prime

    if prune == 'hete':
        evalues /= np.random.uniform(0, 1, len(evalues))
    if prune == 'homo':
        evalues /= np.random.uniform(0, 1)
    sel = eBH(evalues, alpha)

    if not return_evals:
        return sel
    return sel, evalues

def SCoRE_SDR(Dcalib, Dtest, alpha, gamma, prune=None, oracle=False, return_evals=False):
    Lcalib, Lcalib_pred = Dcalib
    Ltest, Ltest_pred = Dtest # True Ltest should not be used
    Ncalib, Ntest = len(Lcalib_pred), len(Ltest_pred)

    M = list(np.concatenate([Lcalib_pred, Ltest_pred])) # (ordered) set of all predictions
    M.sort()

    def FR(j, t, l): # O(n+m)
        calib_avg_loss = (l * (Ltest_pred[j] <= t) + np.sum(Lcalib * (Lcalib_pred <= t))) / (Ncalib + 1)
        test_avg_sel = (1 + np.sum(Ltest_pred <= t) - (Ltest_pred[j] <= t)) / Ntest
        return calib_avg_loss / test_avg_sel

    def FR_hat(j, t, l):
        calib_avg_loss = (l + np.sum(Lcalib * (Lcalib_pred <= t))) / (Ncalib + 1)
        test_avg_sel = (1 + np.sum(Ltest_pred <= t) - (Ltest_pred[j] <= t)) / Ntest
        return calib_avg_loss / test_avg_sel

    def t_gamma(j, l): # O((n+m)^2)
        max_t = -np.inf
        for cur_t in M:
            if FR(j, cur_t, l) <= gamma:
                max_t = max(max_t, cur_t)
        return max_t

    def ell(j, t): # O(n+m)
        return (Ncalib + 1) * gamma / Ntest * (1 + np.sum(Ltest_pred <= t) - (Ltest_pred[j] <= t)) - np.sum(Lcalib * (Lcalib_pred <= t))

    evalues = np.zeros(Ntest)

    for j in range(Ntest): # m iterations
        if not oracle:
            t_1 = t_gamma(j, 1)
            t_0 = t_gamma(j, 0)

            if Ltest_pred[j] > t_1:
                continue # e-value is zero

            if t_1 == t_0:
                evalues[j] = (Ncalib + 1) / (1 + np.sum(Lcalib * (Lcalib_pred <= t_1)))
                continue # same upper/lower bound case

            max_ell = np.zeros(Ntest + Ncalib) # max_ell[rank(t)]: max of l(t') with t' > t, t' in M, and FR(t', 0) <= gamma. 
                                               # max_ell[0] correspond to the smallest t in M, max_ell[-1] correspond to the largest t in M.
            last_max = -np.inf
            for i, t in zip(range(Ntest + Ncalib - 1, -1, -1), reversed(M)): # n+m iterations
                max_ell[i] = last_max

                if FR(j, t, 0) <= gamma:
                    last_max = max(last_max, ell(j, t)) # both O(n+m)

            M_star = []
            for i, t in enumerate(M):
                if t < max(Ltest_pred[j], t_1):
                    continue # this is to keep the index i
                if t > t_0:
                    break

                if FR(j, t, 0) <= gamma and ell(j, t) > max_ell[i]:
                    M_star.append(t)

            evalue = np.inf
            for t in M_star:
                cur_val = (Ncalib + 1) / (ell(j, t) + np.sum(Lcalib * (Lcalib_pred <= t)))
                evalue = min(evalue, cur_val)
            
            evalues[j] = evalue
        else: # oracle e-value
            t_j = t_gamma(j, Ltest[j])
            evalue = (Ncalib + 1) * (Ltest_pred[j] <= t_j) / (Ltest[j] * (Ltest_pred[j] <= t_j) + np.sum(Lcalib * (Lcalib_pred <= t_j)))
            evalues[j] = evalue
    
    if prune == 'hete':
        evalues /= np.random.uniform(0, 1, len(evalues))
    if prune == 'homo':
        evalues /= np.random.uniform(0, 1)
    sel = eBH(evalues, alpha)
    
    if not return_evals:
        return sel
    return sel, evalues

def SCoRE_SDR_fast(Dcalib, Dtest, alpha, gamma, prune=None, oracle=False, return_evals=False):
    Lcalib, Lcalib_pred = Dcalib
    Ltest, Ltest_pred = Dtest # True Ltest should not be used
    Ncalib, Ntest = len(Lcalib_pred), len(Ltest_pred)

    Lcalib_pred_tagged = [(lp, l, 'calib') for lp, l in zip(Lcalib_pred, Lcalib)]
    Ltest_pred_tagged = [(lp, 0, 'test') for lp in Ltest_pred] # 0 is dummy value

    M_tagged = Lcalib_pred_tagged + Ltest_pred_tagged
    M_tagged.sort()

    M = np.array([a[0] for a in M_tagged])

    evalues = np.zeros(Ntest)

    # some intermediate prefix sums
    NUMER = np.zeros(Ncalib + Ntest) # for np.sum(Lcalib * (Lcalib_pred <= t)) with t being the i-th ranked value in M
    DENOM = np.zeros(Ncalib + Ntest) # for 1 + np.sum(Ltest_pred <= t).
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
        FR_j = np.zeros(Ncalib + Ntest) # for oracle only - FR(j, t, Ltest[j])

        ELL = np.zeros(Ncalib + Ntest)

        # pairs of (i, t)
        t_0, t_1, t_j = (-1, -np.inf), (-1, -np.inf), (-1, -np.inf)

        # compute FR and ell
        for i, (t, _, _) in enumerate(M_tagged):
            FR_0[i] = NUMER[i] / (DENOM[i] - (Ltest_pred[j] <= t)) / (Ncalib + 1) * Ntest
            FR_1[i] = (NUMER[i] + (Ltest_pred[j] <= t)) / (DENOM[i] - (Ltest_pred[j] <= t)) / (Ncalib + 1) * Ntest
            
            if oracle:
                FR_j[i] = (NUMER[i] + Ltest[j] * (Ltest_pred[j] <= t)) / (DENOM[i] - (Ltest_pred[j] <= t)) / (Ncalib + 1) * Ntest

            ELL[i] = (Ncalib + 1) * gamma / Ntest * (DENOM[i] - (Ltest_pred[j] <= t)) - NUMER[i]

        # compute t_gamma. Also store the original ranking i
        for i, t in enumerate(M):
            if FR_0[i] <= gamma:
                t_0 = (i, t)
            if FR_1[i] <= gamma:
                t_1 = (i, t)
            if oracle and FR_j[i] <= gamma:
                t_j = (i, t)

        if not oracle:
            if Ltest_pred[j] > t_1[1]:
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
                if t < max(Ltest_pred[j], t_1[1]):
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
        else: # oracle e-value
            evalue = (Ncalib + 1) * (Ltest_pred[j] <= t_j) / (Ltest[j] * (Ltest_pred[j] <= t_j) + NUMER[t_j[0]])
            evalues[j] = evalue
    
    if prune == 'hete':
        evalues /= np.random.uniform(0, 1, len(evalues))
    if prune == 'homo':
        evalues /= np.random.uniform(0, 1)
    sel = eBH(evalues, alpha)
    
    if not return_evals:
        return sel
    return sel, evalues

def SCoRE_SDR_w_fast(Dcalib, Dtest, wcalib, wtest, alpha, gamma, prune=None, oracle=False, return_evals=False):
    Lcalib, Lcalib_pred = Dcalib
    Ltest, Ltest_pred = Dtest # True Ltest should not be used
    Ncalib, Ntest = len(Lcalib_pred), len(Ltest_pred)

    Lcalib_pred_tagged = [(lp, l, w, 'calib') for lp, l, w in zip(Lcalib_pred, Lcalib, wcalib)]
    Ltest_pred_tagged = [(lp, 0, w, 'test') for lp, w in zip(Ltest_pred, wtest)] # 0 is dummy value

    M_tagged = Lcalib_pred_tagged + Ltest_pred_tagged
    M_tagged.sort()

    M = np.array([a[0] for a in M_tagged])

    evalues = np.zeros(Ntest)

    calib_w_sum = np.sum(wcalib)

    # some intermediate prefix sums
    NUMER = np.zeros(Ncalib + Ntest) # for np.sum(wcalib * Lcalib * (Lcalib_pred <= t)) with t being the i-th ranked value in M
    DENOM = np.zeros(Ncalib + Ntest) # for 1 + np.sum(Ltest_pred <= t).
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
        FR_j = np.zeros(Ncalib + Ntest) # for oracle only - FR(j, t, Ltest[j])

        ELL = np.zeros(Ncalib + Ntest)

        # pairs of (i, t)
        t_0, t_1, t_j = (-1, -np.inf), (-1, -np.inf), (-1, -np.inf)

        # compute FR and ell
        for i, (t, _, _, _) in enumerate(M_tagged):
            FR_0[i] = NUMER[i] / (DENOM[i] - (Ltest_pred[j] <= t)) / (calib_w_sum + wtest[j]) * Ntest
            FR_1[i] = (NUMER[i] + wtest[j] * (Ltest_pred[j] <= t)) / (DENOM[i] - (Ltest_pred[j] <= t)) / (calib_w_sum + wtest[j]) * Ntest
            
            if oracle:
                FR_j[i] = (NUMER[i] + wtest[j] * Ltest[j] * (Ltest_pred[j] <= t)) / (DENOM[i] - (Ltest_pred[j] <= t)) / (calib_w_sum + wtest[j]) * Ntest

            ELL[i] = (calib_w_sum + wtest[j]) / wtest[j] * gamma / Ntest * (DENOM[i] - (Ltest_pred[j] <= t)) - NUMER[i] / wtest[j]

        # compute t_gamma. Also store the original ranking i
        for i, t in enumerate(M):
            if FR_0[i] <= gamma:
                t_0 = (i, t)
            if FR_1[i] <= gamma:
                t_1 = (i, t)
            if oracle and FR_j[i] <= gamma:
                t_j = (i, t)

        if not oracle:
            if Ltest_pred[j] > t_1[1]:
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
                if t < max(Ltest_pred[j], t_1[1]):
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
        else: # oracle e-value
            evalue = (calib_w_sum + wtest[j]) * (Ltest_pred[j] <= t_j) / (wtest[j] * Ltest[j] * (Ltest_pred[j] <= t_j) + NUMER[t_j[0]])
            evalues[j] = evalue
    
    if prune == 'hete':
        evalues /= np.random.uniform(0, 1, len(evalues))
    if prune == 'homo':
        evalues /= np.random.uniform(0, 1)
    sel = eBH(evalues, alpha)
    
    if not return_evals:
        return sel
    return sel, evalues