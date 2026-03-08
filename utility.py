import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
from scipy.special import expit

"""
Generate artificial data, similar to Jin and Candes (2023). The loss function is of the form sigmoid(-tau * Y). -- smoothened indicator 
tau is hyperparameter for smoothing. Larger tau means closer to 1{Y <= 0}. tau = np.inf means 1{Y <= 0}.
"""
def loss_Jin2023(Y, tau):
    if tau != np.inf:
        return expit(-Y * tau) # L = smoothened indicator of <= 0
    else:
        return (Y <= 0)

def gen_data_Jin2023(setting, n, sig, dim=20):
    if setting == 1:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
        eps = np.random.normal(size=n) * (5.5 - abs(mu_x)) / 2 * sig
        Y = mu_x + eps
        return X, mu_x, eps, Y

    if setting == 2:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        eps = np.random.normal(size=n) * (5.5 - abs(mu_x)) / 2 * sig
        Y = mu_x + eps
        return X, mu_x, eps, Y

"""
Generate artificial data with loss function of the form L(f, x, y) = y * 1{y > c}. -- expected shortfall
"""
def loss_1(Y):
    return 1/6 * Y * (Y > 2)

def gen_data_1(setting, n, sig, dim=20):
    if setting == 1:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.5 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.5) + 3 # now in (1.5, 4.5)
        eps = np.clip(np.random.normal(size=n) * sig * (5.5 - mu_x), -1.5, 1.5) # clip the noise to be in (-1.5, 1.5)
        Y = mu_x + eps # (0, 6)
        return X, mu_x, eps, Y
    
    if setting == 2:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) + 2 # in (1, 5)
        eps = np.clip(np.random.normal(size=n) * sig * (6 - mu_x) * 0.5, -1, 1) # clip the noise to be in (-1, 1)
        Y = mu_x + eps # (0, 6)
        return X, mu_x, eps, Y
    
"""
Generate artificial data with loss function of the form L(f, x, y) = (y - f(x))^2. -- prediction error
If f is none, L is not included in the output. This could used to fit f.
"""
def loss_2(Y, f, X, clip_const):
    return np.clip((Y - f.predict(X)) ** 2, 0, clip_const) / clip_const

def gen_data_2(setting, n, sig, dim=20):
    if setting == 1:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.5 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.5) + 3 # now in (1.5, 4.5)
        eps = np.clip(np.random.normal(size=n) * sig * (5.5 - mu_x), -1.5, 1.5) # clip the noise to be in (-1.5, 1.5)
        Y = mu_x + eps # (0, 6)
        return X, mu_x, eps, Y
    
    if setting == 2:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) + 2 # in (1, 5)
        eps = np.clip(np.random.normal(size=n) * sig * (6 - mu_x) * 0.5, -1, 1) # clip the noise to be in (-1, 1)
        Y = mu_x + eps # (0, 6)
        return X, mu_x, eps, Y

"""
Given a list of p-values and nominal FDR level q, apply BH procedure to get a rejection set.
"""
def BH(pvals, q):
    ntest = len(pvals)
         
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return np.array([])
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller) + 1)])
        return idx_sel

"""
Given a list of e-values and nominal FDR level q, apply base eBH procedure (no pruning) to get a rejection set.
"""
def eBH(evals, q):
    return BH(np.divide(1.0, evals, np.full_like(evals, np.inf), where=(evals != 0)), q)

"""
Evaluate the selection performance in terms of risk and power (in the MDR sense), for any selection set.
"""
def eval_MDR(L, R, sel):
    if len(sel) == 0:
        return 0, 0
    risk_acc = np.sum(L[sel]) / len(L)
    reward_acc = np.sum(R[sel])
    return risk_acc, reward_acc

"""
Evaluate the selection performance in terms of FDP and power (in the SDR sense), for any selection set.
"""
def eval_SDR(L, R, sel):
    if len(sel) == 0:
        return 0, 0, 0
    true_rej = len(L) - np.sum(L) # number of zeros in L
    sdp = np.sum(L[sel]) / len(sel)
    bin_power = (len(sel) - np.sum(L[sel])) / true_rej if true_rej != 0 else 0 # defined only for 0-1 selection
    power = np.sum(R[sel])
    return sdp, bin_power, power

"""
Encapsulate a Y predictor and a loss function into a L predictor with .predict() method.
"""
class Lpredictor:
    def __init__(self, Ypred, loss_fn):
        self.Ypred = Ypred
        self.loss_fn = loss_fn

    def predict(self, X):
        Y_hat = self.Ypred.predict(X)
        return self.loss_fn(Y_hat, X)