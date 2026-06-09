import numpy as np


def _expit(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _get_rng(random_state):
    if random_state is None:
        return np.random
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)

def loss_Jin2023(Y, tau):
    """Calculates the smoothened indicator loss function, similar to the data generation process in Jin and Candes (2023).
    
    The loss function is of the form sigmoid(-tau * Y).
    
    Args:
        Y (np.ndarray): The target values.
        tau (float): Hyperparameter for smoothing. Larger tau means closer to 1{Y <= 0}. 
            If tau = np.inf, it returns strictly 1{Y <= 0}.
            
    Returns:
        np.ndarray: The computed loss values.
    """
    if tau != np.inf:
        return _expit(-Y * tau) # L = smoothened indicator of <= 0
    else:
        return (Y <= 0)

def gen_data_Jin2023(setting, n, sig, dim=20, random_state=None):
    """Generates artificial data using the data generation process in Jin and Candes (2023).
    
    Args:
        setting (int): The data generation setting (1 or 2).
        n (int): Number of samples to generate.
        sig (float): Noise scaling factor.
        dim (int, optional): Dimensionality of the feature space. Defaults to 20.
        random_state (int or np.random.Generator, optional): Random seed or generator for reproducible samples.
        
    Returns:
        tuple: A tuple (X, mu_x, eps, Y) representing the generated data and components.
    """
    rng = _get_rng(random_state)

    if setting == 1:
        X = rng.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
        eps = rng.normal(size=n) * (5.5 - abs(mu_x)) / 2 * sig
        Y = mu_x + eps
        return X, mu_x, eps, Y

    if setting == 2:
        X = rng.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        eps = rng.normal(size=n) * (5.5 - abs(mu_x)) / 2 * sig
        Y = mu_x + eps
        return X, mu_x, eps, Y

    raise ValueError("setting must be 1 or 2")

def loss_1(Y):
    """Calculates the expected shortfall-like loss function.
    
    The loss takes the form L(f, x, y) = y * 1{y > c}, evaluated against Y.
    
    Args:
        Y (np.ndarray): The target values.
        
    Returns:
        np.ndarray: The computed expected shortfall loss.
    """
    return 1/6 * Y * (Y > 2)

def gen_data_1(setting, n, sig, dim=20, random_state=None):
    """Generates artificial data for the first case.
    
    Args:
        setting (int): The data generation setting (1 or 2).
        n (int): Number of samples to generate.
        sig (float): Noise scaling factor.
        dim (int, optional): Dimensionality of the features. Defaults to 20.
        random_state (int or np.random.Generator, optional): Random seed or generator for reproducible samples.
        
    Returns:
        tuple: A tuple (X, mu_x, eps, Y) containing the covariates and responses.
    """
    rng = _get_rng(random_state)

    if setting == 1:
        X = rng.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.5 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.5) + 3 # now in (1.5, 4.5)
        eps = np.clip(rng.normal(size=n) * sig * (5.5 - mu_x), -1.5, 1.5) # clip the noise to be in (-1.5, 1.5)
        Y = mu_x + eps # (0, 6)
        return X, mu_x, eps, Y
    
    if setting == 2:
        X = rng.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) + 2 # in (1, 5)
        eps = np.clip(rng.normal(size=n) * sig * (6 - mu_x) * 0.5, -1, 1) # clip the noise to be in (-1, 1)
        Y = mu_x + eps # (0, 6)
        return X, mu_x, eps, Y

    raise ValueError("setting must be 1 or 2")
    
def loss_2(Y, f, X, clip_const):
    """Calculates clipped prediction error loss.
    
    Evaluates loss of the form L(f, x, y) = (y - f(x))^2, clipped at clip_const.
    
    Args:
        Y (np.ndarray): The true target values.
        f (object): The regression model to use for prediction (must have `.predict()`).
        X (np.ndarray): The feature matrix to run predictions against.
        clip_const (float): The clipping boundary, i.e., loss is in [0, clip_const].
        
    Returns:
        np.ndarray: The computed normalized prediction error loss over X.
    """
    return np.clip((Y - f.predict(X)) ** 2, 0, clip_const) / clip_const

def gen_data_2(setting, n, sig, dim=20, random_state=None):
    """Generates artificial data for the second case.
    
    Args:
        setting (int): The data generation setting (1 or 2).
        n (int): Number of samples to generate.
        sig (float): Noise scaling factor.
        dim (int, optional): Dimensionality. Defaults to 20.
        random_state (int or np.random.Generator, optional): Random seed or generator for reproducible samples.
        
    Returns:
        tuple: A tuple (X, mu_x, eps, Y) with covariates and label values.
    """
    rng = _get_rng(random_state)

    if setting == 1:
        X = rng.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.5 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.5) + 3 # now in (1.5, 4.5)
        eps = np.clip(rng.normal(size=n) * sig * (5.5 - mu_x), -1.5, 1.5) # clip the noise to be in (-1.5, 1.5)
        Y = mu_x + eps # (0, 6)
        return X, mu_x, eps, Y
    
    if setting == 2:
        X = rng.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) + 2 # in (1, 5)
        eps = np.clip(rng.normal(size=n) * sig * (6 - mu_x) * 0.5, -1, 1) # clip the noise to be in (-1, 1)
        Y = mu_x + eps # (0, 6)
        return X, mu_x, eps, Y

    raise ValueError("setting must be 1 or 2")

def BH(pvals, q):
    """Applies the Benjamini-Hochberg (BH) procedure to a list of p-values.
    
    Args:
        pvals (array-like): List or array of p-values.
        q (float): The nominal False Discovery Rate (FDR) level.
        
    Returns:
        np.ndarray: The indices forming the rejection set.
    """
    pvals = np.asarray(pvals, dtype=float)
    ntest = pvals.size

    if ntest == 0:
        return np.array([], dtype=int)

    order = np.argsort(pvals, kind="mergesort")
    sorted_pvals = pvals[order]
    thresholds = q * np.arange(1, ntest + 1) / ntest
    selected = np.flatnonzero(sorted_pvals <= thresholds)

    if selected.size == 0:
        return np.array([], dtype=int)

    return order[: selected[-1] + 1]

def eBH(evals, q):
    """Applies the base e-BH procedure to a list of e-values.
    
    Args:
        evals (array-like): List or array of e-values.
        q (float): The nominal False Discovery Rate (FDR) level.
        
    Returns:
        np.ndarray: The indices forming the rejection set.
    """
    return BH(np.divide(1.0, evals, np.full_like(evals, np.inf), where=(evals != 0)), q)

def eval_MDR(L, R, sel):
    """Evaluates selection performance for risk and power in the MDR sense.
    
    Args:
        L (np.ndarray): The true loss corresponding to every instance.
        R (np.ndarray): The true rewards for each instance.
        sel (array-like): The selection set generated by the test procedure.
        
    Returns:
        tuple: (risk_acc, reward_acc) indicating the MDR risk and cumulative reward.
    """
    if len(sel) == 0:
        return 0, 0
    risk_acc = np.sum(L[sel]) / len(L)
    reward_acc = np.sum(R[sel])
    return risk_acc, reward_acc

def eval_SDR(L, R, sel):
    """Evaluates selection performance for risk and power in the SDR sense.
    
    Args:
        L (np.ndarray): The true loss corresponding to every instance.
        R (np.ndarray): The true rewards for each instance.
        sel (array-like): The selection set generated by the test procedure.
        
    Returns:
        tuple: (sdr, bin_power, reward) corresponding to SDR, binary power (equivalent to power in the binary loss case), and reward metrics.
    """
    if len(sel) == 0:
        return 0, 0, 0
    true_rej = len(L) - np.sum(L) # number of zeros in L
    sdr = np.sum(L[sel]) / len(sel)
    bin_power = (len(sel) - np.sum(L[sel])) / true_rej if true_rej != 0 else 0 # defined only for 0-1 loss
    reward = np.sum(R[sel])
    return sdr, bin_power, reward

class Lpredictor:
    """Encapsulates a target predictor and a loss mapping into an expected loss predictor.
    
    Acts as a wrapper returning the loss via its `.predict()` method directly.
    """
    def __init__(self, Ypred, loss_fn):
        self.Ypred = Ypred
        self.loss_fn = loss_fn

    def predict(self, X):
        Y_hat = self.Ypred.predict(X)
        return self.loss_fn(Y_hat, X)
