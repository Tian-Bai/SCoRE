import numpy as np
import pandas as pd
import random
import sys
import os
from sklearn.ensemble import RandomForestRegressor

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
root_dir = os.path.normpath(root_dir)
sys.path.append(root_dir)
from SCoRE import eval_MDR, gen_data_1, gen_data_2, gen_data_Jin2023, loss_1, loss_2, loss_Jin2023
from scipy.stats import binom
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='.')
parser.add_argument('case', type=int)
parser.add_argument('setting', type=int)
parser.add_argument('sigma', type=float, default=0.1)
parser.add_argument('dim', type=int, default=20)
parser.add_argument('delta', type=float, default=0.95)
parser.add_argument('tau', type=float, default=10)

parser.add_argument('ncalib', type=int)
parser.add_argument('ntest', type=int)
parser.add_argument('Nrep', type=int)
parser.add_argument('seedgroup', type=int)

parser.add_argument('--reward', dest='reward', type=int, default=0)
parser.add_argument('--fit_Y', dest='fit_Y', type=int, default=0)
parser.add_argument('--oracle', dest='oracle', type=int, default=0)
parser.add_argument('--oracle_mdl', dest='oracle_mdl', type=int, default=0)

args = parser.parse_args()

case_idx = args.case
setting = args.setting
sig = args.sigma
dim = args.dim
tau = args.tau # only relevant for the Jin & Candes case (case 3)

ncalib = args.ncalib
ntest = args.ntest
Nrep = args.Nrep
seedgroup = args.seedgroup

reward_type = args.reward # reward type
fit_Y = bool(args.fit_Y)
oracle = bool(args.oracle)
oracle_mdl = bool(args.oracle_mdl)

q_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

CLIP_LOOKUP = { # (setting, sigma). The returned clip values are approx. larger than the 95-th quantile of (Y - f(X))^2
    (1, 0.05): 0.4,
    (1, 0.1):  0.6,
    (1, 0.2):  1.4,
    (1, 0.3):  2.6,
    (2, 0.05): 0.3,
    (2, 0.1):  0.4,
    (2, 0.2):  0.7,
    (2, 0.3):  1.2,
}

# define a custom reward function here
if reward_type == 0:
    r = lambda X, Y: Y ** 2

all_res = pd.DataFrame()

def avg_loss_at_t(L, S, t):
    return np.mean(L * (S <= t))

def hb_p_value(avg_loss, n, q):
    if avg_loss >= q:
        return 1.0

    def h_1(a, b):
        if a <= 0:
            return -np.log(1 - b)
        if a >= 1:
            return -np.log(b)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    p_hoeffding = np.exp(-n * h_1(avg_loss, q))
    p_bentkus = np.e * binom.cdf(np.floor(n * avg_loss), n, q)
    return min(p_hoeffding, p_bentkus)

for i_itr in tqdm(range(Nrep * seedgroup, Nrep * (seedgroup + 1))):
    random.seed(i_itr)
    np.random.seed(i_itr)
    if case_idx == 1:
        Xtrain, _, _, Ytrain = gen_data_1(setting, 1000, sig, dim)
        Xcalib, _, _, Ycalib = gen_data_1(setting, ncalib, sig, dim)
        Xtest, _, _, Ytest = gen_data_1(setting, ntest, sig, dim)

        Ltrain, Lcalib, Ltest = loss_1(Ytrain), loss_1(Ycalib), loss_1(Ytest)
        Rtrain, Rcalib, Rtest = r(Xtrain, Ytrain), r(Xcalib, Ycalib), r(Xtest, Ytest)

    if case_idx == 2:
        # first, prepare a model f to predict Y
        Xprep, _, _, Yprep = gen_data_2(setting, 1000, sig, dim)
        f = RandomForestRegressor()
        f.fit(Xprep, Yprep)

        clip_const = CLIP_LOOKUP[(setting, sig)]

        Xtrain, _, _, Ytrain = gen_data_2(setting, 1000, sig, dim)
        Xcalib, _, _, Ycalib = gen_data_2(setting, ncalib, sig, dim)
        Xtest, _, _, Ytest = gen_data_2(setting, ntest, sig, dim)

        Ltrain, Lcalib, Ltest = loss_2(Ytrain, f, Xtrain, clip_const), loss_2(Ycalib, f, Xcalib, clip_const), loss_2(Ytest, f, Xtest, clip_const)
        Rtrain, Rcalib, Rtest = r(Xtrain, Ytrain), r(Xcalib, Ycalib), r(Xtest, Ytest)

    if case_idx == 3:
        Xtrain, _, _, Ytrain = gen_data_Jin2023(setting, 1000, sig, dim)
        Xcalib, _, _, Ycalib = gen_data_Jin2023(setting, ncalib, sig, dim)
        Xtest, _, _, Ytest = gen_data_Jin2023(setting, ntest, sig, dim)

        Ltrain, Lcalib, Ltest = loss_Jin2023(Ytrain, tau), loss_Jin2023(Ycalib, tau), loss_Jin2023(Ytest, tau)
        Rtrain, Rcalib, Rtest = r(Xtrain, Ytrain), r(Xcalib, Ycalib), r(Xtest, Ytest)

    # fit models
    if oracle_mdl:
        Lcalib_pred, Ltest_pred = Lcalib, Ltest
        Scalib_pred, Stest_pred = Lcalib_pred / Rcalib, Ltest_pred / Rtest
    else:
        mu_reward = RandomForestRegressor()
        mu_reward.fit(Xtrain, Rtrain)

        mu = RandomForestRegressor()
        if fit_Y:
            mu.fit(Xtrain, Ytrain)
            
            if case_idx == 1:
                Lcalib_pred = loss_1(mu.predict(Xcalib))
                Ltest_pred = loss_1(mu.predict(Xtest))
            if case_idx == 2:
                Lcalib_pred = loss_2(mu.predict(Xcalib), f, Xcalib, clip_const) 
                Ltest_pred = loss_2(mu.predict(Xtest), f, Xtest, clip_const) 
            if case_idx == 3:
                Lcalib_pred = loss_Jin2023(mu.predict(Xcalib), tau)
                Ltest_pred = loss_Jin2023(mu.predict(Xtest), tau)
        else:
            mu.fit(Xtrain, Ltrain)

            Lcalib_pred = mu.predict(Xcalib)
            Ltest_pred = mu.predict(Xtest)
        
        Scalib_pred = Lcalib_pred / mu_reward.predict(Xcalib)
        Stest_pred = Ltest_pred / mu_reward.predict(Xtest)

    K = 101 # LTT can handle an arbitrarily dense grid because there is no penalty via K!
    search_grid = np.linspace(0, 1, K)
    search_grid_r = np.linspace(0, np.max(Scalib_pred) * 1.1, K)
    
    for q in q_list:
        # LTT approach
        t_ltt = -np.inf
        for t in search_grid:
            emp_mean = np.sum(Lcalib * (Lcalib_pred <= t)) / ncalib
            
            # The test: is empirical risk small enough to reject H_t at level q?
            if hb_p_value(avg_loss_at_t(Lcalib, Lcalib_pred, t), ncalib, q) <= args.delta:
                t_ltt = t
            else:
                break

        sel_ltt = np.where(Ltest_pred <= t_ltt)[0]
        mdr_ltt, nsel_ltt = eval_MDR(Ltest, np.ones_like(Ltest), sel_ltt)
        _, reward_ltt = eval_MDR(Ltest, Rtest, sel_ltt)

        # LTT approach for MDR with reward
        t_ltt_r = -np.inf
        for t in search_grid_r:
            if hb_p_value(avg_loss_at_t(Lcalib, Scalib_pred, t), ncalib, q) <= args.delta:
                t_ltt_r = t
            else:
                break
                
        sel_ltt_r = np.where(Stest_pred <= t_ltt_r)[0]
        mdr_ltt_r, nsel_ltt_r = eval_MDR(Ltest, np.ones_like(Ltest), sel_ltt_r)
        _, reward_ltt_r = eval_MDR(Ltest, Rtest, sel_ltt_r)

        SCoRE_df = pd.DataFrame({
            'mdr_ltt': [mdr_ltt],
            'nsel_ltt': [nsel_ltt],
            'reward_ltt': [reward_ltt],
            'mdr_ltt_r': [mdr_ltt_r],
            'nsel_ltt_r': [nsel_ltt_r],
            'reward_ltt_r': [reward_ltt_r],
            'q': [q],
            'setting': [setting],
            'dim': [dim],
            'seed': [i_itr],
            'sigma': [sig],
            'tau': [tau],
            'ncalib': [ncalib],
            'ntest': [ntest],
            'reward_type': [reward_type],
            'fit_Y': [fit_Y],
            'oracle': [oracle]
        })

        all_res = pd.concat((all_res, SCoRE_df))

if not os.path.exists('results_ltt'):
    os.makedirs('results_ltt')

all_res.to_csv(os.path.join('results_ltt', f"LTT_MDR, case={case_idx}, setting={setting}, sigma={sig}, tau={tau}, ncalib={ncalib}, ntest={ntest}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}, fit_Y={fit_Y}, oracle_mdl={oracle_mdl}.csv"))

