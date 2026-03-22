import numpy as np
import pandas as pd
import random
import sys
import os
from sklearn.ensemble import RandomForestRegressor

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
root_dir = os.path.normpath(root_dir)
sys.path.append(root_dir)
from SCoRE import eval_SDR, gen_data_1, gen_data_2, gen_data_Jin2023, loss_1, loss_2, loss_Jin2023, SCoRE_SDR_fast
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='.')
parser.add_argument('case', type=int)
parser.add_argument('setting', type=int)
parser.add_argument('sigma', type=float)
parser.add_argument('delta', type=float, default=0.95)
parser.add_argument('dim', type=int)
parser.add_argument('tau', type=float, default=10)

parser.add_argument('ncalib', type=int)
parser.add_argument('ntest', type=int)
parser.add_argument('Nrep', type=int)
parser.add_argument('seedgroup', type=int)

parser.add_argument('--reward', dest='reward', type=int, default=0)          # do we incorporate reward in SCoRE algorithm?
parser.add_argument('--fit_Y', dest='fit_Y', type=int, default=0)            # is the prediction model fit on Y or the loss L?
parser.add_argument('--oracle', dest='oracle', type=int, default=0)          # do we use the oracle e-values? 
parser.add_argument('--oracle_mdl', dest='oracle_mdl', type=int, default=0)  # do we use the oracle models?

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

def emp_rademacher_complexity(L, S, k=100):
    n = len(L)
    sort_idx = np.argsort(S)
    L_sorted = L[sort_idx]
    S_sorted = S[sort_idx]
    
    is_group_end = np.append(np.diff(S_sorted) != 0, True)
    
    max_sums = np.zeros(k)
    for i in range(k):
        sigma = np.random.choice([-1, 1], size=n)
        prefix_sums = np.cumsum(sigma * L_sorted)
        valid_sums = prefix_sums[is_group_end]
        max_sums[i] = max(0, np.max(valid_sums))
        
    return np.mean(max_sums) / n

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

    if not oracle_mdl:
        mu = RandomForestRegressor()
        mu_reward = RandomForestRegressor()
        mu_reward.fit(Xtrain, Rtrain)
        if fit_Y:
            mu.fit(Xtrain, Ytrain)
        else:
            mu.fit(Xtrain, Ltrain)

    eps = np.sqrt(np.log(4 / args.delta) / (2 * ncalib))
    
    rad_comp_num = emp_rademacher_complexity(Lcalib, Lcalib_pred, k=100)
    rad_comp_den = emp_rademacher_complexity(np.ones_like(Lcalib), Lcalib_pred, k=100)
    rad_comp_num_r = emp_rademacher_complexity(Lcalib, Scalib_pred, k=100)
    rad_comp_den_r = emp_rademacher_complexity(np.ones_like(Lcalib), Scalib_pred, k=100)

    for q in q_list:
        # depending on q
        if oracle_mdl:
            Lcalib_pred, Ltest_pred = Lcalib - q, Ltest - q
            Scalib_pred, Stest_pred = Lcalib_pred / Rcalib, Ltest_pred / Rtest
        else:
            if fit_Y:
                if case_idx == 1:
                    Lcalib_pred = loss_1(mu.predict(Xcalib)) - q
                    Ltest_pred = loss_1(mu.predict(Xtest)) - q
                if case_idx == 2:
                    Lcalib_pred = loss_2(mu.predict(Xcalib), f, Xcalib, clip_const) - q
                    Ltest_pred = loss_2(mu.predict(Xtest), f, Xtest, clip_const) - q
                if case_idx == 3:
                    Lcalib_pred = loss_Jin2023(mu.predict(Xcalib), tau) - q
                    Ltest_pred = loss_Jin2023(mu.predict(Xtest), tau) - q
            else:
                Lcalib_pred = mu.predict(Xcalib) - q
                Ltest_pred = mu.predict(Xtest) - q
            
            Scalib_pred = Lcalib_pred / mu_reward.predict(Xcalib)
            Stest_pred = Ltest_pred / mu_reward.predict(Xtest)

        # Setup search grids depending on scaled prediction
        K = 101
        search_grid = np.linspace(0, np.max(Lcalib_pred), K) if len(Lcalib_pred) > 0 else np.linspace(0, 1, K)
        search_grid_r = np.linspace(0, np.max(Scalib_pred), K) if len(Scalib_pred) > 0 else np.linspace(0, 1, K)
        
        # Precompute empirical means
        emp_num = np.array([np.sum(Lcalib * (Lcalib_pred <= t)) for t in search_grid]) / ncalib
        emp_den = np.array([np.sum(Lcalib_pred <= t) for t in search_grid]) / ncalib
        
        emp_num_r = np.array([np.sum(Lcalib * (Scalib_pred <= t)) for t in search_grid_r]) / ncalib
        emp_den_r = np.array([np.sum(Scalib_pred <= t) for t in search_grid_r]) / ncalib

        # naive
        valid_naive = (emp_den > 0) & ((emp_num / np.maximum(emp_den, 1e-12)) <= q)
        t_naive = np.max(search_grid[valid_naive]) if np.any(valid_naive) else -np.inf
        sel_naive = np.where(Ltest_pred <= t_naive)[0]
        sdr_naive = eval_SDR(Ltest, np.ones_like(Ltest), sel_naive)[0]
        nsel_naive = len(sel_naive)
        reward_naive = eval_SDR(Ltest, Rtest, sel_naive)[2]
        
        valid_naive_r = (emp_den_r > 0) & ((emp_num_r / np.maximum(emp_den_r, 1e-12)) <= q)
        t_naive_r = np.max(search_grid_r[valid_naive_r]) if np.any(valid_naive_r) else -np.inf
        sel_naive_r = np.where(Stest_pred <= t_naive_r)[0]
        sdr_naive_r = eval_SDR(Ltest, np.ones_like(Ltest), sel_naive_r)[0]
        nsel_naive_r = len(sel_naive_r)
        reward_naive_r = eval_SDR(Ltest, Rtest, sel_naive_r)[2]

        # bound using rademacher complexity
        penalty_num_rad = 2 * rad_comp_num + 3 * eps
        penalty_den_rad = 2 * rad_comp_den + 3 * eps
        
        valid_rad = (emp_den - penalty_den_rad > 0) & (((emp_num + penalty_num_rad) / np.maximum(emp_den - penalty_den_rad, 1e-12)) <= q)
        t_rad = np.max(search_grid[valid_rad]) if np.any(valid_rad) else -np.inf
        sel_rad = np.where(Ltest_pred <= t_rad)[0]
        sdr_rad, _, nsel_rad = eval_SDR(Ltest, np.ones_like(Ltest), sel_rad)
        _, _, reward_rad = eval_SDR(Ltest, Rtest, sel_rad)

        penalty_num_rad_r = 2 * rad_comp_num_r + 3 * eps
        penalty_den_rad_r = 2 * rad_comp_den_r + 3 * eps
        
        valid_rad_r = (emp_den_r - penalty_den_rad_r > 0) & (((emp_num_r + penalty_num_rad_r) / np.maximum(emp_den_r - penalty_den_rad_r, 1e-12)) <= q)
        t_rad_r = np.max(search_grid_r[valid_rad_r]) if np.any(valid_rad_r) else -np.inf
        sel_rad_r = np.where(Stest_pred <= t_rad_r)[0]
        sdr_rad_r, _, nsel_rad_r = eval_SDR(Ltest, np.ones_like(Ltest), sel_rad_r)
        _, _, reward_rad_r = eval_SDR(Ltest, Rtest, sel_rad_r)

        # bound using hoeffding on fixed grid
        penalty_num_hoef = np.sqrt(np.log(4 * K / args.delta) / (2 * ncalib))
        penalty_den_hoef = np.sqrt(np.log(4 * K / args.delta) / (2 * ncalib))
        
        valid_hoef = (emp_den - penalty_den_hoef > 0) & (((emp_num + penalty_num_hoef) / np.maximum(emp_den - penalty_den_hoef, 1e-12)) <= q)
        t_hoef = np.max(search_grid[valid_hoef]) if np.any(valid_hoef) else -np.inf
        sel_hoef = np.where(Ltest_pred <= t_hoef)[0]
        sdr_hoef, _, nsel_hoef = eval_SDR(Ltest, np.ones_like(Ltest), sel_hoef)
        _, _, reward_hoef = eval_SDR(Ltest, Rtest, sel_hoef)

        valid_hoef_r = (emp_den_r - penalty_den_hoef > 0) & (((emp_num_r + penalty_num_hoef) / np.maximum(emp_den_r - penalty_den_hoef, 1e-12)) <= q)
        t_hoef_r = np.max(search_grid_r[valid_hoef_r]) if np.any(valid_hoef_r) else -np.inf
        sel_hoef_r = np.where(Stest_pred <= t_hoef_r)[0]
        sdr_hoef_r, _, nsel_hoef_r = eval_SDR(Ltest, np.ones_like(Ltest), sel_hoef_r)
        _, _, reward_hoef_r = eval_SDR(Ltest, Rtest, sel_hoef_r)

        SCoRE_df = pd.DataFrame({
            'sdr_naive': [sdr_naive],
            'nsel_naive': [nsel_naive],
            'reward_naive': [reward_naive],
            'sdr_naive_r': [sdr_naive_r],
            'nsel_naive_r': [nsel_naive_r],
            'reward_naive_r': [reward_naive_r],
            'sdr_rad': [sdr_rad],
            'nsel_rad': [nsel_rad],
            'reward_rad': [reward_rad],
            'sdr_rad_r': [sdr_rad_r],
            'nsel_rad_r': [nsel_rad_r],
            'reward_rad_r': [reward_rad_r],
            'sdr_hoef': [sdr_hoef],
            'nsel_hoef': [nsel_hoef],
            'reward_hoef': [reward_hoef],
            'sdr_hoef_r': [sdr_hoef_r],
            'nsel_hoef_r': [nsel_hoef_r],
            'reward_hoef_r': [reward_hoef_r],
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

if not os.path.exists('results_conc'):
    os.makedirs('results_conc')

all_res.to_csv(os.path.join('results_conc', f"Conc_SDR, case={case_idx}, setting={setting}, sigma={sig}, tau={tau}, ncalib={ncalib}, ntest={ntest}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}, fit_Y={fit_Y}, oracle_mdl={oracle_mdl}.csv"))
        

