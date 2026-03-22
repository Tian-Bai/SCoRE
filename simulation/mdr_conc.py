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

def emp_rademacher_complexity(L, S, k=100):
    n = len(L)
    sort_idx = np.argsort(S)
    L_sorted = L[sort_idx]
    S_sorted = S[sort_idx]
    
    # handles ties in S. If is_group_end is false, using that value is not allowed in finding the supremum
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

    # get the empirical Rademacher complexity of the function class L 1{s(X) <= t}
    rad_comp = emp_rademacher_complexity(Lcalib, Lcalib_pred, k=100)
    rad_comp_r = emp_rademacher_complexity(Lcalib, Scalib_pred, k=100)

    eps = np.sqrt(np.log(2 / args.delta) / (2 * ncalib))

    for q in q_list:
        # search for threshold based on rademacher complexity
        penalty = 2 * rad_comp + 3 * eps # 1 * eps due to empirical mean, 2 * eps due to empirical rademacher
        emp_means = np.array([np.sum(Lcalib * (Lcalib_pred <= t)) for t in Lcalib_pred]) / ncalib
        valid_t = Lcalib_pred[emp_means + penalty <= q]
        t_rad = np.max(valid_t) if len(valid_t) > 0 else -np.inf
        sel_rad = np.where(Ltest_pred <= t_rad)[0]

        mdr_rad, nsel_rad = eval_MDR(Ltest, np.ones_like(Ltest), sel_rad)
        _, reward_rad = eval_MDR(Ltest, Rtest, sel_rad)

        # with reward
        penalty_r = 2 * rad_comp_r + 3 * eps
        emp_means_r = np.array([np.sum(Lcalib * (Scalib_pred <= t)) for t in Scalib_pred]) / ncalib
        valid_t_r = Scalib_pred[emp_means_r + penalty_r <= q]
        t_rad_r = np.max(valid_t_r) if len(valid_t_r) > 0 else -np.inf
        sel_r = np.where(Stest_pred <= t_rad_r)[0]

        mdr_rad_r, nsel_rad_r = eval_MDR(Ltest, np.ones_like(Ltest), sel_r)
        _, reward_rad_r = eval_MDR(Ltest, Rtest, sel_r)

        # Hoeffding approach
        K = 101
        search_grid = np.linspace(0, 1, K)
        penalty_hoef = np.sqrt(np.log(2 * K / args.delta) / (2 * ncalib))
        
        emp_means = np.array([np.sum(Lcalib * (Lcalib_pred <= t)) for t in search_grid]) / ncalib
        valid_t_hoef = search_grid[emp_means + penalty_hoef <= q]
        t_hoef = np.max(valid_t_hoef) if len(valid_t_hoef) > 0 else -np.inf
        sel_hoef = np.where(Ltest_pred <= t_hoef)[0]

        mdr_hoef, nsel_hoef = eval_MDR(Ltest, np.ones_like(Ltest), sel_hoef)
        _, reward_hoef = eval_MDR(Ltest, Rtest, sel_hoef)

        # with reward
        search_grid_r = np.linspace(0, np.max(Scalib_pred), K) # while strictly speaking the grid shouldn't be data-dependent..
        emp_means_r = np.array([np.sum(Lcalib * (Scalib_pred <= t)) for t in search_grid_r]) / ncalib
        valid_t_hoef_r = search_grid_r[emp_means_r + penalty_hoef <= q]
        t_hoef_r = np.max(valid_t_hoef_r) if len(valid_t_hoef_r) > 0 else -np.inf
        sel_hoef_r = np.where(Stest_pred <= t_hoef_r)[0]

        mdr_hoef_r, nsel_hoef_r = eval_MDR(Ltest, np.ones_like(Ltest), sel_hoef_r)
        _, reward_hoef_r = eval_MDR(Ltest, Rtest, sel_hoef_r)

        # Naive approach: no correction at all
        valid_t_naive = search_grid[emp_means <= q]
        t_naive = np.max(valid_t_naive) if len(valid_t_naive) > 0 else -np.inf
        sel_naive = np.where(Ltest_pred <= t_naive)[0]

        mdr_naive, nsel_naive = eval_MDR(Ltest, np.ones_like(Ltest), sel_naive)
        _, reward_naive = eval_MDR(Ltest, Rtest, sel_naive)

        # Naive with reward
        valid_t_naive_r = search_grid_r[emp_means_r <= q]
        t_naive_r = np.max(valid_t_naive_r) if len(valid_t_naive_r) > 0 else -np.inf
        sel_naive_r = np.where(Stest_pred <= t_naive_r)[0]

        mdr_naive_r, nsel_naive_r = eval_MDR(Ltest, np.ones_like(Ltest), sel_naive_r)
        _, reward_naive_r = eval_MDR(Ltest, Rtest, sel_naive_r)

        SCoRE_df = pd.DataFrame({
            'mdr_rad': [mdr_rad],
            'nsel_rad': [nsel_rad],
            'reward_rad': [reward_rad],
            'mdr_rad_r': [mdr_rad_r],
            'nsel_rad_r': [nsel_rad_r],
            'reward_rad_r': [reward_rad_r],
            'mdr_hoef': [mdr_hoef],
            'nsel_hoef': [nsel_hoef],
            'reward_hoef': [reward_hoef],
            'mdr_hoef_r': [mdr_hoef_r],
            'nsel_hoef_r': [nsel_hoef_r],
            'reward_hoef_r': [reward_hoef_r],
            'mdr_naive': [mdr_naive],
            'nsel_naive': [nsel_naive],
            'reward_naive': [reward_naive],
            'mdr_naive_r': [mdr_naive_r],
            'nsel_naive_r': [nsel_naive_r],
            'reward_naive_r': [reward_naive_r],
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

    # alternatively, use concentration to get an upper bound U(t) on E[L 1{s(X) < t}] and select t that U(t) <= alpha

if not os.path.exists('results_conc'):
    os.makedirs('results_conc')

all_res.to_csv(os.path.join('results_conc', f"Conc_MDR, case={case_idx}, setting={setting}, sigma={sig}, tau={tau}, ncalib={ncalib}, ntest={ntest}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}, fit_Y={fit_Y}, oracle_mdl={oracle_mdl}.csv"))
        

