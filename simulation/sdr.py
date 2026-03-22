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

        homo_sel = SCoRE_SDR_fast([Lcalib, Lcalib_pred], [None, Ltest_pred], q, q, 'homo', oracle=oracle)
        hete_sel = SCoRE_SDR_fast([Lcalib, Lcalib_pred], [None, Ltest_pred], q, q, 'hete', oracle=oracle)
        dtm_sel = SCoRE_SDR_fast([Lcalib, Lcalib_pred], [None, Ltest_pred], q, q, None, oracle=oracle)

        homo_sel_r = SCoRE_SDR_fast([Lcalib, Scalib_pred], [None, Stest_pred], q, q, 'homo', oracle=oracle)
        hete_sel_r = SCoRE_SDR_fast([Lcalib, Scalib_pred], [None, Stest_pred], q, q, 'hete', oracle=oracle)
        dtm_sel_r = SCoRE_SDR_fast([Lcalib, Scalib_pred], [None, Stest_pred], q, q, None, oracle=oracle)

        homo_sdr, _, homo_nsel = eval_SDR(Ltest, np.ones_like(Ltest), homo_sel)
        hete_sdr, _, hete_nsel = eval_SDR(Ltest, np.ones_like(Ltest), hete_sel)
        dtm_sdr,  _, dtm_nsel  = eval_SDR(Ltest, np.ones_like(Ltest), dtm_sel)

        _, _, homo_reward = eval_SDR(Ltest, Rtest, homo_sel)
        _, _, hete_reward = eval_SDR(Ltest, Rtest, hete_sel)
        _, _, dtm_reward  = eval_SDR(Ltest, Rtest, dtm_sel)

        homo_sdr_r, _, homo_nsel_r = eval_SDR(Ltest, np.ones_like(Ltest), homo_sel_r)
        hete_sdr_r, _, hete_nsel_r = eval_SDR(Ltest, np.ones_like(Ltest), hete_sel_r)
        dtm_sdr_r,  _, dtm_nsel_r  = eval_SDR(Ltest, np.ones_like(Ltest), dtm_sel_r)

        _, _, homo_reward_r = eval_SDR(Ltest, Rtest, homo_sel_r)
        _, _, hete_reward_r = eval_SDR(Ltest, Rtest, hete_sel_r)
        _, _, dtm_reward_r  = eval_SDR(Ltest, Rtest, dtm_sel_r)

        SCoRE_df = pd.DataFrame({
            'homo_sdr': [homo_sdr],
            'homo_nsel': [homo_nsel],
            'homo_reward': [homo_reward],
            'hete_sdr': [hete_sdr],
            'hete_nsel': [hete_nsel],
            'hete_reward': [hete_reward],
            'dtm_sdr': [dtm_sdr],
            'dtm_nsel': [dtm_nsel],
            'dtm_reward': [dtm_reward],
            'homo_sdr_r': [homo_sdr_r],
            'homo_nsel_r': [homo_nsel_r],
            'homo_reward_r': [homo_reward_r],
            'hete_sdr_r': [hete_sdr_r],
            'hete_nsel_r': [hete_nsel_r],
            'hete_reward_r': [hete_reward_r],
            'dtm_sdr_r': [dtm_sdr_r],
            'dtm_nsel_r': [dtm_nsel_r],
            'dtm_reward_r': [dtm_reward_r],
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

if not os.path.exists('results'):
    os.makedirs('results')

all_res.to_csv(os.path.join('results', f"SCoRE_SDR, case={case_idx}, setting={setting}, sigma={sig}, tau={tau}, ncalib={ncalib}, ntest={ntest}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}, fit_Y={fit_Y}, oracle_mdl={oracle_mdl}.csv"))
        

