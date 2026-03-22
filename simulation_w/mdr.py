import numpy as np
import pandas as pd
import random
import sys
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.special import expit

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
root_dir = os.path.normpath(root_dir)
sys.path.append(root_dir)
from SCoRE import eval_MDR, gen_data_1, gen_data_2, gen_data_Jin2023, loss_1, loss_2, loss_Jin2023, SCoRE_MDR_w
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

parser.add_argument('--reward', dest='reward', type=int, default=0)
parser.add_argument('--weight', dest='weight', type=int, default=0)
parser.add_argument('--fit_Y', dest='fit_Y', type=int, default=0)
parser.add_argument('--oracle', dest='oracle', type=int, default=0)
parser.add_argument('--oracle_mdl', dest='oracle_mdl', type=int, default=0)
parser.add_argument('--oracle_weight', dest='oracle_weight', type=int, default=0)

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
weight_type = args.weight
fit_Y = bool(args.fit_Y)
oracle = bool(args.oracle)
oracle_mdl = bool(args.oracle_mdl)
oracle_weight = bool(args.oracle_weight)

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

# define the weight function
if weight_type == 0:
    def w(x):
        x_prime = x[:,:5] if x.shape[1] >= 5 else x
        return expit(0.1 * np.sum(x_prime, axis=1))
elif weight_type == 1:
    assert dim >= 4
    def w(x):
        x_prime = x[:,:4]
        nonlinear = np.sum(x_prime[:,:3] * x_prime[:,1:4], axis=1)
        osc = np.sin(x_prime[:,0] + x_prime[:,1])
        return expit(0.5 * nonlinear + 0.3 * osc)
elif weight_type == 2:
    assert dim >= 3
    def w(x):
        x_prime = x[:,:3]
        c1 = np.array([2.0, -1.0, 1.0])
        c2 = np.array([-2.0, 1.0, -1.0])
        g = np.exp(-np.sum((x_prime - c1) ** 2, axis=1)) + 0.7 * np.exp(-np.sum((x_prime - c2) ** 2, axis=1))
        return expit(3.0 * g - 2.0)

all_res = pd.DataFrame()

for i_itr in tqdm(range(Nrep * seedgroup, Nrep * (seedgroup + 1))):
    random.seed(i_itr)
    np.random.seed(i_itr)
    if case_idx == 1:
        Xtrain, _, _, Ytrain = gen_data_1(setting, 1000, sig, dim)
        Xcalib, _, _, Ycalib = gen_data_1(setting, ncalib, sig, dim)
        Xtest, Ytest = np.empty((0, dim)), np.empty(0)

        # shifted data with dQ/dP = w, use rejection sampling
        while len(Xtest) < ntest:
            Xnew, _, _, Ynew = gen_data_1(setting, ntest, sig, dim)
            cri = np.random.uniform(0, 1, size=len(Xnew)) <= w(Xnew) # assume w upper bound is 1
            Xnew, Ynew = Xnew[cri], Ynew[cri]
            Xtest = np.concatenate((Xtest, Xnew), axis=0)
            Ytest = np.concatenate((Ytest, Ynew), axis=0)
        Xtest, Ytest = Xtest[:ntest], Ytest[:ntest]

        if not oracle_weight:
            Xtrain_w = np.empty((0, dim))
            while len(Xtrain_w) <= 1000:
                Xnew, _, _, _ = gen_data_1(setting, 1000, sig, dim)
                cri = np.random.uniform(0, 1, size=len(Xnew)) <= w(Xnew)
                Xnew = Xnew[cri]
                Xtrain_w = np.concatenate((Xtrain_w, Xnew), axis=0)
            Xtrain_w = Xtrain_w[:1000]        

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
        Xtest, Ytest = np.empty((0, dim)), np.empty(0)

        # shifted data with dQ/dP = w, use rejection sampling
        while len(Xtest) < ntest:
            Xnew, _, _, Ynew = gen_data_2(setting, ntest, sig, dim)
            cri = np.random.uniform(0, 1, size=len(Xnew)) <= w(Xnew) # assume w upper bound is 1
            Xnew, Ynew = Xnew[cri], Ynew[cri]
            Xtest = np.concatenate((Xtest, Xnew), axis=0)
            Ytest = np.concatenate((Ytest, Ynew), axis=0)
        Xtest, Ytest = Xtest[:ntest], Ytest[:ntest]

        if not oracle_weight:
            Xtrain_w = np.empty((0, dim))
            while len(Xtrain_w) <= 1000:
                Xnew, _, _, _ = gen_data_2(setting, 1000, sig, dim)
                cri = np.random.uniform(0, 1, size=len(Xnew)) <= w(Xnew)
                Xnew = Xnew[cri]
                Xtrain_w = np.concatenate((Xtrain_w, Xnew), axis=0)
            Xtrain_w = Xtrain_w[:1000]
        
        Ltrain, Lcalib, Ltest = loss_2(Ytrain, f, Xtrain, clip_const), loss_2(Ycalib, f, Xcalib, clip_const), loss_2(Ytest, f, Xtest, clip_const)
        Rtrain, Rcalib, Rtest = r(Xtrain, Ytrain), r(Xcalib, Ycalib), r(Xtest, Ytest)

    if case_idx == 3:
        Xtrain, _, _, Ytrain = gen_data_Jin2023(setting, 1000, sig, dim)
        Xcalib, _, _, Ycalib = gen_data_Jin2023(setting, ncalib, sig, dim)
        Xtest, Ytest = np.empty((0, dim)), np.empty(0)

        # shifted data with dQ/dP = w, use rejection sampling
        while len(Xtest) < ntest:
            Xnew, _, _, Ynew = gen_data_Jin2023(setting, ntest, sig, dim)
            cri = np.random.uniform(0, 1, size=len(Xnew)) <= w(Xnew) # assume w upper bound is 1
            Xnew, Ynew = Xnew[cri], Ynew[cri]
            Xtest = np.concatenate((Xtest, Xnew), axis=0)
            Ytest = np.concatenate((Ytest, Ynew), axis=0)
        Xtest, Ytest = Xtest[:ntest], Ytest[:ntest]

        if not oracle_weight:
            Xtrain_w = np.empty((0, dim))
            while len(Xtrain_w) <= 1000:
                Xnew, _, _, _ = gen_data_Jin2023(setting, 1000, sig, dim)
                cri = np.random.uniform(0, 1, size=len(Xnew)) <= w(Xnew)
                Xnew = Xnew[cri]
                Xtrain_w = np.concatenate((Xtrain_w, Xnew), axis=0)
            Xtrain_w = Xtrain_w[:1000]

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

    if oracle_weight:
        wcalib, wtest = w(Xcalib), w(Xtest)
    else:
        # get weight estimate
        cl = RandomForestClassifier()
        cl.fit(np.concatenate((Xtrain, Xtrain_w), axis=0), np.concatenate((np.zeros(1000), np.ones(1000))))
        pred_calib = cl.predict_proba(Xcalib)[:, 1]
        pred_test = cl.predict_proba(Xtest)[:, 1]
        wcalib = pred_calib / (1 - pred_calib)
        wtest = pred_test / (1 - pred_test)

    for q in q_list:
        sel = SCoRE_MDR_w([Lcalib, Lcalib_pred], [None, Ltest_pred], wcalib, wtest, q, q)
        sel_r = SCoRE_MDR_w([Lcalib, Scalib_pred], [None, Stest_pred], wcalib, wtest, q, q)

        # see the number of selection. this will be equal to power with reward type 0
        mdr, nsel = eval_MDR(Ltest, np.ones_like(Ltest), sel)
        _, reward = eval_MDR(Ltest, Rtest, sel)

        mdr_r, nsel_r = eval_MDR(Ltest, np.ones_like(Ltest), sel_r)
        _, reward_r = eval_MDR(Ltest, Rtest, sel_r)

        SCoRE_df = pd.DataFrame({
            'mdr': [mdr],
            'nsel': [nsel],
            'reward': [reward],
            'mdr_r': [mdr_r],
            'nsel_r': [nsel_r],
            'reward_r': [reward_r],
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

if not os.path.exists('mdr_results'):
    os.makedirs('mdr_results')

all_res.to_csv(os.path.join('mdr_results', f"SCoRE_MDR_w{weight_type}, case={case_idx}, setting={setting}, sigma={sig}, tau={tau}, ncalib={ncalib}, ntest={ntest}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}, fit_Y={fit_Y}, oracle_mdl={oracle_mdl}, oracle_w={oracle_weight}.csv"))