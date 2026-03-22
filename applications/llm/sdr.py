import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
root_dir = os.path.normpath(root_dir)
sys.path.append(root_dir)
from SCoRE import SCoRE_SDR, SCoRE_SDR_fast
from SCoRE import eval_SDR

import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='.')
parser.add_argument('ntrain', type=int)
parser.add_argument('ncalib', type=int)
parser.add_argument('model', type=str, default='rf', choices = ['rf', 'svm', 'gb'])
parser.add_argument('reward', type=int, default=0)
parser.add_argument('Nrep', type=int)
parser.add_argument('seedgroup', type=int)
parser.add_argument('--oracle', dest='oracle', type=int, default=0)
parser.add_argument('--indiv', dest='indiv', type=int, default=0)

args = parser.parse_args()
ntrain = args.ntrain
ncalib = args.ncalib
model = args.model
reward_type = args.reward
Nrep = args.Nrep
seedgroup = args.seedgroup
oracle = bool(args.oracle)
get_indiv = bool(args.indiv)

q_list = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]

all_res = pd.DataFrame()

df = pd.read_csv(os.path.join('data', 'saved_feature_and_label_new.csv'))
tot_X, tot_Y = df.drop(columns=['accuracy', 'pos_err', 'npos_err', 'not_ub']).to_numpy(), df[['accuracy', 'pos_err', 'npos_err', 'not_ub']].to_numpy()

assert len(tot_X) == len(tot_Y)
tot_num = len(tot_X)

Xtrain, Xcalibtest = tot_X[:ntrain], tot_X[ntrain:]
Ytrain, Ycalibtest = tot_Y[:ntrain], tot_Y[ntrain:]

ncalibtest = tot_num - ntrain
ntest = ncalibtest - ncalib

print('train mu')
if model == 'rf':
    mu = RandomForestRegressor()
    mu_reward = RandomForestRegressor()
    mu.fit(Xtrain, Ytrain[:, 1] + 0.5 * Ytrain[:, 2]) # Y is a weighted notion of error where false negative is 1 and false positive is 0.5
    mu_reward.fit(Xtrain, Ytrain[:, 0] + 3 * Ytrain[:, 3])
else:
    raise NotImplementedError

print('SCoRE...')
for i_itr in tqdm(range(Nrep * seedgroup, Nrep * (seedgroup + 1))):
    random.seed(i_itr)
    np.random.seed(i_itr)

    rand_perm = np.random.permutation(ncalibtest)
    Xcalib, Xtest = Xcalibtest[rand_perm[:ncalib]], Xcalibtest[rand_perm[ncalib:]]
    Ycalib, Ytest = Ycalibtest[rand_perm[:ncalib]], Ycalibtest[rand_perm[ncalib:]]

    # L(f, X, Y) = Y[:, 1] + 0.5 * Y[:, 2]
    Lcalib, Lcalib_pred = Ycalib[:, 1] + 0.5 * Ycalib[:, 2], mu.predict(Xcalib)
    Ltest, Ltest_pred = Ytest[:, 1] + 0.5 * Ytest[:, 2], mu.predict(Xtest)

    # r(X, Y) = Y[:, 0] + 3 * Y[:, 3] (give more weight to those correct, non-ambiguous answers and weight 1 to other correct answers)
    if reward_type == 0:
        Rcalib, Rcalib_pred = Ycalib[:, 0] + 3 * Ycalib[:, 3], mu_reward.predict(Xcalib)
        Rtest, Rtest_pred = Ytest[:, 0] + 3 * Ytest[:, 3], mu_reward.predict(Xtest)

    for q in q_list:
        Scalib_pred, Stest_pred = (Lcalib_pred - q) / Rcalib_pred, (Ltest_pred - q) / Rtest_pred

        homo_sel = SCoRE_SDR_fast([Lcalib, Lcalib_pred], [None, Ltest_pred], q, q, 'homo')
        hete_sel = SCoRE_SDR_fast([Lcalib, Lcalib_pred], [None, Ltest_pred], q, q, 'hete')
        dtm_sel = SCoRE_SDR_fast([Lcalib, Lcalib_pred], [None, Ltest_pred], q, q, None)

        homo_sel_r = SCoRE_SDR_fast([Lcalib, Scalib_pred], [None, Stest_pred], q, q, 'homo')
        hete_sel_r = SCoRE_SDR_fast([Lcalib, Scalib_pred], [None, Stest_pred], q, q, 'hete')
        dtm_sel_r = SCoRE_SDR_fast([Lcalib, Scalib_pred], [None, Stest_pred], q, q, None)

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
            'seed': [i_itr],
            'ncalib': [ncalib],
            'ntest': [ntest],
            'reward_type': [reward_type],
            'oracle': [oracle]
        })

        all_res = pd.concat((all_res, SCoRE_df))

        if i_itr == Nrep * seedgroup and get_indiv:
            df_indiv = pd.DataFrame({
                'Ytest_acc': Ytest[:, 0].tolist(),
                'Ytest_pos': Ytest[:, 1].tolist(),
                'Ytest_npos': Ytest[:, 2].tolist(),
                'Ytest_ub': Ytest[:, 3].tolist(),
                'Ltest': Ltest.tolist(),
                'Rtest': Rtest.tolist(),
                'nsel_selected': [int(i in dtm_sel) for i in range(ntest)],
                'reward_selected': [int(i in dtm_sel_r) for i in range(ntest)],
            })

            df_indiv.to_csv(os.path.join('indiv', f'indiv_sdr_q={q}.csv'))

if not os.path.exists('sdr_results'):
    os.makedirs('sdr_results')

all_res.to_csv(os.path.join('sdr_results', f"SCoRE_SDR, model={model}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}.csv"))
        

