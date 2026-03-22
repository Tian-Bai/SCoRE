import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
root_dir = os.path.normpath(root_dir)
sys.path.append(root_dir)
from SCoRE import SCoRE_MDR, eval_MDR

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

args = parser.parse_args()
ntrain = args.ntrain
ncalib = args.ncalib
model = args.model
reward_type = args.reward
Nrep = args.Nrep
seedgroup = args.seedgroup
oracle = bool(args.oracle)

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

# we use this artificial reward: factuality degree * lexical similarity -X[:, 1] (to simulate the 'severeity' of patient disease)

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

    Scalib_pred, Stest_pred = Lcalib_pred / Rcalib_pred, Ltest_pred / Rtest_pred

    for q in q_list:
        sel_r = SCoRE_MDR([Lcalib, Scalib_pred], [Ltest, Stest_pred], q, q) # considering reward
        sel = SCoRE_MDR([Lcalib, Lcalib_pred], [Ltest, Ltest_pred], q, q)   # not considering reward

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
            'seed': [i_itr],
            'ncalib': [ncalib],
            'ntest': [ntest],
            'reward_type': [reward_type],
            'oracle': [oracle]
        })

        all_res = pd.concat((all_res, SCoRE_df))

if not os.path.exists('mdr_results'):
    os.makedirs('mdr_results')

all_res.to_csv(os.path.join('mdr_results', f"SCoRE_MDR, model={model}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}.csv"))
        

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""