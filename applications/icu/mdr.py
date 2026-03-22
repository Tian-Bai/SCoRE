import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
root_dir = os.path.normpath(root_dir)
sys.path.append(root_dir)
from SCoRE import SCoRE_MDR
from SCoRE import eval_MDR

import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='.')
parser.add_argument('ntrain', type=int)
parser.add_argument('ncalib', type=int)
parser.add_argument('model', type=str, default='rf', choices = ['rf', 'svm', 'gb'])
parser.add_argument('loss', type=int, default=0)
parser.add_argument('reward', type=int, default=0)
parser.add_argument('Nrep', type=int)
parser.add_argument('seedgroup', type=int)
parser.add_argument('--oracle', dest='oracle', type=int, default=0)
parser.add_argument('--indiv', dest='indiv', type=int, default=0)

args = parser.parse_args()
ntrain = args.ntrain
ncalib = args.ncalib
model = args.model
loss_type = args.loss
reward_type = args.reward
Nrep = args.Nrep
seedgroup = args.seedgroup
oracle = bool(args.oracle)
get_indiv = bool(args.indiv)

# q_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
q_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

all_res = pd.DataFrame()

df_X = pd.read_csv(os.path.join('data_f', 'icu_features_transformed_w.csv'), index_col=0)
df_Y = pd.read_csv(os.path.join('data_f', 'icu_responses_pred_w.csv'), index_col=0)

assert len(df_X) == len(df_Y)
tot_num = len(df_X)

tot_X, tot_Y, tot_f = df_X.to_numpy(), df_Y['los'].to_numpy(), df_Y['pred_los'].to_numpy()
Xtrain, Xcalibtest = tot_X[:ntrain], tot_X[ntrain:]
Ytrain, Ycalibtest = tot_Y[:ntrain], tot_Y[ntrain:]
ftrain, fcalibtest = tot_f[:ntrain], tot_f[ntrain:]

ncalibtest = tot_num - ntrain
ntest = ncalibtest - ncalib

# train L and r predictor
print('train mu')
if model == 'rf':
    mu = RandomForestRegressor()
elif model == 'gb':
    mu = GradientBoostingRegressor()
else:
    raise NotImplementedError

if loss_type == 0:
    mu.fit(Xtrain, (Ytrain - ftrain) ** 2) # error model
elif loss_type == 1:
    mu.fit(Xtrain, np.maximum(Ytrain - ftrain, 0))
elif loss_type == 2:
    mu.fit(Xtrain, (Ytrain - ftrain) ** 2 * ((Ytrain - ftrain) ** 2 > 0.7))

print('SCoRE...')
for i_itr in tqdm(range(Nrep * seedgroup, Nrep * (seedgroup + 1))):
    random.seed(i_itr)
    np.random.seed(i_itr)

    rand_perm = np.random.permutation(ncalibtest)
    Xcalib, Xtest = Xcalibtest[rand_perm[:ncalib]], Xcalibtest[rand_perm[ncalib:]]
    Ycalib, Ytest = Ycalibtest[rand_perm[:ncalib]], Ycalibtest[rand_perm[ncalib:]]
    fcalib, ftest = fcalibtest[rand_perm[:ncalib]], fcalibtest[rand_perm[ncalib:]]

    if loss_type == 0:
        Lcalib, Lcalib_pred = (Ycalib - fcalib) ** 2, mu.predict(Xcalib)
        Ltest, Ltest_pred = (Ytest - ftest) ** 2, mu.predict(Xtest)
    elif loss_type == 1:
        Lcalib, Lcalib_pred = np.maximum(Ycalib - fcalib, 0), mu.predict(Xcalib)
        Ltest, Ltest_pred = np.maximum(Ytest - ftest, 0), mu.predict(Xtest)
    elif loss_type == 2:
        Lcalib, Lcalib_pred = (Ycalib - fcalib) ** 2 * ((Ycalib - fcalib) ** 2 > 0.7), mu.predict(Xcalib)
        Ltest, Ltest_pred = (Ytest - ftest) ** 2 * ((Ytest - ftest) ** 2 > 0.7), mu.predict(Xtest)

    if reward_type == 0:
        # r(X, Y) = Y
        Rcalib, Rcalib_pred = Ycalib, fcalib
        Rtest, Rtest_pred = Ytest, ftest

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

        if i_itr == Nrep * seedgroup and get_indiv: # first seed only
            df_indiv = pd.DataFrame({
                'Ytest': Ytest.tolist(),
                'Ltest': Ltest.tolist(),
                'ftest': ftest.tolist(),
                'Ltest_pred': Ltest_pred.tolist(),
                'Rtest': Rtest.tolist(),
                'Rtest_pred': Rtest_pred.tolist(),
                'nsel_selected': [int(i in sel) for i in range(ntest)],
                'reward_selected': [int(i in sel_r) for i in range(ntest)],
            })

            df_indiv.to_csv(os.path.join('indiv', f'indiv_mdr_q={q}_loss={loss_type}_reward={reward_type}.csv'))

if not os.path.exists('mdr_results'):
    os.makedirs('mdr_results')

all_res.to_csv(os.path.join('mdr_results', f"SCoRE_MDR, model={model}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}.csv"))
        

