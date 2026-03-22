import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
root_dir = os.path.normpath(root_dir)
sys.path.append(root_dir)
from SCoRE import SCoRE_MDR, eval_MDR

import argparse
import random
from tqdm import tqdm

adme_regression_dataset = ['caco2_wang', 'lipophilicity_astrazeneca', 'ppbr_az', 'vdss_lombardo', 'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az']
tox_regression_dataset = ['ld50_zhu']

data_thresholds = {'caco2_wang': -4.7, 'lipophilicity_astrazeneca': 2.9, 'solubility_aqsoldb': -1.5, 'ppbr_az': 98, 
                   'vdss_lombardo': 2.0, 'half_life_obach': 9.0, 'clearance_microsome_az': 33, 'clearance_hepatocyte_az': 50, 
                   'ld50_zhu': 2.9} # 30% good, same as in optcs

parser = argparse.ArgumentParser(description='.')
parser.add_argument('data', type=str, default='caco2_wang', choices = adme_regression_dataset + tox_regression_dataset)
parser.add_argument('model', type=str, default='DGL_AttentiveFP', choices = ['DGL_AttentiveFP', 'Morgan', 'CNN', 'rdkit_2d_normalized', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred'])
parser.add_argument('reward', type=int, default=0)
parser.add_argument('Nrep', type=int)
parser.add_argument('seedgroup', type=int)
parser.add_argument('--oracle', dest='oracle', type=int, default=0)

args = parser.parse_args()
model = args.model
data_name = args.data
reward_type = args.reward
Nrep = args.Nrep
seedgroup = args.seedgroup
oracle = bool(args.oracle)

if model[:3] == 'DGL':
    model = model[4:]
if model == 'rdkit_2d_normalized':
    model = 'rdkit_2d'

# the cost generally ranges from 0 to 3
q_list = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

all_res = pd.DataFrame()

df = pd.read_csv(os.path.join('predicted_results', data_name, model, data_name + '_random_seed1' + '_pred.csv'))
df_calibtest = df[df['split'] == 'calibtest']
ncalibtest = len(df_calibtest)

for i_itr in tqdm(range(Nrep * seedgroup, Nrep * (seedgroup + 1))):
    random.seed(i_itr)
    np.random.seed(i_itr)

    calib_idx, test_idx = train_test_split(np.arange(ncalibtest), train_size=0.5)
    df_calib = df_calibtest.iloc[calib_idx]
    df_test = df_calibtest.iloc[test_idx]
    ncalib, ntest = len(df_calib), len(df_test)

    sa_calib = df_calib['sa'].to_numpy()
    sa_test = df_test['sa'].to_numpy()

    Lcalib, Lcalib_pred = sa_calib * df_calib['1{Y<=c}'].to_numpy(), sa_calib * df_calib['pred_1{Y<=c}'].to_numpy()
    Ltest, Ltest_pred = sa_test * df_test['1{Y<=c}'].to_numpy(), sa_test * df_test['pred_1{Y<=c}'].to_numpy()

    if reward_type == 0: # Y
        Rcalib, Rcalib_pred = df_calib['Y'].to_numpy(), df_calib['pred_Y'].to_numpy()
        Rtest, Rtest_pred = df_test['Y'].to_numpy(), df_test['pred_Y'].to_numpy()

        rmin = np.min(np.concatenate([Rcalib, Rtest]))
        if rmin <= 0: # to avoid negative original Y
            offset = -rmin + 1
            Rcalib, Rcalib_pred, Rtest, Rtest_pred = Rcalib + offset, Rcalib_pred + offset, Rtest + offset, Rtest_pred + offset

    if reward_type == 1: # 1 - tanimoto similarity
        # we note that this reward only depends on X, and thus is observable
        Rcalib, Rcalib_pred = 1 - df_calib['tanimoto'].to_numpy(), 1 - df_calib['tanimoto'].to_numpy()
        Rtest, Rtest_pred = 1 - df_test['tanimoto'].to_numpy(), 1 - df_test['tanimoto'].to_numpy()

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

all_res.to_csv(os.path.join('mdr_results', f"SCoRE_MDR, data={data_name}, model={model}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}.csv"))
        

