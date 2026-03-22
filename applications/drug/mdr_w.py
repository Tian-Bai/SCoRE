import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.special import expit

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
root_dir = os.path.normpath(root_dir)
sys.path.append(root_dir)
from SCoRE import SCoRE_MDR_w
from SCoRE import eval_MDR

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
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
parser.add_argument('--oracle_weight', dest='oracle_weight', type=int, default=0)

args = parser.parse_args()
model = args.model
data_name = args.data
reward_type = args.reward
Nrep = args.Nrep
seedgroup = args.seedgroup
oracle = bool(args.oracle)
oracle_weight = bool(args.oracle_weight)

if model[:3] == 'DGL':
    model = model[4:]
if model == 'rdkit_2d_normalized':
    model = 'rdkit_2d'

# the cost generally ranges from 0 to 3
q_list = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5] 

all_res = pd.DataFrame()

df = pd.read_csv(os.path.join('predicted_results', data_name, model, data_name + '_random_seed1' + '_pred.csv'))

def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp)

print("Training weight estimator...")
df_train = df[df['split'] == 'train']
weights = df_train['w'].to_numpy()
n_train = len(df_train)

indices = np.arange(n_train)
shifted_indices = np.random.choice(indices, size=n_train, replace=True, p=weights/weights.sum())

X_train = np.stack(df_train['SMILES'].apply(get_fp).values)
X_shifted = X_train[shifted_indices]

# Train classifier
X = np.concatenate([X_train, X_shifted], axis=0)
y = np.concatenate([np.zeros(n_train), np.ones(n_train)], axis=0)

cl = RandomForestClassifier()
cl.fit(X, y)

# Predict weights for calibtest data
df_calibtest = df[df['split'] == 'calibtest'].copy()
X_calibtest = np.stack(df_calibtest['SMILES'].apply(get_fp).values)
probs = cl.predict_proba(X_calibtest)[:, 1]

w_pred_calibtest = probs / (1 - probs)
ncalibtest = len(df_calibtest)
w_calibtest = df_calibtest['w'].to_numpy() # all weights

for i_itr in tqdm(range(Nrep * seedgroup, Nrep * (seedgroup + 1))):
    random.seed(i_itr)
    np.random.seed(i_itr)

    calib_idx = np.random.choice(ncalibtest, size=ncalibtest // 2, replace=False) # without shift, simply sample one half
    test_idx = np.empty(0)
    while len(test_idx) <= ncalibtest // 2:
        new_idx = np.random.choice(ncalibtest, size=ncalibtest // 2, replace=False)
        cri = np.random.uniform(0, 1, size=len(new_idx)) <= w_calibtest[new_idx]
        new_idx = new_idx[cri]
        test_idx = np.concatenate((test_idx, new_idx), axis=0)
    test_idx = test_idx[:ncalibtest - len(calib_idx)]

    calib_idx = calib_idx.astype(int)
    test_idx = test_idx.astype(int)

    # this is the unshifted way:
    # calib_idx_noshift, test_idx_noshift = train_test_split(np.arange(ncalibtest), train_size=0.5)
    # df_calib = df_calibtest.iloc[calib_idx.astype(int)]

    df_test = df_calibtest.iloc[test_idx]
    df_calib = df_calibtest.iloc[calib_idx]
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

    if oracle_weight:
        wcalib, wtest = w_calibtest[calib_idx], w_calibtest[test_idx]
    else:
        # use estimated weights
        wcalib, wtest = w_pred_calibtest[calib_idx], w_pred_calibtest[test_idx]

    for q in q_list:
        sel_r = SCoRE_MDR_w([Lcalib, Scalib_pred], [Ltest, Stest_pred], wcalib, wtest, q, q) # considering reward
        sel = SCoRE_MDR_w([Lcalib, Lcalib_pred], [Ltest, Ltest_pred], wcalib, wtest, q, q)   # not considering reward

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

if not os.path.exists('mdr_results_w'):
    os.makedirs('mdr_results_w')

all_res.to_csv(os.path.join('mdr_results_w', f"SCoRE_MDR_w, data={data_name}, model={model}, Nrep={Nrep}, seedgroup={seedgroup}, reward={reward_type}, oracle={oracle}.csv"))
