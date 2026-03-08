# get deeppurpose predictions
# loss: 1{Y <= c}
# reward: Y 1{Y > c} or dissimilarity with a hold-out set (training set?)

from DeepPurpose import utils, CompoundPred
from tdc.utils import create_fold
from tdc.single_pred import ADME, Tox, HTS
import argparse
import sascorer
import pandas as pd
import pickle
import random
import os
from scipy.special import expit

# Let's only consider ADME datasets for now
# regression adme datasets
adme_regression_dataset = ['caco2_wang', 'lipophilicity_astrazeneca', 'ppbr_az', 'vdss_lombardo',  'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az']
tox_regression_dataset = ['ld50_zhu']

data_thresholds = {'caco2_wang': -4.7, 'lipophilicity_astrazeneca': 2.9, 'solubility_aqsoldb': -1.5, 'ppbr_az': 98, 
                   'vdss_lombardo': 2.0, 'half_life_obach': 9.0, 'clearance_microsome_az': 33, 'clearance_hepatocyte_az': 50, 
                   'ld50_zhu': 2.9} # 30% good, same as in optcs

parser = argparse.ArgumentParser('')

parser.add_argument('--model', type=str, default='DGL_AttentiveFP', choices = ['DGL_AttentiveFP', 'Morgan', 'CNN', 'rdkit_2d_normalized', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred'])
parser.add_argument('--split_fct', type=str, default='random', choices = ['random'])

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--data', type=str, default='caco2_wang', choices = adme_regression_dataset + tox_regression_dataset)
parser.add_argument('--pretrain', action="store_true", default=False)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--get_hid_emb', action="store_true", default=False)

args = parser.parse_args()
split_fct = args.split_fct
device = args.device
data_name = args.data
drug_encoding = args.model
seed = args.seed

random.seed(seed)

task = 'admet'
if data_name in adme_regression_dataset + tox_regression_dataset:
    task_type = 'regression'
else:
    task_type = 'classification'
        
if data_name in adme_regression_dataset:
    df = ADME(name=data_name).get_data()
elif data_name in tox_regression_dataset:
    df = Tox(name=data_name).get_data()
batch_size = 128

threshold = data_thresholds[data_name]

# filter the molecules
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm
tqdm.pandas()
def check_mol(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    else:
        return True
df['molecule_check'] = df.Drug.progress_apply(lambda x: check_mol(x))
df = df[df.molecule_check].reset_index(drop = True)

if args.split_fct == 'random':
    df_split = create_fold(df, seed, [0.3, 0.1, 0.6])
    df_train = df_split['train']
    df_val = df_split['valid']
    df_calibtest = df_split['test'] # calib + test
    
# process the data according to given encoding
def dp_data_process(df_train, df_val, df_calibtest, drug_encoding):

    train = utils.data_process(X_drug = df_train.Drug.values, y = df_train.Y.values, 
                            drug_encoding = drug_encoding,
                            split_method='no_split')
    val = utils.data_process(X_drug = df_val.Drug.values, y = df_val.Y.values, 
                            drug_encoding = drug_encoding,
                            split_method='no_split')
    calibtest = utils.data_process(X_drug = df_calibtest.Drug.values, y = df_calibtest.Y.values, 
                            drug_encoding = drug_encoding,
                            split_method='no_split')
    
    def get_mw(smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        return Descriptors.MolWt(mol)

    # Apply the function to your SMILES column
    train['mw'] = train.SMILES.progress_apply(get_mw)
    val['mw'] = val.SMILES.progress_apply(get_mw)
    calibtest['mw'] = calibtest.SMILES.progress_apply(get_mw)

    # try different options?
    train['w'] = expit(abs(train['mw'] - 400) / 400)
    val['w'] = expit(abs(val['mw'] - 400) / 400)
    calibtest['w'] = expit(abs(calibtest['mw'] - 400) / 400)

    # for risk: sa_score, only a function of x
    def get_sa(smi):
        mol = Chem.MolFromSmiles(smi)
        return sascorer.calculateScore(mol)
    train['sa'] = train.SMILES.progress_apply(get_sa)
    val['sa'] = val.SMILES.progress_apply(get_sa)
    calibtest['sa'] = calibtest.SMILES.progress_apply(get_sa)

    # for risk: indicator of false selection
    train['1{Y<=c}'] = 1 * (train['Label'] <= threshold)
    val['1{Y<=c}'] = 1 * (val['Label'] <= threshold)
    calibtest['1{Y<=c}'] = 1 * (calibtest['Label'] <= threshold)

    # for reward: 1 - tanimoto(Y, Dtrain)
    train_mol = [Chem.MolFromSmiles(x) for x in train.SMILES]
    train_fin = [AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048) for x in train_mol]

    def get_avg_tanimoto(smi):
        mol = Chem.MolFromSmiles(smi)
        fin = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        avg_tanimoto = 0
        for x_fin in train_fin:
            avg_tanimoto += TanimotoSimilarity(fin, x_fin)
        avg_tanimoto /= len(train_mol)
        return avg_tanimoto
    train['tanimoto'] = train.SMILES.progress_apply(get_avg_tanimoto)
    val['tanimoto'] = val.SMILES.progress_apply(get_avg_tanimoto)
    calibtest['tanimoto'] = calibtest.SMILES.progress_apply(get_avg_tanimoto)

    return train, val, calibtest  

# processed df's
train, val, calibtest = dp_data_process(df_train, df_val, df_calibtest, drug_encoding)

print(train)

config = utils.generate_config(drug_encoding = drug_encoding, 
                               train_epoch = args.epoch, 
                               batch_size = batch_size)
config['device'] = device

# predict: Y, 1{Y<=c} separately
# since the DeepPurpose package use df.Label as the default response, we would need to rename
# 1. predict Y (in case we would use)
save_model_path = os.path.join('model', data_name + '_' + drug_encoding + '_seed' + str(seed) + '_Y')
    
if True:
    model = CompoundPred.model_pretrained(save_model_path)
else:    
    config['gnn_hid_dim_drug'] = 512
    model = CompoundPred.model_initialize(**config)
    model.train(train, val)    
    model.save_model(save_model_path)

calibtest['pred_Y'] = model.predict(calibtest)
train['pred_Y'] = model.predict(train)
val['pred_Y'] = model.predict(val)

# 2. predict 1{Y<=c}
train.rename(columns={'Label': 'Y'}, inplace=True)
calibtest.rename(columns={'Label': 'Y'}, inplace=True)
val.rename(columns={'Label': 'Y'}, inplace=True)
train.rename(columns={'1{Y<=c}': 'Label'}, inplace=True)
calibtest.rename(columns={'1{Y<=c}': 'Label'}, inplace=True)
val.rename(columns={'1{Y<=c}': 'Label'}, inplace=True)

save_model_path = os.path.join('model', data_name + '_' + drug_encoding + '_seed' + str(seed) + '_Yc')
    
if True:
    model = CompoundPred.model_pretrained(save_model_path)
else:    
    config['gnn_hid_dim_drug'] = 512
    model = CompoundPred.model_initialize(**config)
    model.train(train, val)    
    model.save_model(save_model_path)

calibtest['pred_1{Y<=c}'] = model.predict(calibtest)
train['pred_1{Y<=c}'] = model.predict(train)
val['pred_1{Y<=c}'] = model.predict(val)

train.rename(columns={'Label': '1{Y<=c}'}, inplace=True)
calibtest.rename(columns={'Label': '1{Y<=c}'}, inplace=True)
val.rename(columns={'Label': '1{Y<=c}'}, inplace=True)
train.rename(columns={'w': 'Label'}, inplace=True)
calibtest.rename(columns={'w': 'Label'}, inplace=True)
val.rename(columns={'w': 'Label'}, inplace=True)

# 3. predict w = expit(abs(mw - 400) / 400) (only for covariate shift case)
# ----------------------------------------------- ???? -----------------------------------------
save_model_path = os.path.join('model', data_name + '_' + drug_encoding + '_seed' + str(seed) + '_w')

if args.pretrain:
    model = CompoundPred.model_pretrained(save_model_path)
else:    
    config['gnn_hid_dim_drug'] = 512
    model = CompoundPred.model_initialize(**config)
    model.train(train, val)    
    model.save_model(save_model_path)

calibtest['pred_w'] = model.predict(calibtest)
train['pred_w'] = model.predict(train)
val['pred_w'] = model.predict(val)

# finally rename back
train.rename(columns={'Label': 'w'}, inplace=True)
calibtest.rename(columns={'Label': 'w'}, inplace=True)
val.rename(columns={'Label': 'w'}, inplace=True)

train['split'] = 'train'
calibtest['split'] = 'calibtest'
val['split'] = 'val'

import pandas as pd
df = pd.concat([train, val, calibtest])

df = df[['SMILES', 'split', 'Y', '1{Y<=c}', 'sa', 'mw', 'w', 'pred_w', 'tanimoto', 'pred_Y', 'pred_1{Y<=c}']]
    
if drug_encoding[:3] == 'DGL':
    drug_encoding = drug_encoding[4:]
    
if drug_encoding == 'rdkit_2d_normalized':
    drug_encoding = 'rdkit_2d'

save_folder = os.path.join('predicted_results', data_name, drug_encoding)
save_path = os.path.join(save_folder, data_name + '_' + split_fct + '_seed' + str(seed) + '_pred.csv')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

df.to_csv(save_path, index = False)

if args.get_hid_emb:
    df_ = pd.concat([train, val, calibtest])
        
    hid = model.get_hidden_emb(df_)
    print(hid.shape)

    save_path = os.path.join(save_folder, data_name + '_' + split_fct + '_seed' + str(seed) + '_hid.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(hid, f)