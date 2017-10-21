import numpy as np
import pandas as pd
import os

train_file = '../data/raw-csv/train.csv'
test_file = '../data/raw-csv/test.csv'

#train_df = pd.read_csv(train_file, na_values=['-1', '-1.0'])
#test_df = pd.read_csv(test_file, na_values=['-1', '-1.0'])
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

marker = '4c'
dir_name = 'fe' + marker + '-npy'

# from EDA
binary_feats = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',
                'ps_ind_12_bin', 'ps_ind_13_bin',
                'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',
                'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
                'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']
cat_feats = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',
             'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_04_cat',
             'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat',
             'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat']
int_feats = ['ps_ind_01', 'ps_id_03', 'ps_ind_14', 'ps_ind_15', 'ps_car_11',
             'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07',
             'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11',
             'ps_calc_12', 'ps_calc_13', 'ps_calc_14']
float_feats = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_calc_01',
               'ps_calc_02', 'ps_calc_03', 'ps_car_12', 'ps_car_13',
               'ps_car_14', 'ps_car_15']

# id, target drop
y = train_df['target'].values
id_test = test_df['id'].values

X = train_df.drop(['target', 'id'], axis=1)
X_test = test_df.drop(['id'], axis=1)

'''
# simple filtering (1)
feat_to_drop = ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                'ps_ind_13_bin']
X = X.drop(feat_to_drop, axis=1)
X_test = X_test.drop(feat_to_drop, axis=1)
'''
'''
# new features (4a)
X['sum_NA'] = sum([(X[f] == -1) for f in X.columns])
X_test['sum_NA'] = sum([(X_test[f] == -1) for f in X_test.columns])
'''
'''
# new features (4b)
X['sum_bin'] = sum([X[f] for f in binary_feats])
X_test['sum_bin'] = sum([X_test[f] for f in binary_feats])
'''

# new features (4c)
X['ps_car_15_sqr'] = (X['ps_car_15'])**2
#X = X.drop(['ps_car_15'], axis=1)
X_test['ps_car_15_sqr'] = (X_test['ps_car_15'])**2
#X_test = X_test.drop(['ps_car_15'], axis=1)

'''
# NaN -> median values (3)
for col in X.columns:
    X[col] = X[col].apply(lambda x: None if x == -1 else x)
    X_test[col] = X_test[col].apply(lambda x: None if x == -1 else x)
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X_test[col].median())
'''
if not os.path.exists('../data/' + dir_name):
    os.makedirs('../data/' + dir_name)
np.save('../data/' + dir_name + '/X.npy', X)
np.save('../data/' + dir_name + '/X-test.npy', X_test)
np.save('../data/' + dir_name + '/y.npy', y)
np.save('../data/' + dir_name + '/id-test.npy', id_test)
np.save('../data/' + dir_name + '/feat_names.npy', X.columns)
np.save('../data/' + dir_name + '/X-shape1.npy', X.shape[1])
