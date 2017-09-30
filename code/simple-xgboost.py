import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from gini import gini_xgb

train_path = '../data/train.csv'
test_path = '../data/test.csv'
sub_path = '../subs/'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

y = train_df['target'].values
id_test = test_df['id'].values

X = train_df.drop(['target', 'id'], axis=1)
X_test = test_df.drop(['id'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

d_train = xgb.DMatrix(X_train, y_train)
d_val = xgb.DMatrix(X_val, y_val)
d_test = xgb.DMatrix(X_test)

params = {}
params['objective'] = 'binary:logistic'
watchlist = [(d_train, 'train'), (d_val, 'val')]
model = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=200,
feval=gini_xgb, maximize=True, verbose_eval=10)

p_test = model.predict(d_test)

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = p_test
sub.to_csv(sub_path + 'xgb0.csv', index=False)
