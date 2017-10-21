import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sub_path = '../subs/'
marker = '4c'
dir_name = 'fe' + marker + '-npy'

X = np.load('../data/' + dir_name + '/X.npy')
y = np.load('../data/' + dir_name + '/y.npy')

params = {  'eta': 0.02,
            'max_depth': 4,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': 42}
model = XGBClassifier(**params)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=skf, n_jobs=4, verbose=1)
np.save('../data' + dir_name + '/xgb-' + marker + '-cv.npy', scores)
mean_cv = scores.mean()
model = XGBClassifier(**params)
model.fit(X, y, verbose=True)
X_test = np.load('../data/' + dir_name + '/X-test.npy')
p_test = model.predict_proba(X_test)[:, 1]

sub_df = pd.DataFrame()
id_test = np.load('../data/' + dir_name + '/id-test.npy')
sub_df['id'] = id_test
sub_df['target'] = p_test
sub_name = str(mean_cv)
sub_df.to_csv(sub_path + 'xgb-' + marker + '-' + sub_name + '.csv', index=False)
print('mean cv score: ', mean_cv)
print('Submission file created')
