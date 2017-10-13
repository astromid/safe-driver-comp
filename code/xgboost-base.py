import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sub_path = '../subs/'
dir_name = 'fe4b-npy'

X = np.load('../data/' + dir_name + '/X.npy')
y = np.load('../data/' + dir_name + '/y.npy')

params = {  'max_depth' : 6,
            'learning_rate' : 0.03,
            'reg_lambda' : 4.0,
            'n_estimators' : 916}
model = XGBClassifier(**params)
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=skf, n_jobs=-1, verbose=1)

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
sub_df.to_csv(sub_path + 'xgb-4b-' + sub_name + '.csv', index=False)
print('mean cv score: ', mean_cv)
print('Submission file created')
