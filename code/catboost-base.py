import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sub_path = '../subs/'
marker = '4c'
dir_name = 'fe' + marker + '-npy'

X = np.load('../data/' + dir_name + '/X.npy')
y = np.load('../data/' + dir_name + '/y.npy')

hopt_params = np.load('best-params.npy').item()
'''
# category features (2)
cat_feats = [1, 3, 4, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
fit_params = {'cat_features' : cat_feats}
'''
model = CatBoostClassifier(**hopt_params)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=skf, n_jobs=4,
                            verbose=1)
np.save('../data/' + dir_name + '/catb-' + marker + '-cv.npy', scores)
mean_cv = scores.mean()
model = CatBoostClassifier(**hopt_params)
model.fit(X, y, verbose=True)
np.save('../pics/feat-imp-npy/' + dir_name + '-catb.npy', model._feature_importance)
X_test = np.load('../data/' + dir_name + '/X-test.npy')
p_test = model.predict_proba(X_test)[:, 1]

sub_df = pd.DataFrame()
id_test = np.load('../data/' + dir_name + '/id-test.npy')
sub_df['id'] = id_test
sub_df['target'] = p_test
sub_name = str(mean_cv)
sub_df.to_csv(sub_path + 'catb-' + marker + '-' + sub_name + '.csv', index=False)
print('mean cv score: ', mean_cv)
print('Submission file created')
