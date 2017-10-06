import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
# from sklearn.model_selection

data_path = '../data/'
sub_path = '../subs/'

X = np.load(data_path + 'X-filt.npy')
y = np.load(data_path + 'y.npy')
id_test = np.load(data_path + 'id-test.npy')
X_test = np.load(data_path + 'X-test-filt.npy')

model = CatBoostClassifier()
model.fit(X, y, verbose=True)
np.save('../pics/feat-imp.npy', model._feature_importance)
p_test = model.predict_proba(X_test)[:, 1]

sub_df = pd.DataFrame()
sub_df['id'] = id_test
sub_df['target'] = p_test
sub_df.to_csv(sub_path + 'catb-filt-fe-2.csv', index=False)
print('Submission file created')
