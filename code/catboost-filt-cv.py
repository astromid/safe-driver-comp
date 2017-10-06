import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
sns.set()

data_path = '../data/'
sub_path = '../subs/'

X = np.load(data_path + 'X-filt.npy')
y = np.load(data_path + 'y.npy')
id_test = np.load(data_path + 'id-test.npy')
X_test = np.load(data_path + 'X-test-filt.npy')
feat_names = np.load(data_path + 'feat_names.npy')

model = CatBoostClassifier(class_weights=[0.5, 5.0])
model.fit(X, y, verbose=True)

order = np.argsort(model._feature_importance)
plt.figure(figsize=[10, 15])
plt.plot(np.array(model._feature_importance)[order], range(len(order)), marker='o')
plt.hlines(range(len(order)), np.zeros_like(order), np.array(model._feature_importance)[order], linestyles=':')
plt.yticks(range(X.shape[1]), feat_names[order]);
plt.tick_params(labelsize=16)
plt.xlim([0.1, max(model._feature_importance)*1.5])
plt.ylim(-1, len(order))
plt.xscale('log')
plt.savefig('feature-imp-fe-filt-w.png')
print('Feature importance saved')

p_test = model.predict_proba(X_test)[:, 1]

sub_df = pd.DataFrame()
sub_df['id'] = id_test
sub_df['target'] = p_test
sub_df.to_csv(sub_path + 'catb-filt-fe-w-1.csv', index=False)
print('Submission file created')
