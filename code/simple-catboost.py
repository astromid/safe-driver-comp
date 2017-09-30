import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

sns.set()

train_path = '../data/train.csv'
test_path = '../data/test.csv'
sub_path = '../subs/'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

y = train_df['target'].values
id_test = test_df['id'].values

X = train_df.drop(['target', 'id'], axis=1)
X_test = test_df.drop(['id'], axis=1)
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
model = CatBoostClassifier().fit(X, y, verbose=True)

order = np.argsort(model._feature_importance)
plt.figure(figsize=[10, 15])
plt.plot(np.array(model._feature_importance)[order], range(len(order)), marker='o')
plt.hlines(range(len(order)), np.zeros_like(order), np.array(model._feature_importance)[order], linestyles=':')
plt.yticks(range(X.shape[1]), X.columns[order]);
plt.tick_params(labelsize=16)
plt.xlim([0.1, max(model._feature_importance)*1.5])
plt.ylim(-1, len(order))
plt.xscale('log')
plt.savefig('feature_importance.png')

p_test = model.predict_proba(X_test)[:, 1]

sub_df = pd.DataFrame()
sub_df['id'] = id_test
sub_df['target'] = p_test
sub_df.to_csv(sub_path + 'catb0.csv', index=False)
