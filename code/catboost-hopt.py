import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from hyperopt import hp, fmin, tpe, Trials, space_eval
from sklearn.model_selection import KFold, cross_val_score

data_path = '../data/'
sub_path = '../subs/'

X = np.load(data_path + 'X-filt.npy')
y = np.load(data_path + 'y.npy')
id_test = np.load(data_path + 'id-test.npy')
X_test = np.load(data_path + 'X-test-filt.npy')

space = {}
space['iterations'] = 100 + hp.randint('iterations', 901)
space['depth'] = 3 + hp.randint('depth', 8)
space['learning_rate'] = hp.choice('learning_rate', [0.01, 0.03, 0.1])
space['l2_leaf_reg'] = 1 + hp.randint('l2_leaf_reg', 5)
space['class_weights'] = hp.choice('class_weights', [[1, 1], [0.5, 1.5], [1, 5], [1, 10], [5, 1], [10, 1]])

def hopt_objective(params):
    print('Current params:')
    print(params)
    est = CatBoostClassifier(**params)
    shuffle = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(est, X, y, cv=shuffle, scoring='roc_auc', n_jobs=4, verbose=1)
    return 1-score.mean()

trials = Trials()
best = fmin(hopt_objective, space, algo=tpe.suggest, max_evals=200, trials=trials)

best_params = space_eval(space, best)
np.save('best-params.npy', best_params)
print(best_params)
model = CatBoostClassifier(best_params)
model.fit(X, y, verbose=True)
np.save('../pics/feat-imp.npy', model._feature_importance)
p_test = model.predict_proba(X_test)[:, 1]

sub_df = pd.DataFrame()
sub_df['id'] = id_test
sub_df['target'] = p_test
sub_df.to_csv(sub_path + 'catb-hopt.csv', index=False)
print('Submission file created')
