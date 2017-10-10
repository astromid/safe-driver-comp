import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


train_file = '../data/raw/train.csv'
test_file = '../data/raw/test.csv'
sub_path = '../subs/'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

y = train_df['target'].values
id_test = test_df['id'].values

X = train_df.drop(['target', 'id'], axis=1)
model = CatBoostClassifier()
model.fit(X, y, verbose=True)


p_test = model.predict_proba(X_test)[:, 1]

sub_df = pd.DataFrame()
sub_df['id'] = id_test
sub_df['target'] = p_test
sub_df.to_csv(sub_path + 'catb-filt-1.csv', index=False)
print('Submission file created')
