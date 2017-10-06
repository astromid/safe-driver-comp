import numpy as np
import pandas as pd
from glob import glob

subs_path = '../subs/'
test_path = '../data/test.csv'
filelist = ['catb0.csv', 'catb1.csv', 'catb-filt-1.csv']

test_df = pd.read_csv(test_path)
index = test_df['id'].values
avg = np.zeros(len(index))
for file in filelist:
    df = pd.read_csv(subs_path + file)
    avg += df['target'].values
avg /= len(filelist)
avg_sub = pd.DataFrame()
avg_sub['id'] = index
avg_sub['target'] = avg
avg_sub.to_csv(subs_path + 'avg-catb.csv', index=False)
print('Submission file created')
