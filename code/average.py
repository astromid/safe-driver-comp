import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

subs_path = '../subs/'
test_path = '../data/raw-csv/test.csv'
filelist = glob(subs_path + '*.csv')

test_df = pd.read_csv(test_path)
index = test_df['id'].values
avg = np.zeros(len(index))
for file in tqdm(filelist):
    df = pd.read_csv(file)
    avg += df['target'].values
avg /= len(filelist)
avg_sub = pd.DataFrame()
avg_sub['id'] = index
avg_sub['target'] = avg
avg_sub.to_csv(subs_path + 'averaged.csv', index=False)
print('Submission file created')
