import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

dir_name = 'fe5-npy'

feat_imp =  np.load('../pics/feat-imp-npy/' + dir_name + '-catb.npy')
feat_names = np.load('../data/' + dir_name + '/feat_names.npy')
X_shape1 = np.load('../data/' + dir_name +  '/X-shape1.npy')

order = np.argsort(feat_imp)
plt.figure(figsize=[10, 15])
plt.plot(np.array(feat_imp)[order], range(len(order)), marker='o')
plt.hlines(range(len(order)), np.zeros_like(order), np.array(feat_imp)[order], linestyles=':')
plt.yticks(range(X_shape1), feat_names[order]);
plt.tick_params(labelsize=14)
plt.xlim([0.1, max(feat_imp)*1.5])
plt.ylim(-1, len(order))
plt.xscale('log')
plt.savefig('../pics/' + dir_name + '-catb.png')
print('Feature importance saved')
