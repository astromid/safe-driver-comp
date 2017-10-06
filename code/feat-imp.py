import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

feat_imp =  np.load('../data/feat-imp.npy')
feat_names = np.load('../data/feat_names.npy')

order = np.argsort(feat_imp)
plt.figure(figsize=[10, 15])
plt.plot(np.array(feat_imp)[order], range(len(order)), marker='o')
plt.hlines(range(len(order)), np.zeros_like(order), np.array(feat_imp)[order], linestyles=':')
plt.yticks(range(X.shape[1]), feat_names[order]);
plt.tick_params(labelsize=16)
plt.xlim([0.1, max(feat_imp)*1.5])
plt.ylim(-1, len(order))
plt.xscale('log')
plt.savefig('../pics/feature-imp-fe-filt-w.png')
print('Feature importance saved')
