import numpy as np

def gini(actual, pred, cmpcol=0, sortcol=1):
    assert(len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_norm(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_norm(labels, preds)
    return [('gini', gini_score)]

class CatBoostGini(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        gini_sum = 0.0
        weight_sum = 0.0

        for i in xrange(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            gini_sum += w * gini_norm(target[i], approx[i])
        return gini_sum, weight_sum
