"""Order-independent paired bootstrap for the seven comparisons of Table IV.
Usage: python paired_bootstrap.py path/to/predictions_test.csv
Each comparison k (0..6) uses its own generator numpy.random.default_rng(42 + k), so results do not depend on execution order.
Accuracy-difference intervals: 10,000 resamples of test images. AUC-difference intervals: 2,000 resamples, positives and negatives resampled separately.
Endpoints: 2.5th and 97.5th percentiles (NumPy default linear interpolation)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
P = pd.read_csv(sys.argv[1]); y = P.y_true_pneumonia.values; n = len(y)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
COMPARISONS = [("E", "A"), ("D", "A"), ("F", "A"), ("A2", "A"), ("B", "A"), ("B2", "B"), ("C", "B")]
rows = []
for k, (c, r) in enumerate(COMPARISONS):
    rng = np.random.default_rng(42 + k)
    cc = (P[f"pred_{c}"].values == y); cr = (P[f"pred_{r}"].values == y)
    idx = rng.integers(0, n, (10000, n)); d = cc[idx].mean(1) - cr[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5]) * 100
    pc, pr = P[f"p_{c}"].values, P[f"p_{r}"].values; da = []
    for _ in range(2000):
        j = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        da.append(roc_auc_score(y[j], pc[j]) - roc_auc_score(y[j], pr[j]))
    alo, ahi = np.percentile(da, [2.5, 97.5])
    rows.append(dict(candidate=c, reference=r, seed=42 + k, acc_diff_pp=round((cc.mean() - cr.mean()) * 100, 2),
                     acc_lo=round(lo, 2), acc_hi=round(hi, 2), auc_diff=round(roc_auc_score(y, pc) - roc_auc_score(y, pr), 4),
                     auc_lo=round(alo, 4), auc_hi=round(ahi, 4)))
out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv("paired_bootstrap_fixed_seeds.csv", index=False)
