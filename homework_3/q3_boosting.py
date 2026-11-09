"""Problem 3: generalization of gradient boosting on dset0 and dset1.

The first column of each CSV is the binary target; the other five are features.
Uses 5-fold stratified CV repeated 4 times. For each fold, one 1000-tree model is
fit and its staged predictions give the results for 1, 10, 100, 300, 1000 trees.
A logistic-regression baseline is included for comparison.

Run:  python q3_boosting.py      (about 1 minute)
Output: outputs/q3_results.csv, outputs/q3.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold

HERE = Path(__file__).resolve().parent
TREE_COUNTS = [1, 10, 100, 300, 1000]
T_MAX = 1000

rows = []
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, name in zip(axes, ["dset0", "dset1"]):
    data = np.loadtxt(HERE / "data" / f"{name}.csv", delimiter=",")
    y, X = data[:, 0].astype(int), data[:, 1:]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=1)
    tr_acc, te_acc, te_ll, lr_acc = [], [], [], []
    for i, (tr, te) in enumerate(cv.split(X, y)):
        gb = GradientBoostingClassifier(n_estimators=T_MAX, random_state=i).fit(X[tr], y[tr])
        tr_acc.append([np.mean(p == y[tr]) for p in gb.staged_predict(X[tr])])
        te_acc.append([np.mean(p == y[te]) for p in gb.staged_predict(X[te])])
        te_ll.append([log_loss(y[te], np.clip(p, 1e-15, 1)) for p in gb.staged_predict_proba(X[te])])
        lr_acc.append(LogisticRegression().fit(X[tr], y[tr]).score(X[te], y[te]))
    tr_acc, te_acc, te_ll = map(np.array, (tr_acc, te_acc, te_ll))
    for T in TREE_COUNTS:
        rows.append(dict(dataset=name, trees=T, train_acc=tr_acc[:, T - 1].mean(),
                         test_acc=te_acc[:, T - 1].mean(), test_acc_sd=te_acc[:, T - 1].std(),
                         test_logloss=te_ll[:, T - 1].mean()))
    rows.append(dict(dataset=name, trees="logistic_regression", test_acc=np.mean(lr_acc)))

    t = np.arange(1, T_MAX + 1)
    ax.semilogx(t, tr_acc.mean(0), label="train acc")
    ax.semilogx(t, te_acc.mean(0), label="test acc")
    ax.axhline(np.mean(lr_acc), ls="--", c="gray", label="logistic reg. (test)")
    ax.set(title=name, xlabel="number of trees", ylim=(0.5, 1.01))
    ax.legend(loc="lower right", fontsize=8)

out = HERE / "outputs"
out.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(out / "q3.png", dpi=150)
table = pd.DataFrame(rows)
table.to_csv(out / "q3_results.csv", index=False)
print(table.to_string(index=False, float_format="{:.3f}".format))
