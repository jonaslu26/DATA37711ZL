"""Problem 2: target encoding vs. embeddings vs. both, retail Quantity.

Quantity is a heavy-tailed count (median 6, max 80995), so every model is trained
with squared error on z = log(1 + Quantity) and evaluated by test MSE on that scale.
"""
import copy

import category_encoders as ce
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split

from common import SEED, find_file, save_json, seed_everything

seed_everything()
torch.set_num_threads(2)
CATS = ["StockCode", "CustomerID"]

df = pd.read_csv(find_file("retail.csv"), dtype={"StockCode": str, "CustomerID": str})
df[CATS] = df[CATS].astype("category")
df["z"] = np.log1p(df["Quantity"].astype(float))

# ---------- (a) cardinality ----------
card = {c: int(df[c].nunique()) for c in CATS}
counts = {c: df[c].value_counts() for c in CATS}
card_info = {c: {"unique": card[c], "median_rows_per_level": float(counts[c].median()),
                 "levels_with_<5_rows": int((counts[c] < 5).sum())} for c in CATS}
print(card_info)

tr, tmp = train_test_split(df, test_size=0.3, random_state=SEED)
va, te = train_test_split(tmp, test_size=0.5, random_state=SEED)
unseen = {c: float((~te[c].isin(set(tr[c]))).mean()) for c in CATS}
print("splits:", len(tr), len(va), len(te), "unseen in test:", unseen)

results = {"cardinality": card_info, "n": len(df), "splits": [len(tr), len(va), len(te)],
           "unseen_frac_test": unseen,
           "onehot_dense_gb": len(tr) * sum(card.values()) * 4 / 1e9}


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


rows = [{"model": "Constant (train mean)", "test_mse": mse(te.z, tr.z.mean())}]

# ---------- (b) target encoding + gradient boosting ----------
def make_te():
    return ce.TargetEncoder(cols=CATS, smoothing=10, min_samples_leaf=20)


te_enc = make_te()
Z_tr_naive = te_enc.fit_transform(tr[CATS], tr.z).to_numpy(np.float32)  # in-sample: leaks y_i
# Out-of-fold encoding for training rows: each row encoded by an encoder that never saw it.
Z_tr_oof = np.zeros_like(Z_tr_naive)
for a, b in KFold(5, shuffle=True, random_state=SEED).split(tr):
    Z_tr_oof[b] = make_te().fit(tr[CATS].iloc[a], tr.z.iloc[a]).transform(tr[CATS].iloc[b]).to_numpy(np.float32)
Z_va = te_enc.transform(va[CATS]).to_numpy(np.float32)
Z_te = te_enc.transform(te[CATS]).to_numpy(np.float32)
# Standardize the target-encoded features with training (out-of-fold) statistics, so they enter the
# network on the same scale as the embeddings. Gradient boosting is invariant to this rescaling.
mu, sd = Z_tr_oof.mean(0), Z_tr_oof.std(0)
Z_tr_naive, Z_tr_oof, Z_va, Z_te = [((z - mu) / sd).astype(np.float32) for z in (Z_tr_naive, Z_tr_oof, Z_va, Z_te)]

def fit_gb(Z_tr):
    # number of boosting iterations chosen on our validation split
    best_it, best_v = None, np.inf
    for it in [25, 50, 100, 200, 400]:
        m = HistGradientBoostingRegressor(max_iter=it, early_stopping=False, random_state=SEED).fit(Z_tr, tr.z)
        v = mse(va.z, m.predict(Z_va))
        if v < best_v:
            best_it, best_v, best_m = it, v, m
    return best_m, best_it, best_v


gb_models = {}
for tag, Z_tr in [("in-sample", Z_tr_naive), ("out-of-fold", Z_tr_oof)]:
    m, it, v = fit_gb(Z_tr)
    gb_models[tag] = m
    rows.append({"model": f"Target enc. ({tag}) + HistGB", "iters": it, "val_mse": v,
                 "train_mse": mse(tr.z, m.predict(Z_tr)), "test_mse": mse(te.z, m.predict(Z_te))})
    print(rows[-1])
hgb = gb_models["out-of-fold"]

# ---------- (c)/(d) embedding networks ----------
# Index 0 is reserved for levels unseen in training.
vocab = {c: {str(v): i + 1 for i, v in enumerate(tr[c].unique())} for c in CATS}


def codes(d):
    cols = [d[c].astype(str).map(vocab[c]).fillna(0).astype(int).to_numpy() for c in CATS]
    return torch.tensor(np.stack(cols, 1), dtype=torch.long)


C_tr, C_va, C_te = codes(tr), codes(va), codes(te)
T_va, T_te = torch.from_numpy(Z_va), torch.from_numpy(Z_te)
y_tr, y_va = torch.tensor(tr.z.to_numpy(), dtype=torch.float32), torch.tensor(va.z.to_numpy(), dtype=torch.float32)


class EmbNet(nn.Module):
    def __init__(self, dims=(16, 16), use_te=False, hidden=64):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(len(vocab[c]) + 1, d) for c, d in zip(CATS, dims))
        self.use_te = use_te
        d_in = sum(dims) + (len(CATS) if use_te else 0)
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, c, t):
        h = [e(c[:, j]) for j, e in enumerate(self.embs)]
        if self.use_te:
            h.append(t)
        return self.mlp(torch.cat(h, 1)).squeeze(1)


def train(use_te, T_tr=None, p_unk=0.01, epochs=30, bs=1024, lr=2e-3, wd=1e-5, patience=3):
    seed_everything()
    m = EmbNet(use_te=use_te)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            c = C_tr[idx].clone()
            c[torch.rand(c.shape) < p_unk] = 0  # category dropout: trains the "unseen" embedding
            t = T_tr[idx] if use_te else None
            loss = F.mse_loss(m(c, t), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = F.mse_loss(m(C_va, T_va), y_va).item()
        hist.append(v)
        print(f"[emb te={use_te}] epoch {ep} val {v:.4f}")
        if v < best - 1e-4:
            best, state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        pred = m(C_te, T_te).numpy()
    return m, best, pred, len(hist)


m_emb, v_emb, p_emb, ep_emb = train(use_te=False)
rows.append({"model": "Embeddings (16+16) + MLP", "val_mse": v_emb, "test_mse": mse(te.z, p_emb),
             "epochs": ep_emb})
both = {}
for tag, Z_tr in [("in-sample", Z_tr_naive), ("out-of-fold", Z_tr_oof)]:
    _, v, p, ep = train(use_te=True, T_tr=torch.from_numpy(Z_tr))
    both[tag] = p
    rows.append({"model": f"Embeddings + target enc. ({tag}) + MLP", "val_mse": v,
                 "test_mse": mse(te.z, p), "epochs": ep})
p_both = both["out-of-fold"]

# ---------- (e) error by how often the stock code was seen in training ----------
freq = te["StockCode"].map(tr["StockCode"].value_counts()).astype(float).fillna(0).to_numpy()
bins = [(0, 0), (1, 9), (10, 99), (100, np.inf)]
preds = {"TE+GB": hgb.predict(Z_te), "Emb": p_emb, "Both": p_both}  # out-of-fold TE versions
by_freq = []
for lo, hi in bins:
    msk = (freq >= lo) & (freq <= hi)
    by_freq.append({"train_rows": f"{lo}-{hi}", "n": int(msk.sum()),
                    **{k: mse(te.z.to_numpy()[msk], v[msk]) for k, v in preds.items()}})

# Nearest neighbours in StockCode embedding space for a few frequent codes (sanity check).
E = F.normalize(m_emb.embs[0].weight.detach(), dim=1)
inv = {i: v for v, i in vocab["StockCode"].items()}
top = tr["StockCode"].value_counts().index[:3]
nn_examples = {}
for code in top:
    sims = E @ E[vocab["StockCode"][str(code)]]
    nn_examples[str(code)] = [inv[int(i)] for i in sims.argsort(descending=True)[:6] if int(i) in inv and inv[int(i)] != str(code)][:5]

results.update({"rows": rows, "by_train_frequency": by_freq, "nn_examples": nn_examples})
save_json(results, "q2.json")
