"""Problem 1: heteroscedastic Gaussian MLP for NYC taxi trip duration (minutes)."""
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from common import SEED, find_file, save_json, seed_everything

seed_everything()
torch.set_num_threads(2)
THRESH = 45.0

df = pd.read_parquet(find_file("taxi_trips.parquet"))
y = df["trip_duration"].to_numpy(np.float32)
X = df.drop(columns="trip_duration").to_numpy(np.float32)  # features are pre-standardized
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)
print("train/val/test:", len(y_tr), len(y_va), len(y_te))


def gauss_nll(y, mu, sigma):
    return float(np.mean(np.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + 0.5 * math.log(2 * math.pi)))


def evaluate(name, mu, sigma):
    mu = np.broadcast_to(mu, y_te.shape).astype(float)
    sigma = np.broadcast_to(sigma, y_te.shape).astype(float)
    p = norm.cdf((THRESH - mu) / sigma)                      # P(T < 45 | x)
    truth = (y_te < THRESH).astype(float)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "model": name,
        "rmse": float(np.sqrt(np.mean((y_te - mu) ** 2))),
        "nll": gauss_nll(y_te, mu, sigma),
        "acc_45": float(np.mean((p > 0.5) == truth)),
        "brier_45": float(np.mean((p - truth) ** 2)),
        "logloss_45": float(-np.mean(truth * np.log(pc) + (1 - truth) * np.log(1 - pc))),
        "mean_p_45": float(p.mean()),
    }
    print(out)
    return out


rows = []
# Baseline 1: constant Gaussian fit to training targets.
rows.append(evaluate("Constant N(mean, var)", y_tr.mean(), y_tr.std()))
# Baseline 2: linear regression + constant sigma.
lr = LinearRegression().fit(X_tr, y_tr)
rows.append(evaluate("Linear reg. + const. sigma", lr.predict(X_te),
                     np.std(y_va - lr.predict(X_va))))
# Baseline 3: strong mean model (gradient boosting) + constant sigma.
hgb = HistGradientBoostingRegressor(max_iter=300, random_state=SEED).fit(X_tr, y_tr)
rows.append(evaluate("Grad. boosting + const. sigma", hgb.predict(X_te),
                     np.std(y_va - hgb.predict(X_va))))


class MLP(nn.Module):
    def __init__(self, p, hidden=128, depth=3):
        super().__init__()
        layers, d = [], p
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        g = self.head(self.body(x))
        mu = g[:, 0]
        sigma = F.softplus(g[:, 1]) + 1e-3        # link from 1(a)
        return mu, sigma


def fit(loss_kind, epochs=40, bs=2048, lr=2e-3, patience=5):
    seed_everything()
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va)

    def loss_fn(mu, sigma, yb):
        if loss_kind == "mse":
            return F.mse_loss(mu, yb)
        return (torch.log(sigma) + 0.5 * ((yb - mu) / sigma) ** 2).mean()  # NLL from 1(b)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), bs):
            idx = perm[i:i + bs]
            mu, sigma = model(Xt[idx])
            loss = loss_fn(mu, sigma, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(*model(Xv), yv).item()
        sched.step(v)
        print(f"[{loss_kind}] epoch {ep:2d} val loss {v:.4f}")
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, ep + 1


def predict(model, Z):
    with torch.no_grad():
        mu, sigma = model(torch.from_numpy(Z))
    return mu.numpy(), sigma.numpy()


# Baseline 4: same MLP trained with MSE (homoscedastic) + constant sigma from validation.
m_mse, ep_mse = fit("mse")
mu_va, _ = predict(m_mse, X_va)
mu_te, _ = predict(m_mse, X_te)
rows.append(evaluate("MLP (MSE) + const. sigma", mu_te, np.std(y_va - mu_va)))

# Main model: heteroscedastic MLP trained with Gaussian NLL.
m_het, ep_het = fit("nll")
mu_te, sig_te = predict(m_het, X_te)
rows.append(evaluate("Heteroscedastic MLP (NLL)", mu_te, sig_te))

# Calibration of P(T<45) on fixed bins (decile bins are degenerate: 99.6% of trips are < 45 min).
p = norm.cdf((THRESH - mu_te) / sig_te)
truth = (y_te < THRESH)
calib = []
for lo, hi in [(0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.0 + 1e-9)]:
    msk = (p >= lo) & (p < hi)
    if msk.any():
        calib.append({"bin": f"[{lo}, {min(hi, 1.0)})", "n": int(msk.sum()),
                      "mean_pred": float(p[msk].mean()), "frac_true": float(truth[msk].mean())})
# The rare class: trips of 45+ minutes.
long_trip = ~truth
pred_long = p < 0.5
rare = {"n_long": int(long_trip.sum()), "recall_long": float(pred_long[long_trip].mean()),
        "precision_long": float(long_trip[pred_long].mean()) if pred_long.any() else None,
        "mean_p_long_given_long": float((1 - p[long_trip]).mean()),
        "mean_p_long_given_short": float((1 - p[truth]).mean())}
print(calib, rare)

save_json({
    "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te),
    "frac_under_45_test": float(truth.mean()),
    "epochs": {"mse": ep_mse, "nll": ep_het},
    "sigma_pred_quantiles": np.quantile(sig_te, [0.05, 0.5, 0.95]).tolist(),
    "rows": rows, "calibration": calib, "rare_class": rare,
}, "q1.json")
