"""Problem 3: Random Kitchen Sinks (ridge) vs. 1-hidden-layer MLP on the 2D toy data."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DATA, FIGURES, ROOT, save_json, seed_everything
from dataloader import HW2Dataset
from models import mlp_sweep, mse, sweep_rks_model, MLP, RandomKitchenSinks
from plotting import plot_surface_2d

seed_everything()
ds = HW2Dataset.from_csv(DATA / "train.csv", DATA / "val.csv", DATA / "test.csv")
Xtr, ytr, Xva, yva, Xte = ds.train.X, ds.train.y, ds.val.X, ds.val.y, ds.test.X
print("train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)

# (a) training data
fig, ax = plt.subplots(figsize=(4.6, 3.8))
sc = ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=14)
fig.colorbar(sc, label="y")
ax.set(title="Training data", xlabel="x1", ylabel="x2", xlim=(-3, 3), ylim=(-3, 3))
fig.tight_layout()
fig.savefig(FIGURES / "q3a_train.png", dpi=200)


def sweep_plots(values, tr, va, xlabel, name, logx=True):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, v, t in [(axes[0], tr, "Training MSE"), (axes[1], va, "Validation MSE")]:
        ax.plot(values, v, "o-")
        ax.set(title=t, xlabel=xlabel, ylabel="MSE")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
    best = int(np.argmin(va))
    axes[1].plot(values[best], va[best], "r*", ms=12)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200)


# (b) RKS: sweep kernel scale sigma; D = 1000, lambda = 1e-3 fixed
RKS_FIXED = dict(D=1000, ridge_lambda=1e-3)
sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
rks, rks_val, best_sigma, rks_tr_curve, rks_va_curve = sweep_rks_model(sigmas, Xtr, ytr, Xva, yva, **RKS_FIXED)
sweep_plots(sigmas, rks_tr_curve, rks_va_curve, r"kernel scale $\sigma$", "q3b_rks_sweep.png")
print("RKS best sigma", best_sigma, "val", rks_val)

# (c) MLP: sweep hidden dimension; Adam lr 1e-3, batch 64, 1000 epochs, ReLU fixed
MLP_FIXED = dict(lr=1e-3, batch_size=64, epochs=1000)
hiddens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
mlp, mlp_val, best_h, mlp_tr_curve, mlp_va_curve = mlp_sweep(hiddens, Xtr, ytr, Xva, yva, **MLP_FIXED)
sweep_plots(hiddens, mlp_tr_curve, mlp_va_curve, "hidden dimension", "q3c_mlp_sweep.png")
print("MLP best hidden", best_h, "val", mlp_val)

# (d) refit with the best hyperparameters, report validation MSE, predict test, plot surfaces
rks = RandomKitchenSinks(sigma=best_sigma, **RKS_FIXED).fit(Xtr, ytr)
mlp = MLP(Xtr.shape[1], d_hidden=best_h, **MLP_FIXED).fit(Xtr, ytr)
res_d = {k: {"train_mse": mse(ytr, m.predict(Xtr)), "val_mse": mse(yva, m.predict(Xva))}
         for k, m in [("rks", rks), ("mlp", mlp)]}
np.savetxt(ROOT / "rks_test_predictions.csv", rks.predict(Xte), fmt="%.10f")
np.savetxt(ROOT / "mlp_test_predictions.csv", mlp.predict(Xte), fmt="%.10f")

plot_surface_2d(rks.predict, (-3, 3), (-3, 3), title=rf"RKS ($\sigma$={best_sigma})",
                filename=str(FIGURES / "q3d_rks_surface.png"))
plot_surface_2d(mlp.net, (-3, 3), (-3, 3), title=f"MLP (hidden={best_h})",
                filename=str(FIGURES / "q3d_mlp_surface.png"))

save_json({
    "n": [len(ytr), len(yva), len(Xte)],
    "constant_baseline_val_mse": mse(yva, np.full_like(yva, ytr.mean())),
    "rks": {"fixed": RKS_FIXED, "sigmas": sigmas, "train": rks_tr_curve, "val": rks_va_curve,
            "best_sigma": best_sigma, **res_d["rks"]},
    "mlp": {"fixed": MLP_FIXED, "hidden": hiddens, "train": mlp_tr_curve, "val": mlp_va_curve,
            "best_hidden": best_h, **res_d["mlp"]},
}, "q3.json")
