"""
Random Kitchen Sinks (RKS) with ridge regression, and a 1-hidden-layer MLP (Problem 3).
Completed from the provided skeleton.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.ravel(a) - np.ravel(b)) ** 2))


"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """Random Kitchen Sinks featurization with ridge regression."""

    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3, seed: int = 0):
        """
        D: total number of random features (= 2h; h sin/cos pairs)
        sigma: kernel scale; w_j ~ N(0, I / sigma^2)
        ridge_lambda: ridge regularization strength (intercept not penalized)
        """
        assert D % 2 == 0, "D = 2h must be even"
        self.D, self.sigma, self.ridge_lambda, self.seed = D, sigma, ridge_lambda, seed
        self.W = None  # (d, h)
        self.b = None  # (h,)
        self.coef_ = None

    def _init_weights(self, d: int):
        rng = np.random.default_rng(self.seed)
        h = self.D // 2
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(d, h))
        self.b = rng.uniform(0.0, 2 * np.pi, size=h)

    def _features(self, X: np.ndarray) -> np.ndarray:
        """Return [sin(XW+b), cos(XW+b), 1] of shape (n, D + 1)."""
        proj = X @ self.W + self.b
        return np.hstack([np.sin(proj), np.cos(proj), np.ones((X.shape[0], 1))])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._init_weights(X.shape[1])
        phi = self._features(X)
        reg = self.ridge_lambda * np.eye(phi.shape[1])
        reg[-1, -1] = 0.0  # do not penalize the intercept
        # ridge: minimize (1/n)||y - phi c||^2 + lambda ||c||^2
        n = phi.shape[0]
        self.coef_ = np.linalg.solve(phi.T @ phi / n + reg, phi.T @ y / n)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._features(X) @ self.coef_


def sweep_rks_model(sweep_sigma: List[float], X_train, y_train, X_val, y_val, **kwargs) -> Tuple:
    """Sweep the kernel scale sigma; return (best_model, best_val_mse, best_sigma, train_mses, val_mses)."""
    best_model, best_val_mse, best_sigma = None, float("inf"), None
    train_mses, val_mses = [], []
    for s in sweep_sigma:
        model = RandomKitchenSinks(sigma=s, **kwargs).fit(X_train, y_train)
        tr, va = mse(y_train, model.predict(X_train)), mse(y_val, model.predict(X_val))
        train_mses.append(tr)
        val_mses.append(va)
        if va < best_val_mse:
            best_model, best_val_mse, best_sigma = model, va, s
    return best_model, best_val_mse, best_sigma, train_mses, val_mses


"""
For 1-hidden-layer MLP (MLP)
"""
class MLP(nn.Module):
    """1-hidden-layer MLP for regression, trained with Adam on MSE."""

    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3,
                 batch_size: int = 64, epochs: int = 1000, activation: str = "relu",
                 device: str = "cpu", seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.device, self.lr, self.batch_size, self.epochs, self.seed = device, lr, batch_size, epochs, seed
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), act(), nn.Linear(d_hidden, d_out)).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                loss = loss_fn(self.net(X[idx]), y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.net.eval()
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy().ravel()


def mlp_sweep(sweep_hidden: List[int], X_train, y_train, X_val, y_val, **kwargs) -> Tuple:
    """Sweep the hidden dimension; return (best_model, best_val_mse, best_hidden, train_mses, val_mses)."""
    best_model, best_val_mse, best_h = None, float("inf"), None
    train_mses, val_mses = [], []
    for h in sweep_hidden:
        model = MLP(X_train.shape[1], d_hidden=h, **kwargs).fit(X_train, y_train)
        tr, va = mse(y_train, model.predict(X_train)), mse(y_val, model.predict(X_val))
        train_mses.append(tr)
        val_mses.append(va)
        print(f"  hidden={h}: train {tr:.4f} val {va:.4f}", flush=True)
        if va < best_val_mse:
            best_model, best_val_mse, best_h = model, va, h
    return best_model, best_val_mse, best_h, train_mses, val_mses
