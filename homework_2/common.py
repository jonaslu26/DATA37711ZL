"""Shared helpers: data paths, loading, seeding, result saving."""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 0


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def find_file(*patterns: str) -> Path:
    """Return the first file in data/ matching any glob pattern."""
    for pat in patterns:
        hits = sorted(DATA.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No file matching {patterns} in {DATA}")


def read_csv(path: Path) -> pd.DataFrame:
    """CSV reader that sniffs the delimiter (the UCI wine file uses ';')."""
    return pd.read_csv(path, sep=None, engine="python")


def save_json(obj, name: str) -> None:
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(json.dumps(obj, indent=2, default=float))
