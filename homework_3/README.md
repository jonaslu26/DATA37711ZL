# Homework 3 — DATA 37711

**Author:** Zhengxin Lu

The write-up is `hw3.pdf`, and its LaTeX source is in `report/`.

## Contents
| Path | Description |
|---|---|
| `hw3.pdf` | Solutions to Problems 1–3 |
| `report/hw3.tex`, `report/q3.png` | LaTeX source and figure |
| `q2_wine_grid_search.py` | Problem 2(c): 70/10/20 split, grid search (600 configs each for RF and KNN), validation selection, test evaluation |
| `q3_boosting.py` | Problem 3: gradient boosting with 1, 10, 100, 300, 1000 trees on dset0 and dset1 |
| `data/` | `dset0.csv`, `dset1.csv`, `winequality-white.csv` (the provided files, semicolon-separated for the wine data) |
| `outputs/` | Results written by the scripts |

## How to run
From this folder:
```bash
pip install -r requirements.txt
python q3_boosting.py           # ~1 min  -> outputs/q3_results.csv, outputs/q3.png
python q2_wine_grid_search.py   # ~6 min  -> outputs/q2_results.json
```
Both scripts use fixed random seeds and paths relative to the script, so they can be run from any working directory.
