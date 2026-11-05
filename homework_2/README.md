# Homework 2

Python 3.10+, CPU only. All data needed is in `data/`.

## Run

```bash
cd homework_2
bash run_all.sh
```

`run_all.sh` installs `requirements.txt` and runs the three problems in order. Each script can also be run on its own:

| Script | Problem |
|---|---|
| `python q1_taxi.py` | 1: heteroscedastic Gaussian MLP on taxi trips, P(T < 45 min) |
| `python q2_retail.py` | 2: target encoding vs. embeddings vs. both on retail quantities |
| `python q3_rks_mlp.py` | 3: Random Kitchen Sinks (ridge) vs. 1-hidden-layer MLP |

A full run takes about 15 minutes on a CPU. All randomness is seeded (seed 0); a fresh clone reproduces `results/` and both prediction files exactly.

## Outputs

- `results/q1.json`, `results/q2.json`, `results/q3.json`: all numbers in the write-up
- `figures/`: Problem 3 figures
- `rks_test_predictions.csv`, `mlp_test_predictions.csv`: Problem 3(d) predictions on `data/test.csv`, one per row in the same order
- `report/hw2.pdf`: write-up (`report/hw2.tex` is its source)

## Files

- `common.py`: paths, seeding, JSON output
- `models.py`: RKS and MLP implementations (completed from the provided skeleton)
- `dataloader.py`, `plotting.py`: provided helpers, unchanged
