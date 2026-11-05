# Homework 2

| File | Problem |
|---|---|
| `q1_taxi.py` | 1 heteroscedastic Gaussian MLP (PyTorch) on taxi trips, P(T < 45 min) |
| `q2_retail.py` | 2 target encoding vs. embeddings vs. both on retail quantities |
| `q3_rks_mlp.py`, `models.py` | 3 Random Kitchen Sinks (ridge) vs. 1-hidden-layer MLP |
| `rks_test_predictions.csv`, `mlp_test_predictions.csv` | 3(d) predictions on `test.csv`, one per row, same order |
| `dataloader.py`, `plotting.py` | provided helpers |

Put the provided data in `data/` (`taxi_trips.parquet`, `retail.csv`, `train.csv`, `val.csv`, `test.csv`),
then run `bash run_all.sh`. Results go to `results/*.json`, figures to `figures/`, write-up to `report/hw2.pdf`.
All randomness is seeded (seed 0).
