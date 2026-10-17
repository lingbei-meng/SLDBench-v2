# Reference law and acquisition baselines

## Run the baselines

Use Python 3.12. From the repository root:

```bash
python -m pip install -r requirements-baselines.txt
python -m baselines.acquisition --workers 8
```

No model API, GPU or Docker is required. The command evaluates the 30 selected tasks. Each numerical worker uses one thread; adjust `--workers` to the available CPU cores.

```bash
# Continue an interrupted run with the same configuration.
python -m baselines.acquisition --workers 8 --resume

# Fit the reference using all observations.
python -m baselines.acquisition --full-pool-only --workers 8 --out runs/full_pool

# Run a single task.
python -m baselines.acquisition --full-pool-only --workers 1 --tasks colpret_ibm_model_token_openbookqa_accuracy_scaling_costaware --out runs/reference_smoke
```

## Reference formula

The task-specific fitters are in [reference_laws/](../baselines/reference_laws/). For each group, the nonlinear reference uses:

```text
prediction = E + sum_j A_j * exp(-alpha_j * log1p(abs(x_j)))
```

The implementation uses signed amplitudes, exponents between −0.5 and 3, and six seeded starts of bounded least squares with `soft_l1` loss. Each fitter specifies its log-linear and pooled fallbacks, prediction bounds, row subsampling and 150-second fitting limit. Scores use the benchmark's pooled held-out R² metric.

## Acquisition policies

[acquisition.py](../baselines/acquisition.py) implements four rules:

- **Cheapest-first:** increasing cost, with observation ID breaking ties.
- **Random:** observation IDs shuffled with NumPy seeds 0, 1 and 2.
- **Cheapest per group:** round-robin over groups, taking each group's next cheapest point.
- **Price-spread:** round-robin over groups, visiting low, high, middle and progressively finer price ranks.

All rules receive the same task-specific budget cap: the budget fraction multiplied by the total catalogue cost. A candidate is acquired if its price fits the remaining budget; otherwise the scan continues. Selection uses candidate IDs, groups and prices. Fitting uses acquired outcomes, and held-out outcomes are used only for scoring.

The budget fractions are `0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00`. Random acquisition takes the median across three seeds within each task, followed by the median across tasks. At full budget every rule uses the same full-pool fit. A complete run contains 1,290 fits.

## Data and outputs

Tasks are listed in [metadata/evaluation_tasks.json](../metadata/evaluation_tasks.json). Each task uses its catalogue and measurements under `active/<task>/` and its held-out split at `datasets/<task>/test.parquet`.

Outputs are written to `runs/reference_baselines/`, or a directory specified with `--out`:

- `fits.csv`: per-fit scores, acquisition costs and point counts.
- `summary.json`: aggregate scores and policy comparisons.
- `run_manifest.json`: input checksums, dependency versions and run configuration.
- `scratch/fits_checkpoint.jsonl`: checkpoints used by `--resume`.

Generated run files are excluded from version control.
