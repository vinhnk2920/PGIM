# Reported-method experiment from raw inputs

The validation environment is recorded in `validated_environment.json`. Use the same environment for the closest numerical comparison; switching PyTorch/CUDA or other library versions can change the VAE fit even with the same seed.

Run from the project folder with the complete local data installed:

```powershell
conda activate PGIM_torch
python 05_reproduce_from_raw.py
```

Each invocation creates a separate `outputs/from_raw_<timestamp>` folder. It rebuilds rental and sales transactions, merges macro and property references, filters outliers, trains evaluation and production VAEs, builds features, and fits the observed-only and imputed LightGBM arms at 1, 12 and 24 months. It does not load old VAE weights, filled price panels, engineered model inputs or forecast predictions. Runtime input auditing rejects reads of the archived imputation/model-input/output folders.

The JSON contains scalar hyperparameters and feature names from the reported experiment. The VAE grid search and LightGBM tuning are not repeated: both models are fitted afresh with the recorded settings. Repeating an unrestricted hyperparameter search is a different experiment and can select a different model. Random VAE initialization and numerical differences can still change the filled prices and downstream metrics. A fixed seed is not a guarantee of bitwise identity across environments.

The full input set includes the original transaction CSVs, quarterly/monthly macro files, supply, labour, consulting and policy files, school/MRT/amenity lookups, property details and the historical property/project enrichment references listed in `audit/fresh_input_reads.json`. In particular, the rental property/project reference files are used for attributes, not for imputed prices. They are still required source data; this command does not re-scrape historical external services.

`audit/*_observed_panel.json` checks the series/month/observed-cell counts and the exact price, mask and series fingerprints before training. `audit/*_run_manifest.json` records the configuration, Python/library versions and source-code hashes for each stage. Imputation metrics are scored using masked evaluation inference, separately from production imputation.

## Historical methodology kept only for replication

The explicit profile restores rental rank-first quantile size groups and sales cuts derived from that fresh rental run, the historical missing-month handling, and the historical supply/macro feature transformations. These differ from the corrected default pipeline. Rank-first quantiles depend on transaction order. The historical supply rule fills early lagged sold-share values with 1 for projects that eventually sell out, which uses later project information. The historical loader also aggregates supply variables by date. These rules are preserved to explain and reproduce the reported numbers, not to establish point-in-time forecast validity. The inherited VAE scaler uses the complete observed panel; validation/calibration/test values are masked from network fitting. A separate point-in-time experiment is needed before interpreting these benchmarks as live forecasting performance.

The previous `--reference-profile` command remains a separate forecast-only check using archived engineered inputs. It is not used by `05_reproduce_from_raw.py`.

The production VAE is fitted on the complete observed timeline before forecasting; it is not refitted using only information available at each forecast origin. Thus the forecasting holdout is not an independent test of a fully point-in-time imputation-and-forecasting pipeline. This command reproduces that historical design. Its scores must not be described as a clean live forecasting backtest.

The recorded rental observed-only h=1 and h=12 parameters come from the surviving notebook output; a complete archived baseline model was unavailable. Their provenance and any discrepancy from the reported baseline must be shown in the run comparison rather than presented as exact replication.
