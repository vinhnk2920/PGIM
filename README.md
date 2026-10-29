# PGIM prediction code

Python workflow for full-district Singapore private residential rental and sales forecasting. The pipeline prepares transactions, imputes missing monthly prices with a VAE, builds explanatory features, and fits LightGBM forecasts at 1, 12 and 24 months.

This repository contains Python source, tests, runtime configuration and this guide. Input data, generated outputs, trained models and local validation reports are not included. Obtain the required data separately before running the pipeline.

## Environment setup

Use Python 3.10 or newer. From Anaconda Prompt, create an environment and install the dependencies:

```bash
conda create -n PGIM_torch python=3.11 -y
conda activate PGIM_torch
cd /path/to/PGIM
python -m pip install -r requirements.txt
```

If `PGIM_torch` already exists, activate it instead of creating it again. The requirements are shared by all pipeline stages. To use an NVIDIA GPU for VAE training, install a CUDA-compatible PyTorch build appropriate for your machine; the code also supports CPU execution.

In Git Bash on Windows, initialize Conda once per terminal session if `conda` is not found:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PGIM_torch
cd /path/to/PGIM
```

If your Anaconda environment encounters the duplicate Intel OpenMP runtime error, set the following before running Python. It is a session-only compatibility setting, not a required pipeline argument.

```bash
# Git Bash / Linux shell
export MKL_THREADING_LAYER=SEQUENTIAL
```

```powershell
# PowerShell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
```

## Code structure

```text
PGIM/
  01_preprocess.py
  02_impute.py
  03_build_features.py
  04_forecast.py
  run_pipeline.py
  consolidate_supply.py
  functions/
    preprocessing/
    imputation/
    features/
    forecasting/
    config.py
    data_paths.py
    common.py
    cli.py
  config/selected_features/
  tests/
  requirements.txt
  pyproject.toml
  README.md
```

The numbered files are command-line entry points. Their implementations live in `functions/`, so different stages reuse the same helpers. All executable source is `.py`; notebooks are not required.

## Prepare the local data

By default, place the separately supplied data under `data/` in the repository root:

```text
data/
  data_prep/
    transactions/
      rental/{source,enriched,cleaned}/
      sales/{source,enriched,cleaned}/
    property/
    location/
    macro/{monthly,quarterly}/
    supply/
      developer_sales.csv
      project_details.csv
    labour/
    consulting/
    trend_policy/
  imputation/             # Optional existing imputation snapshots
  model_input/            # Optional existing forecasting inputs
```

Step 1 normally starts from these enriched transaction snapshots:

| Market | Input relative to `data/` |
|---|---|
| Rental | `data_prep/transactions/rental/enriched/URA_merged_full_v3_new.csv` |
| Sales | `data_prep/transactions/sales/enriched/URA_sale_data.csv` |

Enriched transactions already contain property and location attributes; they precede price imputation. Step 1 adds lagged quarterly macros and applies transaction filters and outlier rules. It does not rebuild enrichment unless `--rebuild-raw` is supplied.

The quarterly reference filenames and raw-enrichment reference paths are defined in `functions/data_paths.py`. Feature construction also requires:

- `data_prep/supply/developer_sales.csv` and `project_details.csv`.
- `data_prep/labour/ret_01_total_qtr.csv`, `rcr_01_total_qtr.csv`, `rgr_01_total_qtr.xlsx`, `median_gross_monthly_income_full_time_employed_residents_total.csv` and `share_nonresident_population_by_pass_types_annual.csv`.
- `data_prep/consulting/consulting_prediction_manual_data.csv`.
- `data_prep/trend_policy/condo_rent_external_features_consolidated.csv` and `condo_sales_external_features_consolidated.csv`.

To use a data directory outside the repository, add `--source-root "/absolute/path/to/data"` to each command. It must point to the directory containing `data_prep/`, `imputation/` and `model_input/`, rather than to `data_prep/` itself. Outputs still go to the repository's `outputs/` unless `--output-root` is also supplied.

For example, a code-only clone can use the data and intermediate results from an existing local installation:

```bash
python 04_forecast.py --mode static --source-root "/absolute/path/to/PGIM_prediction_code/data" --output-root "/absolute/path/to/PGIM_prediction_code/outputs"
```

Run the lightweight inspection before training:

```bash
python run_pipeline.py --stage preflight --reuse-existing
```

Preflight reports dependencies, reference-file presence and panel headers. It does not train models, and a missing optional checkpoint can be expected before the relevant stage has run. Inspect the report for the stage you intend to run; it is not a full data-quality certification.

## Run the pipeline

Run commands from the repository root with the environment activated.

### 1. Preprocess transactions

```bash
python 01_preprocess.py
```

Processes both markets and writes cleaned transactions to `outputs/01_preprocessing/`. To process one market, add `--market rental` or `--market sales`.

Raw enrichment can be rebuilt explicitly with `python 01_preprocess.py --rebuild-raw`. This requires the source transaction snapshots and all property, location and monthly-macro references. Some source snapshots already include upstream enrichment; this is not an API downloader for the original providers.

### 2. Impute missing monthly prices

```bash
python 02_impute.py
```

Builds project/size/month panels, tunes VAE models, evaluates masked observations, calibrates intervals and exports completed panels. Rental runs first because it creates the size-bucket ranges used by sales.

To rerun a single market:

```bash
python 02_impute.py --market rental
python 02_impute.py --market sales
```

Sales requires an existing `rental_size_bucket_ranges_full_districts.csv` in `outputs/02_imputation/`. Full tuning uses 162 configurations and up to 300 epochs per configuration, so this stage can take substantial time.

Each market writes its main CSV in `outputs/02_imputation/` and four artifacts under `rental_artifacts/` or `sales_artifacts/`: a `.pkl` bundle, a `.npz` array archive, `test_metrics.json` and `model_weights.pkl`. These include evaluation metrics and separate evaluation/production model weights and scalers. Successful reruns replace the stable imputation filenames.

### 3. Build forecasting features

```bash
python 03_build_features.py
```

Processes both markets together, adding supply, labour, consulting, trend/policy and nearby observed-price features. It reads the completed imputation panels without retraining the VAE. The consulting stage may download `ProsusAI/finbert` on its first use if it is not cached.

The final files are written to `outputs/03_features/`:

- `URA_renting_imputed_vae_trend_policies_nearby_prices_full_districts.csv`
- `URA_sales_imputed_vae_trend_policies_nearby_prices_full_districts.csv`

For sales, the optional source field `house_age_2026` fills missing `Condo_Age_2026` before deriving `condo_age_at_transaction`. Existing canonical ages are preserved; `house_age_2026` is not exported.

### 4. Run forecasting

Start with static forecasting:

```bash
python 04_forecast.py --mode static
```

Static forecasting fits one model per market, horizon and arm. Defaults are training origins beginning in January 2000 and test origins beginning in January 2020. A training row is eligible only when its future target month is strictly before the test cutoff.

For rolling forecasting:

```bash
python 04_forecast.py --mode rolling --rolling-train-months 180 --rolling-test-months 12
```

Rolling forecasting retunes and refits for each test block using a 180-month calendar training pool and 12-month test blocks by default. The eligible labelled training period is shorter than that pool because future training targets must precede the test block.

To run both modes:

```bash
python 04_forecast.py --mode both
```

Each mode runs rental and sales at horizons 1, 12 and 24 months, comparing `imputed` and `observed` arms. To limit a run:

```bash
python 04_forecast.py --mode static --market rental
python 04_forecast.py --mode static --market sales --horizons 1
```

Default tuning uses 50 Optuna trials, the `fast` search profile and two 12-month validation folds. The `fast` search profile still tunes the model; the separate `--fast` flag skips Optuna and uses fixed parameters for an execution check.

Results are saved in timestamped folders under `outputs/04_forecasts/`, with separate folders for each market and mode. They include:

- `metrics.csv`: RMSE, MAE, MAPE, R2 and evaluation-row counts for both arms.
- `predictions_h<horizon>_w<window>.csv`: predictions and residuals.
- Saved models, feature importance and Optuna trial results.
- `settings.json` and `window_schedule.csv`.

Both arms are scored on the same rows with observed origin and future target prices. RMSE and MAE are in SGD per square foot; rental prices are monthly rent per square foot. Recent origins with unknown future targets can appear in prediction exports but are not scored. Compare `model = imputed` rows with the original Full-Singapore static imputed-model table; rerunning the refactored code does not guarantee identical historical model metrics.

### Run all stages in order

```bash
python run_pipeline.py --stage all --mode both
```

This includes preprocessing and full VAE training. If stages 1-3 are already complete, run `04_forecast.py` directly instead.

## Reuse existing intermediate files

Newly generated files in `outputs/` take priority. With `--reuse-existing`, a missing intermediate can fall back to the separately supplied local snapshots:

| Stage | Fallback location relative to `data/` |
|---|---|
| Imputation | `data_prep/transactions/<market>/cleaned/` |
| Feature construction | `imputation/` |
| Forecasting | `model_input/` |

The final model-input fallback files use the same two filenames listed under step 3. The canonical imputation filenames are `URA_renting_imputed_vae_a4_mcra_full_districts.csv` and `URA_sales_imputed_vae_a4_mcra_full_districts.csv`. Sales can also read one unambiguous legacy `imputation/results_sales_full_districts_*/` snapshot. Sales imputation still needs the rental size-bucket reference under `outputs/02_imputation/`.

For example, to forecast from supplied model-input CSVs without rebuilding earlier stages:

```bash
python 04_forecast.py --mode static --reuse-existing
```

For all command-line options:

```bash
python run_pipeline.py --help
```

## Optional supply-table maintenance

Normal runs use the two consolidated supply CSVs. If you need to rebuild them from the original downloads, specify their locations explicitly:

```bash
python consolidate_supply.py --developer-sales-dir "/path/to/developer_sales_downloads" --project-details-dir "/path/to/project_details_downloads"
```

The command writes the consolidated tables under `data/data_prep/supply/` by default and records checksums under `validation/supply/`. It preserves source order and provenance; original download files are not copied into this repository.

## Modelling assumptions and interpretation

- Coverage is districts 1-28 except district 24: 27 districts with available observations.
- Rental models predict price levels. Sales models predict log growth, reconstruct price levels and tune against observed price-level RMSE.
- The default imputed-target training weight is zero; observed targets have weight one. Completed input histories can still affect model features and training-row eligibility.
- The comparison retains the original arm-specific feature choices: rental h=12/h=24 use observed lag families, rental h=24 restricts the observed arm's macros, and sales adds price-context features to the imputed arm. It is therefore not a controlled experiment that changes only one imputation switch.
- All features are used by default. `--selected-features` applies archived drop lists from `config/selected_features/`; these are not reselected inside each rolling window. `--imputed-target-weight 1` changes the target-weight setting used by the historical `selected_w1` experiments.
- The inherited VAE/scaler and outlier rules use the full historical panel. Rolling forecasting refits LightGBM, without refitting VAE imputation, static metadata or feature selection at every historical origin. Interpret these results as conditional on full-panel preprocessing rather than as a fully point-in-time backtest.

## Tests and local files

The tests use synthetic fixtures and do not require the private datasets:

```bash
python -m pip install pytest ruff
python -m pytest -q
python -m ruff check .
```

Every pipeline stage prints progress and saves console logs and completion/failure records in `outputs/logs/`. The repository's `.gitignore` excludes data, outputs, validation reports, caches, notebooks, environments and model artifacts from version control.
