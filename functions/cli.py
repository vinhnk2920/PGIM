"""Command-line entry point; all stages print progress and write run logs."""

import argparse
import gc
import importlib.util
import json
from pathlib import Path

from functions.common import run_log
from functions.config import DEFAULT_SOURCE, PACKAGE_ROOT, Settings


def parser(default_stage=None):
    p = argparse.ArgumentParser(description="PGIM full-district Python pipeline")
    p.add_argument(
        "--stage",
        choices=["preflight", "preprocess", "impute", "features", "forecast", "all"],
        default=default_stage or "preflight",
    )
    p.add_argument("--market", choices=["rental", "sales", "both"], default="both")
    p.add_argument("--mode", choices=["static", "rolling", "both"], default="static")
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "outputs")
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Use original full-district intermediate CSVs where cleaned outputs do not yet exist",
    )
    p.add_argument(
        "--rebuild-raw",
        action="store_true",
        help="Rebuild original enrichment before macros/outliers; requires original 99.co source",
    )
    p.add_argument("--train-start", type=int, default=200001)
    p.add_argument("--test-cutoff", type=int, default=202001)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 12, 24])
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--search-profile", choices=["fast", "full"], default="fast")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fit fixed LightGBM parameters without Optuna; quality weights are preserved",
    )
    p.add_argument("--cv-splits", type=int, default=2)
    p.add_argument("--validation-months", type=int, default=12)
    p.add_argument("--rolling-train-months", type=int, default=180)
    p.add_argument("--rolling-test-months", type=int, default=12)
    p.add_argument(
        "--imputed-target-weight",
        type=float,
        default=0.0,
        help="Main full-district notebooks use 0; selected_w1 variants use 1",
    )
    p.add_argument(
        "--selected-features",
        action="store_true",
        help="Use archived full-district feature-drop lists; see modelling notes",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imputation-max-configs", type=int)
    p.add_argument("--imputation-max-epochs", type=int, default=300)
    p.add_argument("--mc-draws", type=int, default=20)
    return p


def preflight(settings):
    """Inspect files and headers without loading multi-GB inputs or training models."""
    import pandas as pd

    dependencies = {
        name: importlib.util.find_spec(module) is not None
        for name, module in {
            "numpy": "numpy",
            "pandas": "pandas",
            "scikit-learn": "sklearn",
            "torch": "torch",
            "lightgbm": "lightgbm",
            "optuna": "optuna",
            "joblib": "joblib",
            "tqdm": "tqdm",
            "openpyxl": "openpyxl",
            "transformers": "transformers",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
        }.items()
    }
    print("[preflight 1/3] Dependencies:", dependencies)
    checks = []
    prep = settings.prep_path()
    references = [
        settings.transaction_input("rental", "enriched"),
        settings.transaction_input("sales", "enriched"),
        settings.transaction_input("rental", "cleaned"),
        settings.transaction_input("sales", "cleaned"),
        prep / "property" / "property_details_facilities.csv",
        prep / "location" / "condo_latlon_onemap.csv",
        prep / "location" / "full_period_condo_mrt_candidates_walk.csv",
        prep / "location" / "top30_primary_schools.csv",
        prep / "location" / "nearby_5km_amenities.csv",
        prep / "macro" / "monthly" / "sg_macro_monthly.csv",
        prep / "consulting" / "consulting_prediction_manual_data.csv",
        prep / "trend_policy" / "condo_rent_external_features_consolidated.csv",
        prep / "trend_policy" / "condo_sales_external_features_consolidated.csv",
    ]
    references += sorted((prep / "macro" / "quarterly").glob("*.csv"))
    references += [
        prep / "labour" / name
        for name in [
            "ret_01_total_qtr.csv",
            "rcr_01_total_qtr.csv",
            "rgr_01_total_qtr.xlsx",
            "median_gross_monthly_income_full_time_employed_residents_total.csv",
            "share_nonresident_population_by_pass_types_annual.csv",
        ]
    ]
    references += [prep / "supply" / name for name in ["developer_sales.csv", "project_details.csv"]]
    print("[preflight 2/3] Reference files")
    for path in references:
        ok = path.exists()
        checks.append(dict(path=str(path), exists=ok, role="reference"))
        print(f"  {'OK' if ok else 'MISSING'}: {path.name}")
    print("[preflight 3/3] Available full-district panels")
    panels = []
    for market in ("rental", "sales"):
        path = settings.read_model_input(market)
        report = dict(market=market, path=str(path), exists=path.exists())
        if path.exists():
            sample = pd.read_csv(path, nrows=5, low_memory=False)
            price = "rent" if market == "rental" else "sale"
            required = {
                "project_idx",
                "date",
                "condo_name",
                "district",
                "was_observed",
                f"{price}_psf_obs",
                f"{price}_psf_imp",
            }
            report.update(
                columns=len(sample.columns),
                missing_columns=sorted(required - set(sample.columns)),
                bytes=path.stat().st_size,
            )
        panels.append(report)
        print(f"  {market}: {report}")
    missing_raw = prep / "property" / "property_details_facilities.csv"
    ownership_raw = (
        prep
        / "macro"
        / "quarterly"
        / "Assessed Non Landed Private Residential Units By Locals or Foreigners.csv"
    )
    ownership_reference = (
        settings.source_root / "imputation" / "URA_renting_imputed_vae_a4_mcra_full_districts.csv"
    )
    print(f"[reference note] Raw 99.co sales source present: {missing_raw.exists()}")
    print(
        f"[reference note] Raw ownership macro present: {ownership_raw.exists()}; existing derived reference present: {ownership_reference.exists()}"
    )
    report = dict(
        source_root=str(settings.source_root),
        output_root=str(settings.output_root),
        dependencies=dependencies,
        references=checks,
        panels=panels,
        raw_rebuild_missing_reference=str(missing_raw) if not missing_raw.exists() else None,
        ownership_macro_raw_present=ownership_raw.exists(),
        ownership_macro_derived_reference=str(ownership_reference),
        note="Header checks do not establish full-data numerical equivalence or complete data quality.",
    )
    target = settings.folder("logs") / "preflight.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Preflight report: {target}")
    return report


def main(default_stage=None):
    args = vars(parser(default_stage).parse_args())
    stage, market, mode, rebuild = (args.pop(k) for k in ("stage", "market", "mode", "rebuild_raw"))
    args["horizons"] = tuple(args["horizons"])
    settings = Settings(**args)
    markets = ("rental", "sales") if market == "both" else (market,)
    if stage in ("features", "all") and market != "both":
        raise ValueError(
            "The post-imputation feature pipeline jointly builds rental and sales; use --market both"
        )
    stages = ["preprocess", "impute", "features", "forecast"] if stage == "all" else [stage]
    for current in stages:
        with run_log(settings, current):
            if current == "preflight":
                preflight(settings)
            elif current == "preprocess":
                from functions.preprocessing.pipeline import run

                run(settings, markets, rebuild_raw=rebuild)
            elif current == "impute":
                from functions.imputation import rental, sales

                # Rental must run first to establish the bucket reference used by sales.
                for name in markets:
                    (rental if name == "rental" else sales).run(settings)
                    gc.collect()
            elif current == "features":
                from functions.features import (
                    supply,
                    labour,
                    consulting,
                    trend_policy,
                    nearby_prices,
                )

                for index, module in enumerate(
                    [supply, labour, consulting, trend_policy, nearby_prices], 1
                ):
                    print(f"[feature stage {index}/5] {module.__name__.split('.')[-1]}")
                    module.run(settings)
                    gc.collect()
            elif current == "forecast":
                from functions.forecasting.pipeline import run

                for forecast_mode in ("static", "rolling") if mode == "both" else (mode,):
                    run(settings, markets, forecast_mode)


if __name__ == "__main__":
    main()
