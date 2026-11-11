"""Numerical reproduction of a reported run, with a separate fresh-model refit.

All inputs are read-only. Parameters are taken from the historical model, not
selected against the test set. Refit and archived-model inference are labelled
separately, and a tolerance failure raises after writing the comparison report.
"""

import gc
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from functions.common import add_months

KEYS = ["condo_name", "size_tier", "period"]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def score(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Scoring requires finite observed targets and predictions")
    positive = np.abs(y) > 1e-9
    return dict(n=len(y), RMSE=float(np.sqrt(mean_squared_error(y, prediction))),
                MAE=float(mean_absolute_error(y, prediction)),
                MAPE_pct=float(np.mean(abs((y[positive] - prediction[positive]) / y[positive])) * 100),
                R2=float(r2_score(y, prediction)))


def target_arrays(panel, horizon, target_mode):
    """Exact calendar labels, with original row order retained for fitting."""
    if panel.duplicated(KEYS).any():
        raise ValueError("Archived panel has duplicate condo/size/month keys")
    lookup = pd.MultiIndex.from_frame(panel[KEYS])
    wanted = panel[KEYS].copy()
    wanted["period"] = wanted["period"].map(lambda p: add_months(p, horizon))
    locations = lookup.get_indexer(pd.MultiIndex.from_frame(wanted))
    valid = locations >= 0
    price = panel["rent_psf_imp"].to_numpy()
    target_level = np.full(len(panel), np.nan, dtype=price.dtype)
    target_level[valid] = price[locations[valid]]
    target_observed = np.zeros(len(panel), dtype=bool)
    target_observed[valid] = panel["was_observed"].to_numpy(bool)[locations[valid]]
    if target_mode == "log_growth":
        with np.errstate(invalid="ignore"):
            target = np.log(target_level.clip(min=1e-9)) - np.log(price.clip(min=1e-9))
    elif target_mode == "level":
        target = target_level.copy()
    else:
        raise ValueError(f"Unsupported archived target mode: {target_mode}")
    return target, target_level, target_observed, wanted["period"].to_numpy()


def categorical_frame(frame, bundle):
    features = bundle["features"]
    cats = bundle.get("cat_features", bundle.get("categorical_features", []))
    stored = bundle["model"].booster_.pandas_categorical
    if len(cats) != len(stored):
        raise ValueError("Archived categorical metadata is incomplete")
    result = frame[features].copy()
    for name, levels in zip(cats, stored):
        values = result[name]
        if levels and isinstance(levels[0], str):
            values = values.astype("string")
        result[name] = pd.Categorical(values, categories=levels)
    return result, cats


def compare_rows(reference, reproduced, rtol, r2_atol):
    rows = []
    for metric in ["RMSE", "MAE", "MAPE_pct", "R2"]:
        before, after = reference[metric], reproduced[metric]
        tolerance = r2_atol if metric == "R2" else abs(before) * rtol
        rows.append(dict(metric=metric, reference=before, reproduced=after,
                         difference=after - before, tolerance=tolerance,
                         passed=bool(abs(after - before) <= tolerance)))
    if reference["n"] != reproduced["n"]:
        raise ValueError("Reference and reproduced test sample sizes differ")
    return rows


def read_panel(path, bundles, numeric_dtype):
    features = list(dict.fromkeys(f for b in bundles.values() for f in b["features"]))
    cat_names = set(c for b in bundles.values() for c in b.get("cat_features", b.get("categorical_features", [])))
    cols = list(dict.fromkeys(KEYS[:-1] + ["yyyymm", "rent_psf_imp", "was_observed"] + features))
    dtype = {c: numeric_dtype for c in cols
             if c not in cat_names | {"condo_name", "size_tier", "yyyymm", "was_observed"}}
    dtype.update({"condo_name": "category", "size_tier": "category", "was_observed": bool})
    for bundle in bundles.values():
        for col, levels in zip(bundle.get("cat_features", bundle.get("categorical_features", [])),
                               bundle["model"].booster_.pandas_categorical):
            dtype[col] = "string" if levels and isinstance(levels[0], str) else "int32"
    # Keys remain strings across chunks, avoiding per-chunk category-code mismatches.
    dtype["condo_name"] = dtype["size_tier"] = "string"
    pieces = []
    count = 0
    for chunk in pd.read_csv(path, usecols=cols, dtype=dtype, chunksize=100000):
        pieces.append(chunk)
        count += len(chunk)
        if count % 500000 == 0:
            print(f"  Loaded {count:,} archived feature rows", flush=True)
    panel = pd.concat(pieces, ignore_index=True).rename(columns={"yyyymm": "period"})
    for key in KEYS[:2]:
        panel[key] = panel[key].astype("category")
    if panel.duplicated(KEYS).any():
        raise ValueError("Archived panel has duplicate keys")
    return panel


def reference_cases(path, horizon):
    data = pd.read_csv(path, usecols=KEYS + ["horizon", "fair_eval_row", "y_true", "y_pred_imputed"])
    data = data.loc[data["horizon"].eq(horizon) & data["fair_eval_row"].eq(True)].copy()
    if data.duplicated(KEYS).any():
        raise ValueError("Reference predictions have duplicate fair keys")
    if data.empty:
        raise ValueError(f"No reference fair cases for h={horizon}")
    return data


def run_reference(settings, markets):
    profile_path = settings.reference_profile
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    for attr in ("train_start", "test_cutoff", "imputed_target_weight"):
        if getattr(settings, attr) != profile[attr]:
            raise ValueError(f"Reference profile requires {attr}={profile[attr]}; got {getattr(settings, attr)}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = settings.output_root / "04_forecasts" / f"reference_{profile['name']}_{settings.reference_mode}_{stamp}"
    for market in markets:
        source = (profile_path.parent / profile["markets"][market]["reference_run"]).resolve()
        if folder.resolve() == source or source in folder.resolve().parents:
            raise ValueError("Reproduction output must not be inside the reference run")
    folder.mkdir(parents=True)
    manifest = dict(profile=profile, profile_file=fingerprint(profile_path), mode=settings.reference_mode,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    settings=vars(settings), inputs=[], scope=profile.get("scope"))
    metrics, comparisons = [], []
    failed = False
    for market in markets:
        entry = profile["markets"][market]
        source = (profile_path.parent / entry["reference_run"]).resolve()
        if folder.resolve() == source or source in folder.resolve().parents:
            raise ValueError("Reproduction output must not be inside the reference run")
        bundles = {h: joblib.load(source / entry["model_pattern"].format(horizon=h)) for h in settings.horizons}
        panel_path = source / entry["feature_panel"]
        print(f"[reference {market}] {settings.reference_mode}; read-only input: {source}", flush=True)
        panel = read_panel(panel_path, bundles, profile.get("numeric_dtype", "float32"))
        manifest["inputs"].append(fingerprint(panel_path))
        reference_path = source / entry["reference_predictions"]
        manifest["inputs"].append(fingerprint(reference_path))
        for h, bundle in bundles.items():
            print(f"[reference {market} h={h}] Build exact-calendar labels", flush=True)
            mode = bundle["target_mode"]
            target, levels, target_obs, target_periods = target_arrays(panel, h, mode)
            train = (panel["period"].to_numpy() >= settings.train_start) & (target_periods < settings.test_cutoff) & np.isfinite(target)
            reference = reference_cases(reference_path, h)
            lookup = pd.MultiIndex.from_frame(panel[KEYS])
            test_idx = lookup.get_indexer(pd.MultiIndex.from_frame(reference[KEYS]))
            if (test_idx < 0).any():
                raise ValueError("Reference fair test keys are absent from the archived panel")
            if not (panel["was_observed"].to_numpy()[test_idx] & target_obs[test_idx]).all():
                raise ValueError("Fair test cases require observed origin and target")
            if not np.allclose(levels[test_idx], reference["y_true"], rtol=1e-6, atol=1e-6):
                raise ValueError("Reconstructed targets differ from archived actual prices")
            xtest, cats = categorical_frame(panel.iloc[test_idx], bundle)
            # Fresh inference is checked independently of refitting.
            replay = bundle["model"].predict(xtest)
            if mode == "log_growth":
                replay = np.exp(np.log(panel["rent_psf_imp"].to_numpy()[test_idx]) + replay)
            replay_score = score(reference["y_true"], replay)
            ref_score = score(reference["y_true"], reference["y_pred_imputed"])
            for label, row in [("reference", ref_score), ("archived_model_inference", replay_score)]:
                metrics.append(dict(market=market, horizon=h, model=label, **row))
            model_path = source / entry["model_pattern"].format(horizon=h)
            manifest["inputs"].append(fingerprint(model_path))
            if settings.reference_mode == "refit":
                xtrain, _ = categorical_frame(panel.loc[train], bundle)
                weight = np.where(target_obs[train], 1.0, profile["imputed_target_weight"])
                weight /= weight.mean()
                params = bundle["model"].get_params().copy()
                if settings.reference_threads is not None:
                    if not params.get("deterministic"):
                        raise ValueError("Thread overrides require a deterministic archived model")
                    params["n_jobs"] = settings.reference_threads
                model = LGBMRegressor(**params)
                started = time.perf_counter()

                def progress(env):
                    if (env.iteration + 1) % 100 == 0:
                        print(f"  {market} h={h}: tree {env.iteration + 1}/{params['n_estimators']}; {time.perf_counter()-started:.1f}s", flush=True)

                print(f"  Fit fresh model: {len(xtrain):,} train rows; {params['n_estimators']} trees; W{profile['imputed_target_weight']:g}", flush=True)
                model.fit(xtrain, target[train], sample_weight=weight,
                          categorical_feature=cats, callbacks=[progress])
                prediction = model.predict(xtest)
                if mode == "log_growth":
                    prediction = np.exp(np.log(panel["rent_psf_imp"].to_numpy()[test_idx]) + prediction)
                result = score(reference["y_true"], prediction)
                metrics.append(dict(market=market, horizon=h, model="fresh_refit", **result))
                joblib.dump(dict(model=model, features=bundle["features"], cat_features=cats,
                                 target_mode=mode, horizon=h, imputed_target_weight=profile["imputed_target_weight"],
                                 runtime_threads=params["n_jobs"],
                                 reference_model=str(model_path)), folder / f"{market}_h{h}_refit.pkl")
                del xtrain, weight, model
            else:
                prediction, result = replay, replay_score
            rows = compare_rows(ref_score, result, profile["error_metrics_rtol"], profile["r2_atol"])
            for row in rows:
                comparisons.append(dict(market=market, horizon=h, mode=settings.reference_mode, **row))
            failed |= not all(row["passed"] for row in rows)
            output = reference[KEYS + ["y_true", "y_pred_imputed"]].rename(columns={"y_pred_imputed": "reference_prediction"})
            output["reproduced_prediction"] = prediction
            output["archived_model_inference"] = replay
            output.to_csv(folder / f"{market}_h{h}_predictions.csv", index=False)
            pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
            pd.DataFrame(comparisons).to_csv(folder / "comparison.csv", index=False)
            print(f"  {market} h={h}: reference RMSE={ref_score['RMSE']:.6f}; reproduced RMSE={result['RMSE']:.6f}; pass={all(row['passed'] for row in rows)}", flush=True)
            del xtest, target, levels, target_obs, target_periods
            gc.collect()
        del panel, bundles
        gc.collect()
    manifest["status"] = "tolerance_failed" if failed else "passed"
    (folder / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2), encoding="utf-8")
    print(f"[reference complete] {folder}", flush=True)
    if failed:
        raise ValueError(f"Reference tolerance failed; inspect {folder / 'comparison.csv'}")
    return folder
