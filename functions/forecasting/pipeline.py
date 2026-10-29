"""Shared static and fixed-width rolling forecasts for full-district panels."""

import gc
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from functions.common import add_months, rolling_windows
from functions.config import PACKAGE_ROOT
from functions.forecasting.optuna_helpers import (
    DEFAULT_LGBM_PARAMS,
    lgbm_deterministic_params,
    run_optuna,
)
from functions.forecasting.tuning import (
    run_optuna_log_growth_level_rmse,
    run_optuna_target_weighted,
)
from functions.forecasting.utils import month_cv_split


def horizon_frame(panel, horizon, observed_only=False, target_mode="level"):
    """Exact calendar join; project_idx is the existing condo/size series ID."""
    if panel.duplicated(["project_idx", "period"]).any():
        raise ValueError("Forecast input has duplicate project/month keys")
    base_col = "rent_psf_obs" if observed_only else "rent_psf_imp"
    frame = panel.loc[panel["was_observed"]].copy() if observed_only else panel.copy()
    target = panel[["project_idx", "period", base_col, "was_observed"]].rename(
        columns={
            "period": "target_period",
            base_col: "target_level",
            "was_observed": "target_observed",
        }
    )
    frame["target_period"] = frame["period"].map(lambda p: add_months(p, horizon))
    frame = frame.merge(
        target, on=["project_idx", "target_period"], how="left", validate="one_to_one"
    )
    frame["target_observed"] = frame["target_observed"].astype("boolean").fillna(False).astype(bool)
    frame["fair_eval_row"] = frame["was_observed"] & frame["target_observed"]
    frame["base_log"] = np.log(frame[base_col].clip(lower=1e-9))
    frame["model_target"] = (
        np.log(frame["target_level"].clip(lower=1e-9)) - frame["base_log"]
        if target_mode == "log_growth"
        else frame["target_level"]
    )
    if observed_only:
        frame = frame.loc[frame["model_target"].notna()].copy()
    return frame


def training_rows(frame, window, horizon, train_start):
    """Strict label availability: target month must precede the test origin."""
    return (
        frame["period"].ge(max(window["train_start"], train_start))
        & frame["period"].le(window["train_end"])
        & frame["target_period"].lt(window["test_start"])
        & frame["model_target"].notna()
    )


def cv_folds(frame, features, horizon, settings):
    months = np.sort(frame["period"].unique())
    splitter = TimeSeriesSplit(
        n_splits=settings.cv_splits, test_size=settings.validation_months, gap=horizon
    )
    folds = list(
        month_cv_split(frame[features], frame["model_target"], frame["period"], months, splitter)
    )
    for x_train, _, x_valid, _ in folds:
        latest_target = frame.loc[x_train.index, "target_period"].max()
        earliest_valid = frame.loc[x_valid.index, "period"].min()
        if latest_target >= earliest_valid:
            raise ValueError("CV training labels overlap the validation origin")
    return folds


def fit_model(train, features, categoricals, market, arm, horizon, settings):
    from lightgbm import LGBMRegressor

    x, y = train[features], train["model_target"]
    cats = [c for c in categoricals if c in features]
    weights = pd.Series(
        np.where(train["target_observed"], 1.0, settings.imputed_target_weight), index=train.index
    )
    if arm == "observed":
        weights[:] = 1.0
    if not len(train) or weights.mean() <= 0:
        raise ValueError(f"{market} h={horizon} {arm}: no positive-weight training rows")
    # A named fast mode is useful for smoke runs, without changing quality weights.
    if settings.fast:
        params = lgbm_deterministic_params(DEFAULT_LGBM_PARAMS.copy(), settings.seed)
        model = LGBMRegressor(**params)
        model.fit(x, y, sample_weight=(weights / weights.mean()), categorical_feature=cats)
        return params, model, None
    folds = cv_folds(train, features, horizon, settings)
    kwargs = dict(
        n_trials=settings.trials,
        tag=f"h={horizon} {'Imputed' if arm == 'imputed' else 'ObsOnly'}",
        fast_mode=False,
        search_profile=settings.search_profile,
        n_cv_folds_search=settings.cv_splits,
        random_seed=settings.seed,
    )
    if market == "sales":
        return run_optuna_log_growth_level_rmse(
            x,
            y,
            folds,
            cats,
            level_context_df=train,
            base_log_col="base_log",
            level_target_col="target_level",
            sample_weight_context=weights if arm == "imputed" else None,
            eval_mask_col="fair_eval_row" if arm == "imputed" else None,
            **kwargs,
        )
    if arm == "imputed":
        return run_optuna_target_weighted(
            x,
            y,
            folds,
            cats,
            sample_weight_context=weights,
            eval_context_df=train,
            eval_mask_col="fair_eval_row",
            **kwargs,
        )
    return run_optuna(x, y, folds, cats, **kwargs)


def metric_row(frame, pred_col, market, horizon, window_idx, arm):
    fair = frame["fair_eval_row"] & frame["target_level"].notna() & frame[pred_col].notna()
    y, p = frame.loc[fair, "target_level"].to_numpy(), frame.loc[fair, pred_col].to_numpy()
    row = dict(market=market, horizon=horizon, window_idx=window_idx, model=arm, n=len(y))
    if len(y):
        positive = np.abs(y) > 1e-9
        row.update(
            RMSE=float(np.sqrt(mean_squared_error(y, p))),
            MAE=float(mean_absolute_error(y, p)),
            MAPE_pct=float(np.mean(np.abs((y[positive] - p[positive]) / y[positive])) * 100)
            if positive.any()
            else np.nan,
            R2=float(r2_score(y, p)) if len(y) > 1 else np.nan,
        )
    return row


def feature_columns(lists, horizon, arm, market, settings):
    key = (
        ("FEATURES_A" if arm == "imputed" else "FEATURES_OBS")
        if horizon == 1
        else f"FEATURES_T{horizon}" + ("_OBS" if arm == "observed" else "")
    )
    features = list(lists[key])
    if settings.selected_features:
        name = "sale" if market == "sales" else "rental"
        file = (
            PACKAGE_ROOT
            / "config"
            / "selected_features"
            / name
            / f"drop_features_h{horizon}_full_rolling_all_features.json"
        )
        drops = json.loads(file.read_text(encoding="utf-8"))
        features = [f for f in features if f not in set(drops)]
    if not features:
        raise ValueError("No model features remain")
    return list(dict.fromkeys(features))


def run(settings, markets=("rental", "sales"), mode="static"):
    if mode not in ("static", "rolling"):
        raise ValueError("mode must be static or rolling")
    for market in markets:
        from functions.forecasting import prepare_rental, prepare_sales

        prepare = prepare_rental.prepare if market == "rental" else prepare_sales.prepare
        panel, lists, cats, size_tier = prepare(settings)
        target_mode = "log_growth" if market == "sales" else "level"
        end = int(panel["period"].max())
        if end < settings.test_cutoff:
            raise ValueError("Forecast cutoff is later than the last input month")
        windows = (
            rolling_windows(
                settings.test_cutoff,
                end,
                settings.rolling_train_months,
                settings.rolling_test_months,
            )
            if mode == "rolling"
            else [
                dict(
                    train_start=settings.train_start,
                    train_end=add_months(settings.test_cutoff, -1),
                    test_start=settings.test_cutoff,
                    test_end=end,
                )
            ]
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = settings.folder("04_forecasts") / f"{market}_{mode}_{stamp}"
        folder.mkdir(parents=True)
        pd.DataFrame(windows).to_csv(folder / "window_schedule.csv", index=False)
        (folder / "settings.json").write_text(
            json.dumps(vars(settings), default=str, indent=2), encoding="utf-8"
        )
        all_metrics = []
        for horizon in settings.horizons:
            imputed = horizon_frame(panel, horizon, target_mode=target_mode)
            observed = horizon_frame(panel, horizon, observed_only=True, target_mode=target_mode)
            for idx, window in enumerate(windows, 1):
                predictions = None
                print(
                    f"[forecast] {market} {mode} h={horizon} window {idx}/{len(windows)}; test {window['test_start']}-{window['test_end']}"
                )
                for arm, frame in [("imputed", imputed), ("observed", observed)]:
                    features = feature_columns(lists, horizon, arm, market, settings)
                    train = frame.loc[
                        training_rows(frame, window, horizon, settings.train_start)
                    ].copy()
                    test = frame.loc[
                        frame["period"].between(window["test_start"], window["test_end"])
                    ].copy()
                    if test.empty:
                        print(f"  {arm}: no prediction rows in this window")
                        continue
                    print(
                        f"  {arm}: {len(train):,} train rows; last available target={train['target_period'].max()}; {len(test):,} predictions"
                    )
                    params, model, study = fit_model(
                        train, features, cats, market, arm, horizon, settings
                    )
                    predicted = model.predict(test[features])
                    if target_mode == "log_growth":
                        predicted = np.exp(test["base_log"].to_numpy() + predicted)
                    col = f"y_pred_{arm}"
                    test[col] = predicted
                    if not np.isfinite(predicted).all():
                        raise ValueError(f"Nonfinite predictions: {market} h={horizon} {arm}")
                    joblib.dump(
                        dict(
                            model=model,
                            features=features,
                            categorical_features=[c for c in cats if c in features],
                            target_mode=target_mode,
                            market=market,
                            horizon=horizon,
                            window=window,
                            params=params,
                            origin_price_column=("sale" if market == "sales" else "rent")
                            + ("_psf_imp" if arm == "imputed" else "_psf_obs"),
                            arm=arm,
                        ),
                        folder / f"h{horizon}_w{idx}_{arm}.pkl",
                    )
                    pd.DataFrame(
                        {"feature": features, "gain": model.booster_.feature_importance("gain")}
                    ).sort_values("gain", ascending=False).to_csv(
                        folder / f"h{horizon}_w{idx}_{arm}_importance.csv", index=False
                    )
                    if study is not None:
                        study.trials_dataframe().to_csv(
                            folder / f"h{horizon}_w{idx}_{arm}_trials.csv", index=False
                        )
                    if arm == "imputed":
                        meta = [
                            c
                            for c in [
                                "project_idx",
                                "condo_name",
                                "district",
                                "period",
                                "target_period",
                                "target_level",
                                "target_observed",
                                "was_observed",
                                "fair_eval_row",
                            ]
                            if c in test
                        ]
                        predictions = test[meta + [col]].copy()
                        predictions["size_tier"] = predictions["project_idx"].map(size_tier)
                    elif predictions is not None:
                        predictions = predictions.merge(
                            test[["project_idx", "period", col]],
                            on=["project_idx", "period"],
                            how="left",
                            validate="one_to_one",
                        )
                    del train, test, model
                    gc.collect()
                if predictions is None:
                    raise ValueError(
                        f"No imputed predictions for {market} h={horizon} window={idx}"
                    )
                if "y_pred_observed" not in predictions:
                    predictions["y_pred_observed"] = np.nan
                # Enforce identical rows for both model metrics.
                predictions["fair_eval_row"] &= predictions["y_pred_observed"].notna()
                predictions["market"], predictions["horizon"], predictions["window_idx"] = (
                    market,
                    horizon,
                    idx,
                )
                for arm in ("imputed", "observed"):
                    all_metrics.append(
                        metric_row(predictions, f"y_pred_{arm}", market, horizon, idx, arm)
                    )
                    predictions[f"residual_{arm}"] = (
                        predictions["target_level"] - predictions[f"y_pred_{arm}"]
                    )
                predictions.to_csv(folder / f"predictions_h{horizon}_w{idx}.csv", index=False)
                del predictions
            del imputed, observed
            gc.collect()
        pd.DataFrame(all_metrics).to_csv(folder / "metrics.csv", index=False)
        print(f"[forecast complete] {folder}")
        del panel
        gc.collect()
