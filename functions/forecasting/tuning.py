"""Original full-district quality-weighted tuning and level-RMSE scoring. Used by static and rolling forecasts."""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from functions.forecasting.optuna_helpers import (
    run_optuna,
    DEFAULT_LGBM_PARAMS,
    set_global_seed,
    optuna_seed_for_tag,
    SEARCH_PROFILES,
    N_ESTIMATORS_MAX_SEARCH,
    EARLY_STOP_ROUNDS,
    N_CV_FOLDS_SEARCH,
    _FAST_FIXED,
    _suggest_lgbm_params,
    lgbm_deterministic_params,
)


def run_optuna_log_growth_level_rmse(
    X_tr_val,
    y_tr_val,
    cv_splits_list,
    cat_features,
    *,
    level_context_df,
    base_log_col,
    level_target_col,
    sample_weight_context=None,
    eval_mask_col=None,
    n_trials=100,
    tag="",
    fast_mode=True,
    search_profile="fast",
    n_cv_folds_search=N_CV_FOLDS_SEARCH,
    n_estimators_max_search=N_ESTIMATORS_MAX_SEARCH,
    early_stop=EARLY_STOP_ROUNDS,
    linear_tree=False,
    linear_lambda=0.0,
    TARGET_MODE="log_growth",
    random_seed=42,
):
    """Fit log growth; rank trials on masked reconstructed price-level RMSE."""
    if TARGET_MODE != "log_growth":
        raise ValueError("run_optuna_log_growth_level_rmse requires TARGET_MODE=log_growth")
    if search_profile not in SEARCH_PROFILES:
        raise ValueError(f"Unknown search_profile={search_profile!r}")
    if fast_mode:
        if sample_weight_context is not None or eval_mask_col is not None:
            raise ValueError("V3 weighting/masked scoring requires FAST_MODE=False")
        return run_optuna(
            X_tr_val,
            y_tr_val,
            cv_splits_list,
            cat_features,
            n_trials=n_trials,
            tag=tag,
            fast_mode=True,
            search_profile=search_profile,
            linear_tree=linear_tree,
            linear_lambda=linear_lambda,
        )
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv_search = list(cv_splits_list[:n_cv_folds_search])
    if len(cv_search) < n_cv_folds_search:
        raise ValueError(f"Need {n_cv_folds_search} CV folds; got {len(cv_search)}")
    if sample_weight_context is not None:
        sample_weight_context = pd.Series(
            sample_weight_context, index=level_context_df.index, dtype=float
        )
        if (
            sample_weight_context.isna().any()
            or (sample_weight_context < 0).any()
            or sample_weight_context.mean() <= 0
        ):
            raise ValueError(f"Invalid training weights for {tag}")
    task_seed = optuna_seed_for_tag(tag, base_seed=random_seed)
    set_global_seed(task_seed)

    def objective(trial):
        trial_params = _suggest_lgbm_params(trial, search_profile)
        params = lgbm_deterministic_params(
            {"n_estimators": n_estimators_max_search, **trial_params},
            task_seed,
            linear_tree=linear_tree,
            linear_lambda=linear_lambda,
        )
        params["metric"] = "None"
        fold_level_rmse = []
        best_iters = []
        fold_eval_rows = []
        for X_tr, y_tr, X_va, y_va in cv_search:
            base_log_va = level_context_df.loc[X_va.index, base_log_col].to_numpy()
            y_level_va = level_context_df.loc[X_va.index, level_target_col].to_numpy()
            if eval_mask_col is None:
                eval_mask_va = np.ones(len(X_va), dtype=bool)
            else:
                eval_mask_va = (
                    level_context_df.loc[X_va.index, eval_mask_col]
                    .fillna(False)
                    .astype(bool)
                    .to_numpy()
                )
            if eval_mask_va.sum() == 0:
                raise ValueError(f"No masked validation rows in {tag}; check {eval_mask_col!r}")
            if not np.isfinite(base_log_va).all() or not np.isfinite(y_level_va).all():
                raise ValueError(
                    f"Non-finite level context in {tag}: {base_log_col}, {level_target_col}"
                )
            train_weight = None
            if sample_weight_context is not None:
                train_weight = sample_weight_context.loc[X_tr.index].to_numpy(dtype=float)
                train_weight = train_weight / train_weight.mean()

            def reconstructed_level_rmse(y_true_growth, y_pred_growth):
                pred_level = np.exp(base_log_va + np.asarray(y_pred_growth))
                err = y_level_va[eval_mask_va] - pred_level[eval_mask_va]
                rmse = float(np.sqrt(np.mean(err**2)))
                return ("fair_level_rmse_sgd_psf", rmse, False)

            model = LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=train_weight,
                eval_set=[(X_va, y_va)],
                eval_metric=reconstructed_level_rmse,
                callbacks=[
                    lgb.early_stopping(early_stop, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(0),
                ],
                categorical_feature=cat_features,
            )
            pred_growth = model.predict(X_va)
            pred_level = np.exp(base_log_va + pred_growth)
            fold_level_rmse.append(
                float(np.sqrt(np.mean((y_level_va[eval_mask_va] - pred_level[eval_mask_va]) ** 2)))
            )
            best_iters.append(model.best_iteration_)
            fold_eval_rows.append(int(eval_mask_va.sum()))
        trial.set_user_attr("best_iterations", best_iters)
        trial.set_user_attr("fair_eval_rows", fold_eval_rows)
        return float(np.mean(fold_level_rmse))

    score_label = (
        f"masked by {eval_mask_col}" if eval_mask_col is not None else "all validation rows"
    )
    print(
        f"\n  Optuna [{tag}] — weighted log-growth target, price-level RMSE scoring ({score_label}); profile={search_profile!r}, trials={n_trials}, folds={len(cv_search)}"
    )
    started = time.time()
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=task_seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = lgbm_deterministic_params(
        study.best_params.copy(), task_seed, linear_tree=linear_tree, linear_lambda=linear_lambda
    )
    if search_profile == "fast":
        best_params.update(_FAST_FIXED)
    best_iters = study.best_trial.user_attrs.get("best_iterations", [])
    best_params["n_estimators"] = max(1, int(np.median(best_iters))) if best_iters else 500
    best_params["metric"] = "l2"
    print(
        f"  Done in {(time.time() - started) / 60:.1f} min | best CV fair price-level RMSE={study.best_value:.4f} SGD/sqft"
    )
    print(
        f"  Best-trial fair eval rows per fold: {study.best_trial.user_attrs.get('fair_eval_rows', [])}"
    )
    print(f"  Best params: {best_params}")
    final_weight = None
    if sample_weight_context is not None:
        final_weight = sample_weight_context.loc[X_tr_val.index].to_numpy(dtype=float)
        final_weight = final_weight / final_weight.mean()
    set_global_seed(task_seed)
    final_model = LGBMRegressor(**best_params)
    final_model.fit(
        X_tr_val, y_tr_val, sample_weight=final_weight, categorical_feature=cat_features
    )
    return (best_params, final_model, study)


def run_optuna_target_weighted(
    X_tr_val,
    y_tr_val,
    cv_splits_list,
    cat_features,
    *,
    sample_weight_context,
    eval_context_df,
    eval_mask_col,
    n_trials=100,
    tag="",
    fast_mode=True,
    search_profile="fast",
    n_cv_folds_search=N_CV_FOLDS_SEARCH,
    n_estimators_max_search=N_ESTIMATORS_MAX_SEARCH,
    early_stop=EARLY_STOP_ROUNDS,
    linear_tree=False,
    linear_lambda=0.0,
    random_seed=42,
):
    """Tune/final-fit with target-quality weights and observed-row CV RMSE."""
    if search_profile not in SEARCH_PROFILES:
        raise ValueError(f"Unknown search_profile={search_profile!r}")
    weights = pd.Series(sample_weight_context, index=eval_context_df.index, dtype=float)
    if weights.isna().any() or (weights < 0).any() or weights.mean() <= 0:
        raise ValueError(f"Invalid target-quality weights for {tag}")
    task_seed = optuna_seed_for_tag(tag, base_seed=random_seed)
    set_global_seed(task_seed)

    def normalized_weight(index):
        out = weights.loc[index].to_numpy(dtype=float)
        if out.mean() <= 0:
            raise ValueError(f"All training weights are zero for {tag}")
        return out / out.mean()

    if fast_mode:
        params = lgbm_deterministic_params(
            DEFAULT_LGBM_PARAMS.copy(),
            task_seed,
            linear_tree=linear_tree,
            linear_lambda=linear_lambda,
        )
        model = LGBMRegressor(**params)
        model.fit(
            X_tr_val,
            y_tr_val,
            sample_weight=normalized_weight(X_tr_val.index),
            categorical_feature=cat_features,
        )
        return (params, model, None)
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv_search = list(cv_splits_list[:n_cv_folds_search])
    if len(cv_search) < n_cv_folds_search:
        raise ValueError(f"Need {n_cv_folds_search} CV folds; got {len(cv_search)}")

    def objective(trial):
        trial_params = _suggest_lgbm_params(trial, search_profile)
        params = lgbm_deterministic_params(
            {"n_estimators": n_estimators_max_search, **trial_params},
            task_seed,
            linear_tree=linear_tree,
            linear_lambda=linear_lambda,
        )
        params["metric"] = "None"
        fold_rmse, best_iters, fold_eval_rows = ([], [], [])
        for X_tr, y_tr, X_va, y_va in cv_search:
            eval_mask = (
                eval_context_df.loc[X_va.index, eval_mask_col].fillna(False).astype(bool).to_numpy()
            )
            if eval_mask.sum() == 0:
                raise ValueError(f"No observed validation rows in {tag}")

            def observed_rmse(y_true, y_pred):
                err = np.asarray(y_true)[eval_mask] - np.asarray(y_pred)[eval_mask]
                return ("fair_observed_rmse", float(np.sqrt(np.mean(err**2))), False)

            model = LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=normalized_weight(X_tr.index),
                eval_set=[(X_va, y_va)],
                eval_metric=observed_rmse,
                callbacks=[
                    lgb.early_stopping(early_stop, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(0),
                ],
                categorical_feature=cat_features,
            )
            pred = model.predict(X_va)
            fold_rmse.append(
                float(np.sqrt(np.mean((np.asarray(y_va)[eval_mask] - pred[eval_mask]) ** 2)))
            )
            best_iters.append(model.best_iteration_)
            fold_eval_rows.append(int(eval_mask.sum()))
        trial.set_user_attr("best_iterations", best_iters)
        trial.set_user_attr("fair_eval_rows", fold_eval_rows)
        return float(np.mean(fold_rmse))

    print(
        f"\n  Optuna [{tag}] — target-quality weighted fitting; observed-row CV RMSE; profile={search_profile!r}, trials={n_trials}"
    )
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=task_seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = lgbm_deterministic_params(
        study.best_params.copy(), task_seed, linear_tree=linear_tree, linear_lambda=linear_lambda
    )
    if search_profile == "fast":
        best_params.update(_FAST_FIXED)
    best_iters = study.best_trial.user_attrs.get("best_iterations", [])
    best_params["n_estimators"] = max(1, int(np.median(best_iters))) if best_iters else 500
    print(f"  Best fair observed CV RMSE: {study.best_value:.4f}")
    print(f"  Fair eval rows: {study.best_trial.user_attrs.get('fair_eval_rows', [])}")
    set_global_seed(task_seed)
    final_model = LGBMRegressor(**best_params)
    final_model.fit(
        X_tr_val,
        y_tr_val,
        sample_weight=normalized_weight(X_tr_val.index),
        categorical_feature=cat_features,
    )
    return (best_params, final_model, study)
