"""
optuna_helpers.py
-----------------
LightGBM / Optuna utilities shared by all prediction notebooks.

Public API
----------
DEFAULT_LGBM_PARAMS        : dict — sensible defaults (objective=regression / L2 loss)
load_fixed_lgbm_params     : load fixed-params JSON (optional); fallback to defaults above
SEARCH_PROFILES            : ('fast', 'full')
N_ESTIMATORS_MAX_SEARCH    : tree ceiling during Optuna CV (early stopping applies)
EARLY_STOP_ROUNDS          : early-stopping patience in CV objective
N_CV_FOLDS_SEARCH          : CV folds used inside Optuna (set N_SPLITS >= this, typically 2)
set_global_seed(seed)      : reset Python / NumPy RNGs
optuna_seed_for_tag(tag)   : stable per-task seed derived from SEED
lgbm_deterministic_params(params, seed) : add reproducibility knobs
run_optuna(...)            : run Optuna search or FAST_MODE fit; return (params, model, study)
refit_lgbm(...)            : refit with fixed params (rolling windows between Optuna recalibrations)
"""

from __future__ import annotations
import json
import random
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor

SEED = 42
SEARCH_PROFILES = ("fast", "full")
N_ESTIMATORS_MAX_SEARCH = 1500
EARLY_STOP_ROUNDS = 50
N_CV_FOLDS_SEARCH = 2
LGBM_LOSS_PARAMS = {"objective": "regression", "metric": "l2"}
DEFAULT_LGBM_PARAMS = {
    **LGBM_LOSS_PARAMS,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "verbose": -1,
}
LINEAR_TREE_DEFAULT = False
LINEAR_LAMBDA_DEFAULT = 0.0
DEFAULT_LGBM_PARAMS_LINEAR_TREE = {
    **LGBM_LOSS_PARAMS,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "verbose": -1,
}
_FAST_FIXED = {
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}


def load_fixed_lgbm_params(
    fixed_params_path: str | Path | None, *, linear_tree: bool = False
) -> dict:
    """
    Load fixed LightGBM hyperparameters from a JSON file.

    Expected shape::

        {
          "_comment": "...",
          "standard": { "n_estimators": 500, ... },
          "linear_tree": { "n_estimators": 100, ... }
        }

    ``objective`` / ``metric`` are always taken from ``LGBM_LOSS_PARAMS``;
    keys starting with ``_`` are ignored.  If the file is missing or invalid,
    falls back to ``DEFAULT_LGBM_PARAMS`` or ``DEFAULT_LGBM_PARAMS_LINEAR_TREE``.
    """
    key = "linear_tree" if linear_tree else "standard"
    fallback = DEFAULT_LGBM_PARAMS_LINEAR_TREE if linear_tree else DEFAULT_LGBM_PARAMS
    if fixed_params_path is None:
        return dict(fallback)
    path = Path(fixed_params_path)
    if not path.is_file():
        print(
            f"  Fixed-params file not found ({path}); using built-in {key} defaults from optuna_helpers"
        )
        return dict(fallback)
    try:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"  Could not read fixed-params file ({path}): {exc}; using built-in {key} defaults from optuna_helpers"
        )
        return dict(fallback)
    if not isinstance(raw, dict):
        print(
            f"  Fixed-params file must be a JSON object ({path}); using built-in {key} defaults from optuna_helpers"
        )
        return dict(fallback)
    if key in raw and isinstance(raw[key], dict):
        user = {k: v for k, v in raw[key].items() if not str(k).startswith("_")}
    elif "params" in raw and isinstance(raw["params"], dict):
        user = {k: v for k, v in raw["params"].items() if not str(k).startswith("_")}
    else:
        skip = {"standard", "linear_tree", "params", "_comment", "_meta"}
        user = {k: v for k, v in raw.items() if k not in skip and (not str(k).startswith("_"))}
    params = {**LGBM_LOSS_PARAMS, **user}
    print(f"  Loaded fixed LightGBM params from {path} ({key})")
    return params


def set_global_seed(seed: int) -> None:
    """Reset Python / NumPy RNGs before an Optuna study or CV refit."""
    random.seed(seed)
    np.random.seed(seed)


def optuna_seed_for_tag(tag: str, base_seed: int = SEED) -> int:
    """
    Stable per-task seed.  h=1/h=12/h=24 × Imputed/ObsOnly are independent
    but fully reproducible across runs.
    """
    s = sum(((i + 1) * ord(c) for i, c in enumerate(str(tag))))
    return int(base_seed) + s % 10000


def lgbm_deterministic_params(
    params: dict,
    seed: int,
    *,
    linear_tree: bool = LINEAR_TREE_DEFAULT,
    linear_lambda: float = LINEAR_LAMBDA_DEFAULT,
) -> dict:
    """Add LightGBM knobs that improve repeatability across runs/machines."""
    p = dict(params)
    p["random_state"] = seed
    p.setdefault("force_col_wise", True)
    p.setdefault("deterministic", True)
    p["n_jobs"] = 1
    p["verbose"] = p.get("verbose", -1)
    for k, v in LGBM_LOSS_PARAMS.items():
        p.setdefault(k, v)
    if linear_tree:
        p["linear_tree"] = True
        p["linear_lambda"] = float(linear_lambda)
    else:
        p.pop("linear_tree", None)
        p.pop("linear_lambda", None)
    return p


def _cv_splits_for_search(cv_splits_list: list, n_cv_folds_search: int) -> list:
    """Use the first N folds during Optuna; notebooks may still build more for other use."""
    if not cv_splits_list:
        return cv_splits_list
    n = min(n_cv_folds_search, len(cv_splits_list))
    return cv_splits_list[:n]


def _suggest_lgbm_params(trial, search_profile: str) -> dict:
    """Return tunable LightGBM hyperparameters for one Optuna trial."""
    if search_profile == "fast":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127]),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [50, 100, 200]),
            **_FAST_FIXED,
        }
    if search_profile == "full":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        }
    raise ValueError(f"search_profile must be one of {SEARCH_PROFILES}, got {search_profile!r}")


def _make_optuna_objective(
    cv_splits_list,
    cat_features,
    lgbm_seed,
    search_profile: str,
    *,
    n_estimators_max: int = N_ESTIMATORS_MAX_SEARCH,
    early_stop: int = EARLY_STOP_ROUNDS,
    linear_tree: bool = LINEAR_TREE_DEFAULT,
    linear_lambda: float = LINEAR_LAMBDA_DEFAULT,
):
    """Optuna objective — early stopping on held-out val fold during search."""

    def objective(trial):
        trial_params = _suggest_lgbm_params(trial, search_profile)
        params = lgbm_deterministic_params(
            {"n_estimators": n_estimators_max, **trial_params},
            lgbm_seed,
            linear_tree=linear_tree,
            linear_lambda=linear_lambda,
        )
        fold_maes, best_iters = ([], [])
        for X_tr, y_tr, X_va, y_va in cv_splits_list:
            m = LGBMRegressor(**params)
            m.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)],
                categorical_feature=cat_features,
            )
            preds = m.predict(X_va)
            rmse = np.sqrt(np.mean((y_va - preds) ** 2))
            fold_maes.append(rmse)
            best_iters.append(m.best_iteration_)
        trial.set_user_attr("best_iterations", best_iters)
        return np.mean(fold_maes)

    return objective


def run_optuna(
    X_tr_val,
    y_tr_val,
    cv_splits_list: list,
    cat_features: list,
    n_trials: int = 100,
    tag: str = "",
    fast_mode: bool = True,
    search_profile: str = "fast",
    n_cv_folds_search: int = N_CV_FOLDS_SEARCH,
    n_estimators_max_search: int = N_ESTIMATORS_MAX_SEARCH,
    early_stop: int = EARLY_STOP_ROUNDS,
    linear_tree: bool = LINEAR_TREE_DEFAULT,
    linear_lambda: float = LINEAR_LAMBDA_DEFAULT,
    fixed_params_path: str | Path | None = None,
    *,
    random_seed=42,
):
    """
    Run Optuna hyperparameter search (or skip when fast_mode=True).

    Parameters
    ----------
    X_tr_val, y_tr_val : full training feature matrix and target
    cv_splits_list     : list of (X_tr, y_tr, X_va, y_va) tuples; may be []
                         when fast_mode=True (bypassed anyway)
    cat_features       : list of categorical column names
    n_trials           : number of Optuna trials (ignored in fast_mode)
    tag                : string key for reproducible per-task seed
    fast_mode          : if True, skip Optuna; use fixed_params_path JSON or defaults
    fixed_params_path  : optional JSON for fast_mode fixed params (see load_fixed_lgbm_params)
    search_profile     : 'fast' (narrow grid) or 'full' (original wide grid)
    n_cv_folds_search  : use first N folds from cv_splits_list during search
    n_estimators_max_search : tree ceiling per CV fold in search
    early_stop         : early-stopping patience in search CV
    linear_tree        : if True, fit piecewise linear models at leaves (L2 objective only)
    linear_lambda      : L2 regularization on leaf linear coefficients when linear_tree=True

    Returns
    -------
    (best_params, best_model, study)  — study is None in fast_mode
    """
    if search_profile not in SEARCH_PROFILES:
        raise ValueError(f"search_profile must be one of {SEARCH_PROFILES}, got {search_profile!r}")
    task_seed = optuna_seed_for_tag(tag, base_seed=random_seed)
    set_global_seed(task_seed)
    if fast_mode:
        fast_params = load_fixed_lgbm_params(fixed_params_path, linear_tree=linear_tree)
        print(
            f"\n  FAST MODE [{tag}] — skipping Optuna ({n_trials} trials); fixed LightGBM params"
            + ("; linear_tree=True" if linear_tree else "")
        )
        t0 = time.time()
        best_params = lgbm_deterministic_params(
            fast_params, task_seed, linear_tree=linear_tree, linear_lambda=linear_lambda
        )
        model = LGBMRegressor(**best_params)
        model.fit(X_tr_val, y_tr_val, categorical_feature=cat_features)
        print(f"  Done in {time.time() - t0:.1f}s  |  seed={task_seed}")
        return (best_params, model, None)
    try:
        import optuna as _optuna

        _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError("pip install optuna") from e
    cv_search = _cv_splits_for_search(cv_splits_list, n_cv_folds_search)
    if len(cv_search) < n_cv_folds_search:
        raise ValueError(
            f"Need at least {n_cv_folds_search} CV folds for Optuna search; got {len(cv_search)} (build cv_splits with N_SPLITS >= {n_cv_folds_search})"
        )
    print(
        f"\n  Optuna search [{tag}] — profile={search_profile!r}, {n_trials} trials, {len(cv_search)} CV fold(s), n_estimators_max={n_estimators_max_search}, early_stop={early_stop}"
        + (", linear_tree=True" if linear_tree else "")
        + f"  |  seed={task_seed} ..."
    )
    t0 = time.time()
    study = _optuna.create_study(
        direction="minimize", sampler=_optuna.samplers.TPESampler(seed=task_seed)
    )
    study.optimize(
        _make_optuna_objective(
            cv_search,
            cat_features,
            task_seed,
            search_profile,
            n_estimators_max=n_estimators_max_search,
            early_stop=early_stop,
            linear_tree=linear_tree,
            linear_lambda=linear_lambda,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    best_params = lgbm_deterministic_params(
        study.best_params.copy(), task_seed, linear_tree=linear_tree, linear_lambda=linear_lambda
    )
    if search_profile == "fast":
        best_params.update(_FAST_FIXED)
    best_iters = study.best_trial.user_attrs.get("best_iterations", [])
    best_params["n_estimators"] = max(1, int(np.median(best_iters))) if best_iters else 500
    print(f"  Done in {elapsed / 60:.1f} min  |  best CV RMSE: {study.best_value:.4f}")
    print(f"  Best params: {best_params}")
    set_global_seed(task_seed)
    model = LGBMRegressor(**best_params)
    model.fit(X_tr_val, y_tr_val, categorical_feature=cat_features)
    return (best_params, model, study)


def refit_lgbm(
    X_tr_val,
    y_tr_val,
    params: dict,
    cat_features: list,
    tag: str = "",
    linear_tree: bool = LINEAR_TREE_DEFAULT,
    linear_lambda: float = LINEAR_LAMBDA_DEFAULT,
):
    """
    Refit LightGBM on new training data using fixed hyperparameters (no Optuna).
    Used between rolling-window recalibration steps.
    """
    task_seed = optuna_seed_for_tag(tag)
    set_global_seed(task_seed)
    fit_params = lgbm_deterministic_params(
        dict(params), task_seed, linear_tree=linear_tree, linear_lambda=linear_lambda
    )
    print(
        f"\n  Refit [{tag}] — reused hyperparameters (n_estimators={fit_params.get('n_estimators', '?')}) ...",
        end=" ",
    )
    t0 = time.time()
    model = LGBMRegressor(**fit_params)
    model.fit(X_tr_val, y_tr_val, categorical_feature=cat_features)
    print(f"done in {time.time() - t0:.1f}s  |  seed={task_seed}")
    return (fit_params, model)
