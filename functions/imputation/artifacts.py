"""Shared rental/sales serialization for evaluated VAE runs and model weights."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np

ARRAY_KEYS = (
    "X_full", "M_full", "X_imputed", "X_eval_imputed", "X_pi_lo", "X_pi_hi",
    "X_decoder_mu", "X_decoder_sigma",
)


def run_timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def artifact_paths(settings, market):
    if market not in {"rental", "sales"}:
        raise ValueError(f"Unknown imputation market: {market}")
    folder = settings.output_root / "02_imputation" / f"{market}_artifacts"
    stem = settings.imputed(market).stem
    return {"folder": folder, "bundle": folder / f"{stem}.pkl",
            "arrays": folder / f"{stem}.npz", "metrics": folder / "test_metrics.json",
            "weights": folder / "model_weights.pkl"}


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _median(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return _number(np.median(values)) if values.size else None


def _model_states(models):
    """CPU copies preserve the trained modules' device and support portable loading."""
    return {bucket: {
        "enc_state": {key: value.detach().cpu().clone() for key, value in model["enc"].state_dict().items()},
        "dec_state": {key: value.detach().cpu().clone() for key, value in model["dec"].state_dict().items()},
        "sc": model["sc"],
    } for bucket, model in models.items()}


def save_artifacts(settings, market, bundle, *, results, coverage, eval_vms, prod_vms,
                   context_matrix, psf_reasonable_range, started_at):
    """Export the same four artifacts for each market; never refit or resample models."""
    paths = artifact_paths(settings, market)
    paths["folder"].mkdir(parents=True, exist_ok=True)
    missing = set(ARRAY_KEYS) - set(bundle)
    if missing:
        raise ValueError(f"Missing imputation arrays: {sorted(missing)}")
    full, observed = bundle["X_full"], bundle["M_full"] == 1
    metrics = {
        "market": market, "run_timestamp": started_at, "seed": settings.seed,
        "producer_version": "vae_a4_cs_mcra_v6", "model_config": bundle["model_config"],
        "n_test": len(bundle["test_pos"]), "n_cal": len(bundle["cal_pos"]),
        "RMSE": _number(results["RMSE"]), "MAE": _number(results["MAE"]),
        "MAPE_pct": _number(results["MAPE"]), "R2": _number(results["R2"]),
        "PI90_target_pct": 90.0, "PI90_actual_pct": _number(coverage),
        "MC_PI90_actual_pct": _number(results.get("PI90")),
        "obs_psf_median": _median(full[observed]),
        "imp_psf_median": _median(bundle["X_eval_imputed"][~observed]),
        "note": "Point-error metrics use masked inference from the evaluation model. PI90_actual_pct is conformal interval coverage; MC_PI90_actual_pct is Monte Carlo draw coverage. Production estimates are not used to score held-out test cells.",
        "scaler_policy": "Inherited method: the scaler uses the full observed panel, including held-out values; neural-network training masks validation, calibration and test cells. This export does not change that methodology.",
    }
    saved_bundle = {**bundle, "schema_version": 4, "producer_version": "vae_a4_cs_mcra_v6",
                    "market": market, "seed": settings.seed, "run_timestamp": started_at,
                    "test_metrics": metrics}
    context_matrix = np.asarray(context_matrix, dtype=np.float32)
    weights = {
        "schema_version": 2, "producer_version": "vae_a4_cs_mcra_v6", "market": market,
        "run_timestamp": started_at, "seed": settings.seed, "model_config": bundle["model_config"],
        "P": int(bundle["P"]), "T": int(bundle["T"]), "B": int(bundle["B"]),
        "K_CTX": int(context_matrix.shape[1]), "context_matrix": context_matrix,
        "all_projects": bundle["all_projects"], "all_months": bundle["all_months"],
        "psf_reasonable_range": tuple(psf_reasonable_range),
        "eval_vms": _model_states(eval_vms), "prod_vms": _model_states(prod_vms),
        "q_hat_prod": float(bundle["q_hat"]),
        "note": "Encoder/decoder state dictionaries are CPU tensors. Rebuild A2Enc/A3HetDec from functions.imputation.core with P, z_dim, h, drop and K_CTX, load their states, and reuse the stored scaler. eval_vms were fitted on the training split; prod_vms were fitted on the full observed panel. The legacy q_hat_prod name refers to calibration using the evaluation model.",
    }
    # Prepare every artifact before replacing any existing artifact file. A failed
    # serialization therefore does not overwrite files from a completed run.
    with tempfile.TemporaryDirectory(prefix="._export_", dir=paths["folder"]) as temporary:
        staged = {key: Path(temporary) / paths[key].name for key in ("bundle", "arrays", "metrics", "weights")}
        joblib.dump(saved_bundle, staged["bundle"], compress=3)
        np.savez_compressed(staged["arrays"], **{key: saved_bundle[key] for key in ARRAY_KEYS})
        staged["metrics"].write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        joblib.dump(weights, staged["weights"], compress=3)
        for key, path in staged.items():
            path.replace(paths[key])
    for key in ("bundle", "arrays", "metrics", "weights"):
        print(f"Saved {paths[key]} ({paths[key].stat().st_size / 1_000_000:.1f} MB)")
    return paths
