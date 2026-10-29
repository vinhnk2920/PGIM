"""Both markets must export matching artifacts with reloadable VAE weights."""

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from functions.config import Settings
from functions.imputation.artifacts import ARRAY_KEYS, artifact_paths, save_artifacts


@pytest.mark.parametrize("market", ["rental", "sales"])
def test_saved_weights_reproduce_inference_and_artifacts_keep_evaluation_separate(tmp_path, market):
    import torch
    from functions.imputation.core import A2Enc, A3HetDec, build_context_matrix, train_vae_a4, infer_vae_a4

    torch.set_num_threads(1)
    torch.manual_seed(42)
    full = np.random.default_rng(42).uniform(2, 6, (4, 8, 1)).astype("float32")
    mask = np.ones_like(full)
    mask[0, 0, 0] = 0
    x = full.copy()
    x[mask == 0] = np.nan
    context = SimpleNamespace(P=4, T=8, B=1, DEVICE=torch.device("cpu"))
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    c = build_context_matrix(dates)
    models = train_vae_a4(2, 0.01, 1, x, mask, x, mask, c, h=8, batch=4, context=context)
    expected = np.nan_to_num(x, nan=3.0)
    bundle = {"model_config": {"z_dim": 2, "h": 8, "drop": 0.1}, "q_hat": 1.5,
              "P": 4, "T": 8, "B": 1, "X_full": x, "M_full": mask.astype("uint8"),
              "X_imputed": expected, "X_eval_imputed": expected + 0.5,
              "X_pi_lo": expected - 1, "X_pi_hi": expected + 1,
              "X_decoder_mu": expected, "X_decoder_sigma": np.ones_like(full),
              "test_pos": [(1, 1, 0)], "cal_pos": [(2, 2, 0)], "train_pos": [(3, 3, 0)],
              "val_es_pos": [(1, 2, 0)], "all_projects": ["A", "B", "C", "D"],
              "all_months": dates.strftime("%Y-%m").tolist(), "series_meta": pd.DataFrame({"project_idx": range(4)})}
    settings = Settings(output_root=tmp_path)
    results = {"RMSE": 1.25, "MAE": 1.0, "MAPE": 2.0, "R2": float("nan"), "PI90": 80.0}
    paths = save_artifacts(settings, market, bundle, results=results, coverage=90.0,
                           eval_vms=models, prod_vms=models, context_matrix=c,
                           psf_reasonable_range=(0.5, 40), started_at="test-run")
    assert paths["folder"].name == f"{market}_artifacts"
    assert set(path.name for path in paths["folder"].iterdir()) == {
        paths[key].name for key in ("bundle", "arrays", "metrics", "weights")}
    metrics = json.loads(paths["metrics"].read_text())
    assert metrics["RMSE"] == 1.25 and metrics["PI90_actual_pct"] == 90
    assert metrics["MC_PI90_actual_pct"] == 80 and metrics["R2"] is None
    saved = joblib.load(paths["bundle"])
    with np.load(paths["arrays"]) as arrays:
        assert set(arrays.files) == set(ARRAY_KEYS)
        for key in ARRAY_KEYS:
            np.testing.assert_array_equal(arrays[key], bundle[key])
            np.testing.assert_array_equal(saved[key], bundle[key])
    assert saved["test_metrics"] == metrics
    weights = joblib.load(paths["weights"])
    assert weights["market"] == market and weights["run_timestamp"] == "test-run"
    restored = {}
    for key, state in weights["prod_vms"].items():
        encoder, decoder = A2Enc(4, z=2, h=8, drop=0.1, k_ctx=3), A3HetDec(4, z=2, h=8, drop=0.1, k_ctx=3)
        encoder.load_state_dict(state["enc_state"])
        decoder.load_state_dict(state["dec_state"])
        restored[key] = {"enc": encoder, "dec": decoder, "sc": state["sc"]}
        assert all(tensor.device.type == "cpu" for tensor in state["enc_state"].values())
    torch.manual_seed(123)
    old = infer_vae_a4(models, x, mask, c, n_samples=2, context=context)
    torch.manual_seed(123)
    new = infer_vae_a4(restored, x, mask, weights["context_matrix"], n_samples=2, context=context)
    for left, right in [*zip(old[0], new[0]), (old[1], new[1]), (old[2], new[2])]:
        np.testing.assert_array_equal(left, right)


def test_failed_export_preserves_completed_artifacts(tmp_path, monkeypatch):
    from functions.imputation import artifacts

    settings = Settings(output_root=tmp_path)
    paths = artifact_paths(settings, "rental")
    paths["folder"].mkdir(parents=True)
    for key in ("bundle", "arrays", "metrics", "weights"):
        paths[key].write_bytes(b"previous completed run")
    bundle = {key: np.ones((1, 1, 1)) for key in ARRAY_KEYS}
    bundle.update(model_config={}, P=1, T=1, B=1, test_pos=[], cal_pos=[], q_hat=1,
                  all_projects=["A"], all_months=["2020-01"])
    original_dump = artifacts.joblib.dump

    def fail_weights(obj, file, **kwargs):
        if file.name == "model_weights.pkl":
            raise OSError("simulated disk failure")
        return original_dump(obj, file, **kwargs)

    monkeypatch.setattr(artifacts.joblib, "dump", fail_weights)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_artifacts(settings, "rental", bundle,
                       results={"RMSE": 1, "MAE": 1, "MAPE": 1, "R2": 1}, coverage=90,
                       eval_vms={}, prod_vms={}, context_matrix=np.ones((1, 3)),
                       psf_reasonable_range=(0.5, 40), started_at="failed-run")
    assert all(paths[key].read_bytes() == b"previous completed run"
               for key in ("bundle", "arrays", "metrics", "weights"))
    assert not list(paths["folder"].glob("._export_*"))
