"""Tiny VAE test, shared by rental and sales; no source data needed."""

from types import SimpleNamespace

import numpy as np


def test_vae_preserves_observed_values_and_all_array_shapes():
    import torch
    from functions.imputation.core import build_context_matrix, train_vae_a4, infer_vae_a4
    import pandas as pd

    torch.set_num_threads(1)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    x = rng.uniform(2, 6, (4, 8, 1)).astype("float32")
    mask = np.ones_like(x)
    mask[0, 0, 0] = 0
    x[0, 0, 0] = np.nan
    context = SimpleNamespace(P=4, T=8, B=1, DEVICE=torch.device("cpu"))
    c = build_context_matrix(pd.date_range("2020-01-01", periods=8, freq="MS"))
    models = train_vae_a4(2, 0.01, 2, x, mask, x, mask, c, h=8, batch=4, context=context)
    runs, mu, sigma = infer_vae_a4(models, x, mask, c, n_samples=2, context=context)
    assert mu.shape == sigma.shape == x.shape
    assert np.isfinite(mu).all() and np.isfinite(sigma).all()
    for draw in runs:
        np.testing.assert_array_equal(draw[mask == 1], x[mask == 1])
        assert np.isfinite(draw).all()
