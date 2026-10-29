"""Shared VAE, scaling, MCRA evaluation and conformal functions. Per-run arrays are explicit context, never module globals."""

import calendar
import re
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def norm_name(x):
    return re.sub("[^a-z0-9]+", "", str(x).lower())


def pick_col(df, candidates, required=False):
    norm_map = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        nc = norm_name(cand)
        if nc in norm_map:
            return norm_map[nc]
    for cand in candidates:
        nc = norm_name(cand)
        for k, v in norm_map.items():
            if nc in k or k in nc:
                return v
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}")
    return None


def convert_range_to_number(range_str):
    if pd.isna(range_str) or not isinstance(range_str, str):
        return np.nan
    parts = range_str.replace(",", "").split(" to ")
    try:
        return sum([float(p) for p in parts]) / len(parts)
    except:
        return np.nan


def make_size_bucket(area_sqft, q=5):
    s_valid = area_sqft.dropna()
    out = pd.Series(index=area_sqft.index, dtype="object")
    if s_valid.nunique() < 2:
        out[:] = "SZ1"
        return out
    q_use = min(q, int(s_valid.nunique()))
    lab_use = [f"SZ{i + 1}" for i in range(q_use)]
    try:
        buckets = pd.qcut(s_valid.rank(method="first"), q=q_use, labels=lab_use, duplicates="drop")
    except Exception:
        buckets = pd.cut(
            s_valid, bins=q_use, labels=lab_use, include_lowest=True, duplicates="drop"
        )
    out.loc[s_valid.index] = buckets.astype(str)
    return out.fillna("SZ1")


def export_bucket_ranges(area_sqft, size_bucket):
    tmp = pd.DataFrame({"Area_Cleaned": area_sqft, "size_bucket": size_bucket})
    tmp = tmp.dropna(subset=["Area_Cleaned", "size_bucket"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["size_bucket", "lower_bound", "upper_bound"])
    ranges = tmp.groupby("size_bucket", as_index=False)["Area_Cleaned"].agg(
        lower_bound="min", upper_bound="max"
    )
    ranges["bucket_order"] = ranges["size_bucket"].str.extract("SZ(\\\\d+)").astype(float)
    ranges = ranges.sort_values("bucket_order").drop(columns="bucket_order").reset_index(drop=True)
    return ranges


def clean_num(series):
    if series is None:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    s0 = series.astype(str).str.strip()
    rng2 = s0.str.extract("^\\s*(-?\\d+(?:\\.\\d+)?)\\s*(?:to|-|–|—)\\s*(-?\\d+(?:\\.\\d+)?)\\s*$")
    midpoint = (
        pd.to_numeric(rng2[0], errors="coerce") + pd.to_numeric(rng2[1], errors="coerce")
    ) / 2.0
    s = s0.str.replace(",", "", regex=False).str.replace("[^0-9.\\-]+", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    out = pd.to_numeric(s, errors="coerce")
    out = out.where(midpoint.isna(), midpoint)
    return out.astype(float)


def parse_month(series):
    if series is None:
        return pd.Series(dtype="float64")
    out = clean_num(series)
    if out.notna().mean() > 0.8:
        return out
    s = series.astype(str).str.strip().str.lower()
    month_map = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    month_map.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
    out2 = s.map(month_map)
    return pd.to_numeric(out.fillna(out2), errors="coerce")


def fit_log_obs_scaler(X_data, M_obs):
    """Fit StandardScaler on log1p of all observed values."""
    x_obs = X_data[M_obs == 1]
    x_obs = x_obs[np.isfinite(x_obs)]
    sc = StandardScaler()
    sc.fit(np.log1p(np.maximum(x_obs, 0)).reshape(-1, 1))
    return sc


def log_scale_X(X_data, sc):
    X_log = np.where(np.isfinite(X_data), np.log1p(np.maximum(X_data, 0)), 0.0)
    Xs = sc.transform(X_log.reshape(-1, 1)).reshape(X_data.shape)
    Xs[~np.isfinite(Xs)] = 0.0
    return Xs


def log_unscale_X(Xsc, sc):
    return np.expm1(np.clip(sc.inverse_transform(Xsc.reshape(-1, 1)).reshape(Xsc.shape), -10, 20))


def build_context_matrix(dates_index):
    months = np.array([d.month for d in dates_index], dtype=np.float32)
    years = np.array([d.year for d in dates_index], dtype=np.float32)
    yr_n = (years - years.min()) / max(years.max() - years.min(), 1.0)
    return np.stack(
        [np.sin(2 * np.pi * months / 12), np.cos(2 * np.pi * months / 12), yr_n], axis=1
    )


def eval_positions(X_imp, pos_list, *, context):
    """RMSE on a specific set of (p,t,b) positions, using X_full as ground truth."""
    y_t = np.array([context.X_full[p, t, b] for p, t, b in pos_list], dtype=float)
    y_p = np.array([X_imp[p, t, b] for p, t, b in pos_list], dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def evaluate(X_imp, model_name, t_elapsed, samples=None, *, context):
    mask = context.M_test == 1
    y_true = context.X_full[mask].ravel()
    y_pred = X_imp[mask].ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-06))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    pi90 = None
    if samples:
        draws = np.stack([s[mask].ravel() for s in samples])
        pi90 = float(
            np.mean(
                (y_true >= np.percentile(draws, 5, axis=0))
                & (y_true <= np.percentile(draws, 95, axis=0))
            )
            * 100
        )
    pi_str = f"  PI90={pi90:.1f}%" if pi90 is not None else ""
    print(
        f"[{model_name}] RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.1f}%  R2={r2:.4f}{pi_str}  ({t_elapsed:.1f}s)"
    )
    return dict(Model=model_name, RMSE=rmse, MAE=mae, MAPE=mape, R2=r2, PI90=pi90)


class A2Enc(nn.Module):
    def __init__(self, p, z=8, h=128, drop=0.1, k_ctx=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(p * 2 + k_ctx, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop),
        )
        self.mu_fc = nn.Linear(h, z)
        self.lv_fc = nn.Linear(h, z)

    def forward(self, x, m, c):
        h = self.shared(torch.cat([x * m, m, c], dim=1))
        return (self.mu_fc(h), torch.clamp(self.lv_fc(h), -10, 10))


class A3HetDec(nn.Module):
    def __init__(self, p, z=8, h=128, drop=0.1, k_ctx=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(z + k_ctx, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop),
        )
        self.fc_mu = nn.Linear(h, p)
        self.fc_lv = nn.Linear(h, p)

    def forward(self, z, c):
        h = self.shared(torch.cat([z, c], dim=1))
        return (self.fc_mu(h), torch.clamp(self.fc_lv(h), -6, 4))


def train_vae_a4(
    z_dim,
    beta,
    epochs,
    X_scale,
    M_scale,
    X_tr,
    M_tr,
    C,
    lr=0.001,
    batch=32,
    h=128,
    drop=0.1,
    warmup_epochs=50,
    free_bits=0.5,
    device=None,
    *,
    context,
):
    """
    Fix A: scaler fitted on X_scale/M_scale (always X_full/M_full).
           model trained on X_tr/M_tr (train-only split, val_es+cal+test masked out).
    Fix G: KL free-bits applied per-dimension with .sum(dim=1).mean()/z_dim
           so each latent dimension has its own minimum information budget.
    """
    device = context.DEVICE if device is None else device
    sc = fit_log_obs_scaler(X_scale, M_scale)
    Xsc_tr = log_scale_X(X_tr, sc)
    Ct = torch.FloatTensor(C)
    vms = {}
    for b_idx in range(context.B):
        xs = torch.FloatTensor(Xsc_tr[:, :, b_idx].T)
        ms = torch.FloatTensor(M_tr[:, :, b_idx].T)
        loader = DataLoader(
            TensorDataset(xs, ms, Ct), batch_size=batch, shuffle=True, drop_last=False
        )
        enc = A2Enc(context.P, z=z_dim, h=h, drop=drop).to(device)
        dec = A3HetDec(context.P, z=z_dim, h=h, drop=drop).to(device)
        params = list(enc.parameters()) + list(dec.parameters())
        opt = optim.Adam(params, lr=lr, weight_decay=1e-05)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
        for epoch in range(1, epochs + 1):
            enc.train()
            dec.train()
            beta_t = beta * min(1.0, epoch / max(1, warmup_epochs))
            for xb, mb, cb in loader:
                xb, mb, cb = (xb.to(device), mb.to(device), cb.to(device))
                mu_z, lv_z = enc(xb, mb, cb)
                z = mu_z + torch.exp(0.5 * lv_z) * torch.randn_like(mu_z)
                mu_x, lv_x = dec(z, cb)
                nll = 0.5 * (lv_x + (xb - mu_x).pow(2) / (lv_x.exp() + 1e-08))
                recon = (nll * mb).sum() / (mb.sum() + 1e-08)
                kl = (
                    torch.clamp(-0.5 * (1 + lv_z - mu_z.pow(2) - lv_z.exp()), min=free_bits)
                    .sum(dim=1)
                    .mean()
                    / z_dim
                )
                loss = recon + beta_t * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sch.step()
        vms[b_idx] = {"enc": enc, "dec": dec, "sc": sc}
    return vms


def infer_vae_a4(vms, X_data, M_obs, C, n_samples=1, device=None, *, context):
    """
    Fix D: mu_store and sig_store are accumulated across ALL samples and
           averaged at the end — not captured from sample 0 only.
    """
    device = context.DEVICE if device is None else device
    runs = []
    mu_accum = np.zeros_like(X_data, dtype=float)
    sig_accum = np.zeros_like(X_data, dtype=float)
    Ct = torch.FloatTensor(C)
    for s_idx in range(n_samples):
        Xout = np.full_like(X_data, np.nan, dtype=float)
        mu_out = np.full_like(X_data, np.nan, dtype=float)
        sig_out = np.full_like(X_data, np.nan, dtype=float)
        for b_idx in range(context.B):
            enc, dec, sc = (vms[b_idx]["enc"], vms[b_idx]["dec"], vms[b_idx]["sc"])
            enc.eval()
            dec.eval()
            Xsc = log_scale_X(X_data, sc)
            xs = torch.FloatTensor(Xsc[:, :, b_idx].T).to(device)
            ms = torch.FloatTensor(M_obs[:, :, b_idx].T).to(device)
            cb = Ct.to(device)
            with torch.no_grad():
                mu_z, lv_z = enc(xs, ms, cb)
                z = mu_z + torch.exp(0.5 * lv_z) * torch.randn_like(mu_z)
                mu_x, lv_x = dec(z, cb)
                sigma_x = torch.exp(0.5 * lv_x)
                pr_scaled = mu_x + sigma_x * torch.randn_like(mu_x)
            pr_orig = log_unscale_X(pr_scaled.cpu().numpy().T, sc)
            mu_orig = log_unscale_X(mu_x.cpu().numpy().T, sc)
            sig_orig = sigma_x.cpu().numpy().T
            Xout[:, :, b_idx] = np.where(M_obs[:, :, b_idx] == 1, X_data[:, :, b_idx], pr_orig)
            mu_out[:, :, b_idx] = np.where(M_obs[:, :, b_idx] == 1, X_data[:, :, b_idx], mu_orig)
            sig_out[:, :, b_idx] = sig_orig
        runs.append(Xout)
        mu_accum += mu_out
        sig_accum += sig_out
    mu_store = mu_accum / n_samples
    sig_store = sig_accum / n_samples
    return (runs, mu_store, sig_store)


def val_rmse_a4(
    z_dim,
    beta,
    lr,
    C,
    h=128,
    drop=0.1,
    max_epochs=300,
    eval_every=25,
    patience=4,
    n_eval_draws=3,
    device=None,
    *,
    context,
):
    """
    Fix A: grid search trains only on X_train/M_train.
    Fix C: early stopping uses val_es_pos (disjoint from cal_pos and test_pos).
    Fix I: continuous training — networks are initialised once and trained
           end-to-end. Weights are checkpointed at each eval_every interval
           so we never restart from epoch 0. Cost is O(max_epochs) not
           O(max_epochs^2 / eval_every).
    """
    device = context.DEVICE if device is None else device
    sc = fit_log_obs_scaler(context.X_full, context.M_full)
    Xsc_tr = log_scale_X(context.X_train, sc)
    Ct = torch.FloatTensor(C)
    encs, decs, opts, schs = ({}, {}, {}, {})
    for b_idx in range(context.B):
        enc = A2Enc(context.P, z=z_dim, h=h, drop=drop).to(device)
        dec = A3HetDec(context.P, z=z_dim, h=h, drop=drop).to(device)
        params = list(enc.parameters()) + list(dec.parameters())
        opt = optim.Adam(params, lr=lr, weight_decay=1e-05)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
        encs[b_idx] = enc
        decs[b_idx] = dec
        opts[b_idx] = opt
        schs[b_idx] = sch
    loaders = {}
    for b_idx in range(context.B):
        xs = torch.FloatTensor(Xsc_tr[:, :, b_idx].T)
        ms = torch.FloatTensor(context.M_train[:, :, b_idx].T)
        loaders[b_idx] = DataLoader(
            TensorDataset(xs, ms, Ct), batch_size=32, shuffle=True, drop_last=False
        )
    best_vms, best_rv, best_e, bad = (None, np.inf, 0, 0)
    epoch = 0
    while epoch < max_epochs:
        for _ in range(eval_every):
            epoch += 1
            if epoch > max_epochs:
                break
            beta_t = beta * min(1.0, epoch / max(1, 50))
            for b_idx in range(context.B):
                encs[b_idx].train()
                decs[b_idx].train()
                for xb, mb, cb in loaders[b_idx]:
                    xb, mb, cb = (xb.to(device), mb.to(device), cb.to(device))
                    mu_z, lv_z = encs[b_idx](xb, mb, cb)
                    z = mu_z + torch.exp(0.5 * lv_z) * torch.randn_like(mu_z)
                    mu_x, lv_x = decs[b_idx](z, cb)
                    nll = 0.5 * (lv_x + (xb - mu_x).pow(2) / (lv_x.exp() + 1e-08))
                    recon = (nll * mb).sum() / (mb.sum() + 1e-08)
                    kl = (
                        torch.clamp(-0.5 * (1 + lv_z - mu_z.pow(2) - lv_z.exp()), min=0.5)
                        .sum(dim=1)
                        .mean()
                        / z_dim
                    )
                    loss = recon + beta_t * kl
                    opts[b_idx].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(encs[b_idx].parameters()) + list(decs[b_idx].parameters()), 1.0
                    )
                    opts[b_idx].step()
                    schs[b_idx].step()
        import copy

        vms_ckpt = {
            b_idx: {"enc": copy.deepcopy(encs[b_idx]), "dec": copy.deepcopy(decs[b_idx]), "sc": sc}
            for b_idx in range(context.B)
        }
        draws, _, _ = infer_vae_a4(
            vms_ckpt,
            context.X_train,
            context.M_train,
            C,
            n_samples=n_eval_draws,
            device=device,
            context=context,
        )
        X_mean = np.mean(np.stack(draws), axis=0)
        rv = eval_positions(X_mean, context.val_es_pos, context=context)
        if rv + 1e-06 < best_rv:
            best_rv, best_vms, best_e, bad = (rv, vms_ckpt, epoch, 0)
        else:
            bad += 1
            if bad >= patience:
                break
    return (best_vms, best_rv, best_e)


def conformal_calibrate_v6(
    vms, X_train, M_train, C, cal_positions, alpha=0.1, device=None, *, context
):
    """
    Split-conformal calibration — final clean implementation.

    Design (follows ablation notebook pattern, extended to B dims):
      Encoder context : X_train / M_train
                        Only training observations visible; cal/val_es/test all masked.
                        This matches the distribution the encoder was trained on exactly.
      Scores          : normalised residuals in log1p(PSF) space
                          s_i = |log1p(y_true) - log1p(mu_pred)| / (sig_pred + 1e-6)
                        y_true  = X_full[cal_pos]  (ground truth, never NaN)
                        mu_pred = decoder mean unscaled back to PSF, then log1p
                        sig_pred = decoder sigma in log-scaled space (relative proxy)
      Deterministic   : z = mu_z (no sampling) for stable, reproducible scores
      q_hat           : finite-sample-corrected (1-alpha) quantile of scores

    Split guarantee: cal_pos ∩ train_pos = ∅  (enforced in Section 2)
                     cal_pos ∩ val_es_pos = ∅  (idem)
                     cal_pos ∩ test_pos   = ∅  (idem)
    """
    device = context.DEVICE if device is None else device
    Ct = torch.FloatTensor(C)
    all_scores = []
    mu_psf_B = np.zeros((context.P, context.T, context.B), dtype=float)
    sig_sc_B = np.zeros((context.P, context.T, context.B), dtype=float)
    for b_idx in range(context.B):
        sc = vms[b_idx]["sc"]
        enc = vms[b_idx]["enc"]
        dec = vms[b_idx]["dec"]
        enc.eval()
        dec.eval()
        Xsc = log_scale_X(context.X_train, sc)
        xs = torch.FloatTensor(Xsc[:, :, b_idx].T).to(device)
        ms = torch.FloatTensor(context.M_train[:, :, b_idx].T).to(device)
        with torch.no_grad():
            mu_z, _ = enc(xs, ms, Ct.to(device))
            mu_x, lv_x = dec(mu_z, Ct.to(device))
            sig_x = torch.exp(0.5 * lv_x)
        mu_psf = log_unscale_X(mu_x.cpu().numpy().T, sc)
        sig_sc = sig_x.cpu().numpy().T
        mu_psf_B[:, :, b_idx] = mu_psf
        sig_sc_B[:, :, b_idx] = sig_sc
        for p, t, b in cal_positions:
            if b != b_idx:
                continue
            y_true = float(context.X_full[p, t, b])
            mu_hat = float(mu_psf[p, t])
            sig_hat = float(sig_sc[p, t]) + 1e-06
            log_y = np.log1p(max(y_true, 0))
            log_mu = np.log1p(max(mu_hat, 0))
            all_scores.append(abs(log_y - log_mu) / sig_hat)
    scores = np.array(all_scores)
    n_cal = len(scores)
    q_idx = min(int(np.ceil((n_cal + 1) * (1 - alpha))) - 1, n_cal - 1)
    q_hat = float(np.sort(scores)[q_idx])
    print("Split-conformal calibration:")
    print(f"  n_cal={n_cal}  alpha={alpha}  q_hat={q_hat:.4f}")
    print(
        f"  score 50th={np.median(scores):.3f}  90th={np.percentile(scores, 90):.3f}  99th={np.percentile(scores, 99):.3f}"
    )
    log_mu_B = np.log1p(np.maximum(mu_psf_B, 0))
    pi_lo_log = log_mu_B - q_hat * sig_sc_B
    pi_hi_log = log_mu_B + q_hat * sig_sc_B
    X_pi_lo = np.expm1(np.maximum(pi_lo_log, 0))
    X_pi_hi = np.expm1(pi_hi_log)
    X_pi_lo = np.clip(X_pi_lo, context.PSF_REASONABLE_RANGE[0], context.PSF_REASONABLE_RANGE[1])
    X_pi_hi = np.clip(X_pi_hi, context.PSF_REASONABLE_RANGE[0], context.PSF_REASONABLE_RANGE[1])
    return (q_hat, mu_psf_B, sig_sc_B, X_pi_lo, X_pi_hi)
