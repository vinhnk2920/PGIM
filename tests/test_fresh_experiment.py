import json
import hashlib
import numpy as np
import pandas as pd
import pytest

from functions.experiment import rental_buckets, bucket_ranges, sales_buckets, record_panel
from functions.input_audit import FreshInputPolicy
from functions.config import Settings


def test_reported_quantiles_preserve_rank_first_ties():
    area = pd.Series([500.] * 10 + [1000.] * 10 + [1500.] * 10 + [2000.] * 10 + [2500.] * 10)
    result = rental_buckets(area)
    expected = pd.qcut(area.rank(method="first"), q=5, labels=["SZ1","SZ2","SZ3","SZ4","SZ5"]).astype(str)
    assert result.equals(expected.astype(object))
    ranges = bucket_ranges(area,result)
    assert ranges.upper_bound.tolist() == [500.,1000.,1500.,2000.,2500.]
    assert sales_buckets(pd.Series([500.,501.,2501.]), ranges).tolist() == ["SZ1","SZ2","SZ5"]


def test_input_policy_allows_raw_and_new_outputs_but_blocks_checkpoints(tmp_path):
    source=tmp_path/"project/data";output=tmp_path/"new_run"
    policy=FreshInputPolicy(source,output)
    policy.check(source/"data_prep/transactions/sales/source/raw.csv")
    policy.check(output/"02_imputation/fresh.csv")
    for path in [source/"imputation/old.csv",source/"model_input/old.csv",source/"snapshots/old.csv",
                 source.parent/"outputs/old.pkl",source.parent.parent/"rolling_window_models/old.pkl",
                 source.parent.parent/"imputation_v2/old.pkl"]:
        with pytest.raises(PermissionError):policy.check(path)


def test_experiment_cannot_use_reference_or_reuse_existing(tmp_path):
    with pytest.raises(ValueError,match="cannot reuse"):
        Settings(experiment_profile=tmp_path/"config.json",reuse_existing=True)


def test_current_bucket_policy_remains_separate():
    from functions.imputation.core import make_size_bucket
    area=pd.Series([650.,651.,950.,951.])
    assert make_size_bucket(area).tolist() == ["SZ1","SZ2","SZ2","SZ3"]


def test_fresh_preflight_requires_raw_inputs_even_if_enriched_cache_exists(tmp_path):
    from functions.cli import preflight
    config=tmp_path/"profile.json"
    config.write_text(json.dumps({"methodology":"reported_aug2026"}))
    settings=Settings(source_root=tmp_path/"data",output_root=tmp_path/"out",experiment_profile=config)
    cached=settings.transaction_input("rental","enriched")
    cached.parent.mkdir(parents=True)
    cached.write_text("cached_column\n1\n")
    with pytest.raises(FileNotFoundError,match="Required from-raw inputs"):
        preflight(settings)
    report=json.loads((settings.output_root/"logs/preflight.json").read_text())
    checks={row["path"]:row["exists"] for row in report["references"]}
    assert not checks[str(settings.transaction_input("rental","source"))]
    assert str(cached) not in checks


def test_observed_panel_mismatch_stops_before_training(tmp_path):
    full=np.array([[[1.],[np.nan]],[[2.],[3.]]],dtype=np.float32)
    mask=np.isfinite(full).astype(np.float32)
    config=tmp_path/"profile.json"
    config.write_text(json.dumps({"methodology":"reported_aug2026", "expected_panels":{
        "rental":{"P":2,"T":2,"observed_cells":3,
                  "observed_price_sha256":hashlib.sha256(full[mask.astype(bool)].tobytes()).hexdigest()}}}))
    settings=Settings(experiment_profile=config,output_root=tmp_path/"out")
    record_panel(settings,"rental",full,mask,["A","B"],["2000-01","2000-02"])
    changed=full.copy();changed[0,0,0]=99.
    with pytest.raises(ValueError,match="differs from reported panel"):
        record_panel(settings,"rental",changed,mask,["A","B"],["2000-01","2000-02"])
    with pytest.raises(ValueError,match="differs from reported panel"):
        record_panel(settings,"rental",full,np.zeros_like(mask),["A","B"],["2000-01","2000-02"])
