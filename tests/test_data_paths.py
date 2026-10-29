"""Regression checks for moving preprocessing inputs without changing read/write roles."""

from pathlib import Path

import pytest

from functions.config import PrepPaths, Settings


@pytest.mark.parametrize("market", ["rental", "sales"])
def test_cleaned_snapshot_is_read_only_fallback_to_new_outputs(tmp_path, market):
    settings = Settings(
        source_root=tmp_path / "data", output_root=tmp_path / "out", reuse_existing=True
    )
    original = settings.transaction_input(market, "cleaned")
    original.parent.mkdir(parents=True)
    original.write_text("original input", encoding="utf-8")
    assert (
        original.parent.relative_to(settings.source_root)
        == Path("data_prep/transactions") / market / "cleaned"
    )
    assert settings.read_prepared(market) == original
    output = settings.prepared(market)
    output.parent.mkdir(parents=True)
    output.write_text("new run", encoding="utf-8")
    assert settings.read_prepared(market) == output
    assert original.read_text(encoding="utf-8") == "original input"


def test_raw_rebuild_resolves_shared_reference_and_keeps_generated_outputs_separate(tmp_path):
    settings = Settings(source_root=tmp_path / "data", output_root=tmp_path / "out")
    coordinate = settings.prep_path("location", "condo_latlon_onemap.csv")
    coordinate.parent.mkdir(parents=True)
    coordinate.write_text("coordinates", encoding="utf-8")
    source = settings.transaction_input("sales", "source")
    source.parent.mkdir(parents=True)
    source.write_text("transactions", encoding="utf-8")
    paths = PrepPaths(settings, generated={"URA_sale_data.csv"})
    assert paths.file("../URA_full/condo_latlon_onemap.csv") == coordinate
    assert paths.file("raw_realis_transaction.csv") == source
    assert (
        paths.file("URA_sale_data.csv")
        == settings.output_root / "01_preprocessing" / "URA_sale_data.csv"
    )
    assert not settings.transaction_input("sales", "enriched").exists()


def test_missing_reference_is_not_replaced_by_similarly_named_file(tmp_path):
    settings = Settings(source_root=tmp_path / "data", output_root=tmp_path / "out")
    unrelated = settings.prep_path(
        "macro",
        "quarterly",
        "Uncompleted Non Landed Private Residential Units By Locals or Foreigners.csv",
    )
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("different table", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        PrepPaths(settings).file(
            "Assessed Non Landed Private Residential Units By Locals or Foreigners.csv"
        )
