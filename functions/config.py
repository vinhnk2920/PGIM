"""All input/output locations and run settings live here."""

from dataclasses import dataclass, field
from pathlib import Path

from functions.data_paths import REFERENCE_PATHS, TRANSACTION_FILES

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PACKAGE_ROOT / "data"


@dataclass
class Settings:
    source_root: Path = DEFAULT_SOURCE
    output_root: Path = PACKAGE_ROOT / "outputs"
    train_start: int = 200001
    test_cutoff: int = 202001
    horizons: tuple = (1, 12, 24)
    trials: int = 50
    search_profile: str = "fast"
    fast: bool = False
    cv_splits: int = 2
    validation_months: int = 12
    rolling_train_months: int = 180
    rolling_test_months: int = 12
    imputed_target_weight: float = 0.0
    selected_features: bool = False
    seed: int = 42
    imputation_max_configs: int | None = None
    imputation_max_epochs: int = 300
    mc_draws: int = 20
    reuse_existing: bool = False

    def __post_init__(self):
        self.source_root = Path(self.source_root).resolve()
        self.output_root = Path(self.output_root).resolve()
        protected = [self.source_root / name for name in ("data_prep", "imputation", "model_input")]
        if self.output_root == self.source_root or any(
            self.output_root == path or path in self.output_root.parents for path in protected
        ):
            raise ValueError("Outputs must use a separate folder, away from original inputs")
        if not self.horizons or any(h not in (1, 12, 24) for h in self.horizons):
            raise ValueError("Horizons must be chosen from 1, 12, 24")
        if not 0 <= self.imputed_target_weight <= 1:
            raise ValueError("imputed_target_weight must be between 0 and 1")
        if self.trials < 1 or self.cv_splits < 2 or self.validation_months < 1:
            raise ValueError("Invalid tuning/CV settings")
        if self.rolling_train_months < 1 or self.rolling_test_months < 1:
            raise ValueError("Rolling window sizes must be positive")
        if self.imputation_max_configs is not None and self.imputation_max_configs < 1:
            raise ValueError("imputation_max_configs must be positive")
        if self.imputation_max_epochs < 1 or self.mc_draws < 1:
            raise ValueError("Imputation epochs and draws must be positive")

    def folder(self, name):
        path = self.output_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prep_path(self, *parts):
        """A location under the organized input data_prep folder; never creates outputs."""
        return self.source_root / "data_prep" / Path(*parts)

    def transaction_input(self, market, stage="enriched"):
        """Resolve the existing source, enriched or cleaned transaction snapshot."""
        try:
            name = TRANSACTION_FILES[market][stage]
        except KeyError as exc:
            raise ValueError(f"Unknown transaction input: {market}/{stage}") from exc
        return self.prep_path("transactions", market, stage, name)

    def prepared(self, market):
        name = (
            "URA_merged_full_v3_new_macro_removed_outliers.csv"
            if market == "rental"
            else "URA_sale_removed_outliers_2.csv"
        )
        return self.output_root / "01_preprocessing" / name

    def imputed(self, market):
        prefix = "renting" if market == "rental" else "sales"
        return (
            self.output_root
            / "02_imputation"
            / f"URA_{prefix}_imputed_vae_a4_mcra_full_districts.csv"
        )

    def model_input(self, market):
        prefix = "renting" if market == "rental" else "sales"
        return (
            self.output_root
            / "03_features"
            / f"URA_{prefix}_imputed_vae_trend_policies_nearby_prices_full_districts.csv"
        )

    def bucket_ranges(self):
        return self.output_root / "02_imputation" / "rental_size_bucket_ranges_full_districts.csv"

    def read_prepared(self, market):
        path = self.prepared(market)
        if self.reuse_existing and not path.exists():
            path = self.transaction_input(market, "cleaned")
        return path

    def read_imputed(self, market):
        path = self.imputed(market)
        if self.reuse_existing and not path.exists():
            source = self.source_root / "imputation"
            canonical = source / path.name
            if canonical.exists() or market == "rental":
                return canonical
            candidates = sorted(source.glob("results_sales_full_districts_*/" + path.name))
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one sales imputation snapshot; found {len(candidates)}. Supply the chosen CSV at {path}"
                )
            return candidates[0]
        return path

    def read_model_input(self, market):
        path = self.model_input(market)
        if self.reuse_existing and not path.exists():
            path = self.source_root / "model_input" / path.name
        return path

    def read_bucket_ranges(self):
        path = self.bucket_ranges()
        if self.reuse_existing and not path.exists():
            path = self.source_root / "imputation" / path.name
        return path


@dataclass
class PrepPaths:
    """Resolve original references explicitly; generated files stay in outputs."""

    settings: Settings
    generated: set = field(default_factory=set)

    def file(self, name):
        name = Path(name.replace("\\", "/")).name
        name = {"URA_merged_full_v3.csv": "URA_merged_full_v3_new.csv"}.get(name, name)
        if name in self.generated:
            return self.settings.folder("01_preprocessing") / name
        relative = REFERENCE_PATHS.get(name)
        if relative is not None:
            path = self.settings.prep_path(relative)
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Reference file {name!r} was not found. See preflight and source_manifest.json."
        )
