# Preprocessing inputs

The directory groups inputs by subject. GitHub contains the small reference tables listed in [the data guide](../README.md); the complete large transaction and property datasets will be uploaded separately, with previews available under [data/snapshots/data_prep/](../snapshots/data_prep/).

```text
data_prep/
  transactions/
    rental/{source,enriched,cleaned}/
    sales/{source,enriched,cleaned}/
  property/
  location/
  macro/{monthly,quarterly}/
  supply/
    developer_sales.csv
    project_details.csv
  labour/
  consulting/
  trend_policy/
```

## Transaction stages

| Market | Stage | Filename | Role |
|---|---|---|---|
| Rental | source | `rental_merged_tenure_1999-2026.csv` | Optional raw-enrichment input; includes upstream tenure information. |
| Rental | enriched | `URA_merged_full_v3_new.csv` | Default preprocessing input. |
| Rental | cleaned | `URA_merged_full_v3_new_macro_removed_outliers.csv` | Existing imputation fallback with `--reuse-existing`. |
| Sales | source | `raw_realis_transaction.csv` | Optional raw-enrichment input. |
| Sales | enriched | `URA_sale_data.csv` | Default preprocessing input. |
| Sales | cleaned | `URA_sale_removed_outliers_2.csv` | Existing imputation fallback with `--reuse-existing`. |

Enrichment precedes imputation. Default step 1 uses enriched snapshots; `--rebuild-raw` requires the source transactions and all relevant reference tables. Additional rental rebuild references are `transactions/rental/enriched/rental_merged_tenure_1999-2026_enriched_macro.csv` and `URA_merged_full_v1.csv`.

## Reference groups

- `property/`: property/facilities attributes. The included `property_details_facilities.csv` was recovered from an original checkpoint; its recency relative to the research run is not independently established. The complete larger 99.co-enriched reference will be uploaded separately.
- `location/`: shared coordinates, walking distances, school and amenity references, plus the supplementary sales MRT table.
- `macro/monthly/`: shared monthly macro series.
- `macro/quarterly/`: quarterly URA tables, including ownership counts. The ownership table's `-` markers for 2006Q2 and 2006Q3 remain missing; they are not filled in the source.
- `supply/`: the two complete consolidated inputs used by normal feature runs, with per-row source provenance.
- `labour/`: five MOM CSV/Excel references.
- `consulting/`: the manually compiled consulting forecast table.
- `trend_policy/`: complete external-feature tables for rental and sales.

## Supply reference data

`developer_sales.csv` contains 47,334 rows consolidated from 4,553 monthly downloads. `project_details.csv` contains 32,844 rows from 1,978 quarterly downloads. These counts describe the validated October 2026 snapshots. Normal feature construction reads only these two files.

Each row preserves `source_file` and `source_row`; filename-derived district and period fields are retained. Consolidation follows sorted source filenames and retains row order before the feature stage applies its existing deduplication rules.

To rebuild the tables from separately held original downloads:

```bash
python consolidate_supply.py --developer-sales-dir "/path/to/developer_sales_downloads" --project-details-dir "/path/to/project_details_downloads"
```

The script validates serialized values and row order, then writes checksums under `validation/supply/`. Original downloads and generated audit reports are not included on GitHub.

## Data flow

Enriched transactions -> preprocessing -> `outputs/01_preprocessing/` -> imputation -> `outputs/02_imputation/` -> feature construction -> `outputs/03_features/` -> forecasting.

See the [main README](../../README.md#run-the-pipeline) for the commands. Large-data previews are separate from this input tree and are never selected automatically by the pipeline.
