# Data guide

The repository includes 26 complete small reference files (each below 100 MiB), plus five-row previews of the 13 larger CSVs. Full transaction, imputation and model-input datasets will be uploaded separately. The previews are for understanding the schema and example values only; a fresh clone cannot run the complete pipeline from the previews.

## Included complete reference files

Files in the following table are copied without changing their contents. Their checksums were verified against the original reference files.

| Path relative to `data/` | Size (MiB) |
|---|---:|
| [data_prep/consulting/consulting_prediction_manual_data.csv](<data_prep/consulting/consulting_prediction_manual_data.csv>) | 0.031 |
| [data_prep/labour/median_gross_monthly_income_full_time_employed_residents_total.csv](<data_prep/labour/median_gross_monthly_income_full_time_employed_residents_total.csv>) | 0.000 |
| [data_prep/labour/rcr_01_total_qtr.csv](<data_prep/labour/rcr_01_total_qtr.csv>) | 0.001 |
| [data_prep/labour/ret_01_total_qtr.csv](<data_prep/labour/ret_01_total_qtr.csv>) | 0.002 |
| [data_prep/labour/rgr_01_total_qtr.xlsx](<data_prep/labour/rgr_01_total_qtr.xlsx>) | 0.012 |
| [data_prep/labour/share_nonresident_population_by_pass_types_annual.csv](<data_prep/labour/share_nonresident_population_by_pass_types_annual.csv>) | 0.001 |
| [data_prep/location/condo_latlon_onemap.csv](<data_prep/location/condo_latlon_onemap.csv>) | 0.677 |
| [data_prep/location/full_period_condo_mrt_candidates_walk.csv](<data_prep/location/full_period_condo_mrt_candidates_walk.csv>) | 1.329 |
| [data_prep/location/nearby_5km_amenities.csv](<data_prep/location/nearby_5km_amenities.csv>) | 8.450 |
| [data_prep/location/sale_missing_project_mrt_candidates_walk.csv](<data_prep/location/sale_missing_project_mrt_candidates_walk.csv>) | 0.088 |
| [data_prep/location/top30_primary_schools.csv](<data_prep/location/top30_primary_schools.csv>) | 0.002 |
| [data_prep/macro/monthly/sg_macro_monthly.csv](<data_prep/macro/monthly/sg_macro_monthly.csv>) | 0.177 |
| [data_prep/macro/quarterly/Assessed Non Landed Private Residential Units By Locals or Foreigners.csv](<data_prep/macro/quarterly/Assessed Non Landed Private Residential Units By Locals or Foreigners.csv>) | 0.005 |
| [data_prep/macro/quarterly/Available and Vacant Non-Landed Residential.csv](<data_prep/macro/quarterly/Available and Vacant Non-Landed Residential.csv>) | 0.005 |
| [data_prep/macro/quarterly/Price Index of Non-Landed Residential Properties.csv](<data_prep/macro/quarterly/Price Index of Non-Landed Residential Properties.csv>) | 0.007 |
| [data_prep/macro/quarterly/Rental Index of Non-Landed Properties.csv](<data_prep/macro/quarterly/Rental Index of Non-Landed Properties.csv>) | 0.004 |
| [data_prep/macro/quarterly/Supply of Non-Landed Private Residential.csv](<data_prep/macro/quarterly/Supply of Non-Landed Private Residential.csv>) | 0.007 |
| [data_prep/macro/quarterly/Uncompleted Non Landed Private Residential Units By Locals or Foreigners.csv](<data_prep/macro/quarterly/Uncompleted Non Landed Private Residential Units By Locals or Foreigners.csv>) | 0.004 |
| [data_prep/macro/quarterly/Unit Sold or Unsold.csv](<data_prep/macro/quarterly/Unit Sold or Unsold.csv>) | 0.004 |
| [data_prep/macro/quarterly/Units Launched or Not Yet Launched.csv](<data_prep/macro/quarterly/Units Launched or Not Yet Launched.csv>) | 0.004 |
| [data_prep/property/property_details_facilities.csv](<data_prep/property/property_details_facilities.csv>) | 0.683 |
| [data_prep/supply/developer_sales.csv](<data_prep/supply/developer_sales.csv>) | 7.628 |
| [data_prep/supply/project_details.csv](<data_prep/supply/project_details.csv>) | 5.230 |
| [data_prep/trend_policy/condo_rent_external_features_consolidated.csv](<data_prep/trend_policy/condo_rent_external_features_consolidated.csv>) | 0.017 |
| [data_prep/trend_policy/condo_sales_external_features_consolidated.csv](<data_prep/trend_policy/condo_sales_external_features_consolidated.csv>) | 0.023 |
| [imputation/rental_size_bucket_ranges_full_districts.csv](<imputation/rental_size_bucket_ranges_full_districts.csv>) | 0.000 |

The small `imputation/rental_size_bucket_ranges_full_districts.csv` is a reference snapshot. When running sales imputation on its own, the code expects the matching bucket file in `outputs/02_imputation/`; running rental imputation creates it there.

## Large-file previews

See [snapshots/README.md](snapshots/README.md) for all 13 preview files and full-file sizes. Each preview has five data rows and the original header. The corresponding full files are deliberately excluded from Git.

## Full-data availability

The full datasets will be uploaded separately. Download links and access instructions will be added here once they are available.

For now, this repository contains the complete small reference files and five-row previews of the larger datasets. The previews illustrate the schema and example values; they cannot replace the complete files when running the pipeline.

Once the full datasets are available, place them under `data/` using the relative paths listed in [the preview manifest](snapshots/manifest.json), then follow the commands in the [main README](../README.md#run-the-pipeline).

## Git tracking

Only the reviewed small references, guides and previews are allowed by `.gitignore`. Other files under `data/`, including full transaction and model-input CSVs, remain ignored. Outputs, weights, validation reports and checkpoints remain excluded.
