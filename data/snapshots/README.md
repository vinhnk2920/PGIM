# Large-data previews

Each CSV in this directory contains the original header and the first five data records from one larger dataset whose full version will be uploaded separately. All columns, blank fields and parsed text values are preserved; no numeric conversion or random sampling was applied. These rows show the file structure and example values only. They are not representative samples, training datasets or complete forecasting inputs.

The subdirectories mirror the original paths under `data/`. For example, `data/snapshots/model_input/<filename>.csv` previews the full file expected at `data/model_input/<filename>.csv`. The pipeline never reads `data/snapshots/` by default. Do not copy these previews into the full-data input paths.

| Full-data path relative to `data/` (link opens preview) | Sample rows | Columns | Full file size (MiB) |
|---|---:|---:|---:|
| [data_prep/property/URA_enriched_with_99co_v2.csv](data_prep/property/URA_enriched_with_99co_v2.csv) | 5 | 286 | 1,127.55 |
| [data_prep/transactions/rental/cleaned/URA_merged_full_v3_new_macro_removed_outliers.csv](data_prep/transactions/rental/cleaned/URA_merged_full_v3_new_macro_removed_outliers.csv) | 5 | 393 | 2,967.01 |
| [data_prep/transactions/rental/enriched/rental_merged_tenure_1999-2026_enriched_macro.csv](data_prep/transactions/rental/enriched/rental_merged_tenure_1999-2026_enriched_macro.csv) | 5 | 29 | 394.13 |
| [data_prep/transactions/rental/enriched/URA_merged_full_v1.csv](data_prep/transactions/rental/enriched/URA_merged_full_v1.csv) | 5 | 367 | 2,783.18 |
| [data_prep/transactions/rental/enriched/URA_merged_full_v3_new.csv](data_prep/transactions/rental/enriched/URA_merged_full_v3_new.csv) | 5 | 367 | 2,784.32 |
| [data_prep/transactions/rental/source/rental_merged_tenure_1999-2026.csv](data_prep/transactions/rental/source/rental_merged_tenure_1999-2026.csv) | 5 | 17 | 238.74 |
| [data_prep/transactions/sales/cleaned/URA_sale_removed_outliers_2.csv](data_prep/transactions/sales/cleaned/URA_sale_removed_outliers_2.csv) | 5 | 424 | 1,255.03 |
| [data_prep/transactions/sales/enriched/URA_sale_data.csv](data_prep/transactions/sales/enriched/URA_sale_data.csv) | 5 | 413 | 1,201.38 |
| [data_prep/transactions/sales/source/raw_realis_transaction.csv](data_prep/transactions/sales/source/raw_realis_transaction.csv) | 5 | 22 | 129.43 |
| [imputation/URA_renting_imputed_vae_a4_mcra_full_districts.csv](imputation/URA_renting_imputed_vae_a4_mcra_full_districts.csv) | 5 | 174 | 2,676.90 |
| [imputation/URA_sales_imputed_vae_a4_mcra_full_districts.csv](imputation/URA_sales_imputed_vae_a4_mcra_full_districts.csv) | 5 | 175 | 2,786.16 |
| [model_input/URA_renting_imputed_vae_trend_policies_nearby_prices_full_districts.csv](model_input/URA_renting_imputed_vae_trend_policies_nearby_prices_full_districts.csv) | 5 | 221 | 3,401.67 |
| [model_input/URA_sales_imputed_vae_trend_policies_nearby_prices_full_districts.csv](model_input/URA_sales_imputed_vae_trend_policies_nearby_prices_full_districts.csv) | 5 | 225 | 3,530.09 |

`manifest.json` records source sizes, sample rules and preview checksums. The full datasets will be uploaded separately. Download links and access instructions will be added to [the data guide](../README.md#full-data-availability) when they are available.
