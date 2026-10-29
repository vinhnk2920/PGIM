"""Explicit file locations for organized preprocessing inputs; no fuzzy file lookup."""

TRANSACTION_FILES = {
    "rental": {
        "source": "rental_merged_tenure_1999-2026.csv",
        "enriched": "URA_merged_full_v3_new.csv",
        "cleaned": "URA_merged_full_v3_new_macro_removed_outliers.csv",
    },
    "sales": {
        "source": "raw_realis_transaction.csv",
        "enriched": "URA_sale_data.csv",
        "cleaned": "URA_sale_removed_outliers_2.csv",
    },
}

REFERENCE_PATHS = {
    "Assessed Non Landed Private Residential Units By Locals or Foreigners.csv": "macro/quarterly/Assessed Non Landed Private Residential Units By Locals or Foreigners.csv",
    "Available and Vacant Non-Landed Residential.csv": "macro/quarterly/Available and Vacant Non-Landed Residential.csv",
    "Price Index of Non-Landed Residential Properties.csv": "macro/quarterly/Price Index of Non-Landed Residential Properties.csv",
    "Rental Index of Non-Landed Properties.csv": "macro/quarterly/Rental Index of Non-Landed Properties.csv",
    "Supply of Non-Landed Private Residential.csv": "macro/quarterly/Supply of Non-Landed Private Residential.csv",
    "URA_enriched_with_99co_v2.csv": "property/URA_enriched_with_99co_v2.csv",
    "URA_merged_full_v1.csv": "transactions/rental/enriched/URA_merged_full_v1.csv",
    "URA_merged_full_v3_new.csv": "transactions/rental/enriched/URA_merged_full_v3_new.csv",
    "URA_merged_full_v3_new_macro_removed_outliers.csv": "transactions/rental/cleaned/URA_merged_full_v3_new_macro_removed_outliers.csv",
    "URA_sale_data.csv": "transactions/sales/enriched/URA_sale_data.csv",
    "URA_sale_removed_outliers_2.csv": "transactions/sales/cleaned/URA_sale_removed_outliers_2.csv",
    "Uncompleted Non Landed Private Residential Units By Locals or Foreigners.csv": "macro/quarterly/Uncompleted Non Landed Private Residential Units By Locals or Foreigners.csv",
    "Unit Sold or Unsold.csv": "macro/quarterly/Unit Sold or Unsold.csv",
    "Units Launched or Not Yet Launched.csv": "macro/quarterly/Units Launched or Not Yet Launched.csv",
    "condo_latlon_onemap.csv": "location/condo_latlon_onemap.csv",
    "full_period_condo_mrt_candidates_walk.csv": "location/full_period_condo_mrt_candidates_walk.csv",
    "nearby_5km_amenities.csv": "location/nearby_5km_amenities.csv",
    "property_details_facilities.csv": "property/property_details_facilities.csv",
    "raw_realis_transaction.csv": "transactions/sales/source/raw_realis_transaction.csv",
    "rental_merged_tenure_1999-2026.csv": "transactions/rental/source/rental_merged_tenure_1999-2026.csv",
    "rental_merged_tenure_1999-2026_enriched_macro.csv": "transactions/rental/enriched/rental_merged_tenure_1999-2026_enriched_macro.csv",
    "sale_missing_project_mrt_candidates_walk.csv": "location/sale_missing_project_mrt_candidates_walk.csv",
    "sg_macro_monthly.csv": "macro/monthly/sg_macro_monthly.csv",
    "top30_primary_schools.csv": "location/top30_primary_schools.csv",
}
