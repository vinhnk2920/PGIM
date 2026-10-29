"""Shared lagged quarterly macro transformations from the original rental/sale notebooks."""


def add_quarterly_macros(frame, date_column, source_dir, *, assessed_monthly=None):
    import pandas as pd
    import numpy as np

    df_ura = frame.copy()
    _date = pd.to_datetime(df_ura[date_column], errors="coerce")
    if _date.isna().any():
        raise ValueError(f"Invalid dates in {date_column}")
    df_ura = df_ura.assign(Year=_date.dt.year, Month=_date.dt.month, Quarter=_date.dt.quarter)

    def parse_quarter(df, quarter_col="Quarter"):
        df = df.copy()
        df["Year"] = df[quarter_col].str[:4].astype(int)
        df["Quarter"] = df[quarter_col].str[-1].astype(int)
        return df

    def replace_dash_with_nan(df):
        return df.replace("-", np.nan)

    def yoy_log_growth(series):
        log_s = np.log(series.astype(float))
        return log_s - log_s.shift(4)

    def log1p_level(series):
        return np.log1p(series.astype(float))

    def lag1q(series):
        return series.shift(1)

    def quarterly_to_monthly(df_q, year_col="Year", quarter_col="Quarter"):
        quarter_to_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
        rows = []
        for _, row in df_q.iterrows():
            for m in quarter_to_months[row[quarter_col]]:
                r = row.to_dict()
                r["Month"] = m
                rows.append(r)
        df_m = pd.DataFrame(rows)
        return df_m

    df_rental = pd.read_csv(source_dir / "Rental Index of Non-Landed Properties.csv")
    df_rental = replace_dash_with_nan(df_rental)
    df_rental = parse_quarter(df_rental)
    df_rental.columns = df_rental.columns.str.strip()
    col_map = {
        "Rental Index of Non-Landed Properties (INDEX)": "_rental_overall",
        "Rental Index of Non-Landed Properties in Core Central Region (INDEX)": "_rental_ccr",
        "Rental Index of Non-Landed Properties in Outside Central Region (INDEX)": "_rental_ocr",
        "Rental Index of Non-Landed Properties in Rest of Central Region (INDEX)": "_rental_rcr",
    }
    df_rental.rename(columns=col_map, inplace=True)
    df_rental = df_rental.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    for raw_col, feat_col in [
        ("_rental_overall", "rental_index_nonlanded_yoy_log_growth_lag1q"),
        ("_rental_ccr", "rental_index_nonlanded_in_ccr_yoy_log_growth_lag1q"),
        ("_rental_ocr", "rental_index_nonlanded_in_ocr_yoy_log_growth_lag1q"),
        ("_rental_rcr", "rental_index_nonlanded_in_rcr_yoy_log_growth_lag1q"),
    ]:
        df_rental[feat_col] = lag1q(yoy_log_growth(df_rental[raw_col]))
    feat_cols_rental = [
        "Year",
        "Quarter",
        "rental_index_nonlanded_yoy_log_growth_lag1q",
        "rental_index_nonlanded_in_ccr_yoy_log_growth_lag1q",
        "rental_index_nonlanded_in_ocr_yoy_log_growth_lag1q",
        "rental_index_nonlanded_in_rcr_yoy_log_growth_lag1q",
    ]
    df_rental_q = df_rental[feat_cols_rental].copy()
    df_rental_m = quarterly_to_monthly(df_rental_q)
    print(f"Rental Index monthly shape: {df_rental_m.shape}")
    df_ppi = pd.read_csv(source_dir / "Price Index of Non-Landed Residential Properties.csv")
    df_ppi = replace_dash_with_nan(df_ppi)
    df_ppi = parse_quarter(df_ppi)
    col_map = {
        "Property Price Index of Non-Landed Residential Properties (INDEX)": "_ppi_overall",
        "Property Price Index of Non-Landed Residential Properties in Core Central Region (INDEX)": "_ppi_ccr",
        "Property Price Index of Non-Landed Residential Properties in Outside Central Region (INDEX)": "_ppi_ocr",
        "Property Price Index of Non-Landed Residential Properties in Rest of Central Region (INDEX)": "_ppi_rcr",
    }
    df_ppi.rename(columns=col_map, inplace=True)
    df_ppi = df_ppi.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    for raw_col, feat_col in [
        ("_ppi_overall", "ppi_nonlanded_yoy_log_growth_lag1q"),
        ("_ppi_ccr", "ppi_nonlanded_in_ccr_yoy_log_growth_lag1q"),
        ("_ppi_ocr", "ppi_nonlanded_in_ocr_yoy_log_growth_lag1q"),
        ("_ppi_rcr", "ppi_nonlanded_in_rcr_yoy_log_growth_lag1q"),
    ]:
        df_ppi[feat_col] = lag1q(yoy_log_growth(df_ppi[raw_col]))
    feat_cols_ppi = [
        "Year",
        "Quarter",
        "ppi_nonlanded_yoy_log_growth_lag1q",
        "ppi_nonlanded_in_ccr_yoy_log_growth_lag1q",
        "ppi_nonlanded_in_ocr_yoy_log_growth_lag1q",
        "ppi_nonlanded_in_rcr_yoy_log_growth_lag1q",
    ]
    df_ppi_q = df_ppi[feat_cols_ppi].copy()
    df_ppi_m = quarterly_to_monthly(df_ppi_q)
    print(f"Price Index monthly shape: {df_ppi_m.shape}")
    df_vac = pd.read_csv(source_dir / "Available and Vacant Non-Landed Residential.csv")
    df_vac = replace_dash_with_nan(df_vac)
    df_vac = parse_quarter(df_vac)
    col_map = {
        "Vacant Non-Landed Residential Units (NUMBER)": "_vacant",
        "Vacancy Rate of Non-Landed Residential Units (per cent)": "_vacancy_rate",
    }
    df_vac.rename(columns=col_map, inplace=True)
    df_vac = df_vac.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    df_vac["log1p_vacant_nonlanded_units_lag1q"] = lag1q(log1p_level(df_vac["_vacant"]))
    df_vac["vacancy_rate_nonlanded_units_lag1q"] = lag1q(df_vac["_vacancy_rate"].astype(float))
    feat_cols_vac = [
        "Year",
        "Quarter",
        "log1p_vacant_nonlanded_units_lag1q",
        "vacancy_rate_nonlanded_units_lag1q",
    ]
    df_vac_m = quarterly_to_monthly(df_vac[feat_cols_vac])
    print(f"Vacancy monthly shape: {df_vac_m.shape}")
    df_sup = pd.read_csv(source_dir / "Supply of Non-Landed Private Residential.csv")
    df_sup = replace_dash_with_nan(df_sup)
    df_sup = parse_quarter(df_sup)
    col_map = {
        "Supply of Non-Landed Private Residential Units in the Pipeline (NUMBER)": "_total_pipeline",
        "Supply of Non-Landed Private Residential Units in the Pipeline under Construction (NUMBER)": "_under_constr",
    }
    df_sup.rename(columns=col_map, inplace=True)
    df_sup = df_sup.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    df_sup[["_total_pipeline", "_under_constr"]] = df_sup[
        ["_total_pipeline", "_under_constr"]
    ].astype(float)
    df_sup["log1p_nonlanded_private_units_in_the_pipeline_lag1q"] = lag1q(
        log1p_level(df_sup["_total_pipeline"])
    )
    df_sup["nonlanded_private_units_in_the_pipeline_under_construction_share_lag1q"] = lag1q(
        df_sup["_under_constr"] / df_sup["_total_pipeline"]
    )
    feat_cols_sup = [
        "Year",
        "Quarter",
        "log1p_nonlanded_private_units_in_the_pipeline_lag1q",
        "nonlanded_private_units_in_the_pipeline_under_construction_share_lag1q",
    ]
    df_sup_m = quarterly_to_monthly(df_sup[feat_cols_sup])
    print(f"Supply monthly shape: {df_sup_m.shape}")
    df_launch = pd.read_csv(source_dir / "Units Launched or Not Yet Launched.csv")
    df_launch = replace_dash_with_nan(df_launch)
    df_launch = parse_quarter(df_launch)
    col_map = {
        "Non-Landed Units Launched in Private Residential Projects with Pre-requisites for Sale (NUMBER)": "_launched",
        "Non-Landed Units in Private Residential Projects with Pre-requisites but not yet Launched for Sale (NUMBER)": "_not_launched",
        "Non-Landed Units in Private Residential Projects with Pre-requisites for Sale (NUMBER)": "_total_prereq",
    }
    df_launch.rename(columns=col_map, inplace=True)
    df_launch = df_launch.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    df_launch[["_launched", "_not_launched", "_total_prereq"]] = df_launch[
        ["_launched", "_not_launched", "_total_prereq"]
    ].astype(float)
    df_launch[
        "log1p_non_landed_units_launched_in_private_res_projects_with_prereq_for_sale_lag1q"
    ] = lag1q(log1p_level(df_launch["_launched"]))
    df_launch["non_landed_units_in_private_res_projects_with_prereq_not_launched_share_lag1q"] = (
        lag1q(df_launch["_not_launched"] / df_launch["_total_prereq"])
    )
    feat_cols_launch = [
        "Year",
        "Quarter",
        "log1p_non_landed_units_launched_in_private_res_projects_with_prereq_for_sale_lag1q",
        "non_landed_units_in_private_res_projects_with_prereq_not_launched_share_lag1q",
    ]
    df_launch_m = quarterly_to_monthly(df_launch[feat_cols_launch])
    print(f"Launch monthly shape: {df_launch_m.shape}")
    df_sold = pd.read_csv(source_dir / "Unit Sold or Unsold.csv")
    df_sold = replace_dash_with_nan(df_sold)
    df_sold = parse_quarter(df_sold)
    col_map = {
        "Unsold Completed Non-Landed Private Residential Units (NUMBER)": "_unsold_completed",
        "Unsold Non-Landed Units in Launched Private Residential Projects (NUMBER)": "_unsold_launched",
    }
    df_sold.rename(columns=col_map, inplace=True)
    df_sold = df_sold.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    df_sold[["_unsold_completed", "_unsold_launched"]] = df_sold[
        ["_unsold_completed", "_unsold_launched"]
    ].astype(float)
    df_sold["log1p_unsold_completed_nonlanded_private_units_lag1q"] = lag1q(
        log1p_level(df_sold["_unsold_completed"])
    )
    df_sold["log1p_unsold_non_landed_units_in_launched_private_res_projects_lag1q"] = lag1q(
        log1p_level(df_sold["_unsold_launched"])
    )
    feat_cols_sold = [
        "Year",
        "Quarter",
        "log1p_unsold_completed_nonlanded_private_units_lag1q",
        "log1p_unsold_non_landed_units_in_launched_private_res_projects_lag1q",
    ]
    df_sold_m = quarterly_to_monthly(df_sold[feat_cols_sold])
    print(f"Sold/Unsold monthly shape: {df_sold_m.shape}")
    if assessed_monthly is not None:
        df_assessed_m = assessed_monthly.copy()
    else:
        df_assessed = pd.read_csv(
            source_dir / "Assessed Non Landed Private Residential Units By Locals or Foreigners.csv"
        )
        df_assessed = replace_dash_with_nan(df_assessed)
        df_assessed = parse_quarter(df_assessed)
        col_map = {
            "Assessed Non Landed Private Residential Units (NUMBER)": "_total",
            "Assessed Non Landed Private Residential Units Owned by Companies (NUMBER)": "_companies",
            "Assessed Non Landed Private Residential Units Owned by Permanent Residents and Foreigners (NUMBER)": "_pr_foreigners",
            "Assessed Non Landed Private Residential Units Owned by Singaporeans (NUMBER)": "_singaporeans",
        }
        df_assessed.rename(columns=col_map, inplace=True)
        df_assessed = df_assessed.sort_values(["Year", "Quarter"]).reset_index(drop=True)
        df_assessed[["_total", "_companies", "_pr_foreigners", "_singaporeans"]] = df_assessed[
            ["_total", "_companies", "_pr_foreigners", "_singaporeans"]
        ].astype(float)
        df_assessed["assessed_non_landed_private_residential_owned_by_companies_share_lag1q"] = (
            lag1q(df_assessed["_companies"] / df_assessed["_total"])
        )
        df_assessed[
            "assessed_non_landed_private_residential_owned_by_pr_foreigners_share_lag1q"
        ] = lag1q(df_assessed["_pr_foreigners"] / df_assessed["_total"])
        feat_cols_assessed = [
            "Year",
            "Quarter",
            "assessed_non_landed_private_residential_owned_by_companies_share_lag1q",
            "assessed_non_landed_private_residential_owned_by_pr_foreigners_share_lag1q",
        ]
        df_assessed_m = quarterly_to_monthly(df_assessed[feat_cols_assessed])
    print(f"Assessed monthly shape: {df_assessed_m.shape}")
    df_uncomp = pd.read_csv(
        source_dir / "Uncompleted Non Landed Private Residential Units By Locals or Foreigners.csv"
    )
    df_uncomp = replace_dash_with_nan(df_uncomp)
    df_uncomp = parse_quarter(df_uncomp)
    col_map = {
        "Uncompleted Non Landed Private Residential Units Purchased by Companies (NUMBER)": "_companies",
        "Uncompleted Non Landed Private Residential Units Purchased by Foreigners (NUMBER)": "_foreigners",
        "Uncompleted Non Landed Private Residential Units Purchased by Singapore Permanent Residents (NUMBER)": "_spr",
        "Uncompleted Non Landed Private Residential Units Purchased by Singaporeans (NUMBER)": "_singaporeans",
    }
    df_uncomp.rename(columns=col_map, inplace=True)
    df_uncomp = df_uncomp.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    df_uncomp[["_companies", "_foreigners", "_spr", "_singaporeans"]] = df_uncomp[
        ["_companies", "_foreigners", "_spr", "_singaporeans"]
    ].astype(float)
    df_uncomp["_total"] = df_uncomp[["_companies", "_foreigners", "_spr", "_singaporeans"]].sum(
        axis=1
    )
    df_uncomp["uncompleted_non_landed_private_residential_purchased_by_companies_share_lag1q"] = (
        lag1q(df_uncomp["_companies"] / df_uncomp["_total"])
    )
    df_uncomp["uncompleted_non_landed_private_residential_purchased_by_foreigners_share_lag1q"] = (
        lag1q(df_uncomp["_foreigners"] / df_uncomp["_total"])
    )
    df_uncomp["uncompleted_non_landed_private_residential_purchased_by_spr_share_lag1q"] = lag1q(
        df_uncomp["_spr"] / df_uncomp["_total"]
    )
    feat_cols_uncomp = [
        "Year",
        "Quarter",
        "uncompleted_non_landed_private_residential_purchased_by_companies_share_lag1q",
        "uncompleted_non_landed_private_residential_purchased_by_foreigners_share_lag1q",
        "uncompleted_non_landed_private_residential_purchased_by_spr_share_lag1q",
    ]
    df_uncomp_m = quarterly_to_monthly(df_uncomp[feat_cols_uncomp])
    print(f"Uncompleted monthly shape: {df_uncomp_m.shape}")
    macro_dfs = [
        df_rental_m,
        df_ppi_m,
        df_vac_m,
        df_sup_m,
        df_launch_m,
        df_sold_m,
        df_assessed_m,
        df_uncomp_m,
    ]
    df_macro = macro_dfs[0]
    for df_next in macro_dfs[1:]:
        df_macro = pd.merge(df_macro, df_next, on=["Year", "Quarter", "Month"], how="outer")
    df_macro = df_macro.sort_values(["Year", "Month"]).reset_index(drop=True)
    print(f"Combined macro shape: {df_macro.shape}")
    print(f"Year range: {df_macro['Year'].min()} - {df_macro['Year'].max()}")
    df_final = pd.merge(
        df_ura, df_macro.drop(columns=["Quarter"]), on=["Year", "Month"], how="left"
    )
    print(f"Final shape: {df_final.shape}")
    print(f"Row count unchanged: {len(df_final) == len(df_ura)}")
    matched = df_final["vacancy_rate_nonlanded_units_lag1q"].notna().sum()
    print(f"Match rate: {matched}/{len(df_final)} ({matched / len(df_final) * 100:.1f}%)")
    if len(df_final) != len(frame):
        raise ValueError("Quarterly macro merge changed transaction count")
    return df_final
