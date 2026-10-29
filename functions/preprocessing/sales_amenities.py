"""Optional raw rebuild migrated from data_prep/sale_data/2_merging_data_sale_2.ipynb. See README for missing original reference requirements."""


def run(settings):
    from functions.config import PrepPaths

    paths = PrepPaths(
        settings,
        generated={
            "URA_sale_data.csv",
            "URA_sale_schools_mrt.csv",
            "URA_sale_final_enriched.csv",
            "sale_missing_projects.csv",
            "URA_sale_final_enriched_v2.csv",
            "URA_sale_address.csv",
            "URA_merged_full_v3_new.csv",
            "URA_sale_primary_schools.csv",
        },
    )
    print("[raw preprocessing] source phase 1")
    import pandas as pd

    file_path = paths.file("./URA_sale_final_enriched_v2.csv")
    try:
        df = pd.read_csv(file_path)
        print("--- Dataset Summary ---")
        print("\n--- Data Preview (Top 5 Rows) ---")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    print("[raw preprocessing] source phase 6")
    import pandas as pd

    df_amenities = pd.read_csv(paths.file("./nearby_5km_amenities.csv"))
    columns_to_exclude = ["MRT", "Schools", "URL"]
    cols_to_keep = [
        col
        for col in df_amenities.columns
        if not any((word in col for word in columns_to_exclude))
        and (not any((x in col for x in ["6th", "7th", "8th", "9th", "10th"])))
        and (col != "Project Name")
    ]
    df_amenities_filtered = df_amenities[cols_to_keep].copy()

    def clean_name(name):
        if pd.isna(name):
            return name
        cleaned = str(name).strip().upper().replace("#", "")
        return " ".join(cleaned.split())

    df_amenities_filtered["Project_Key"] = df_amenities_filtered["Project Name in Realis"].apply(
        clean_name
    )
    df["Project_Key"] = df["Project Name"].apply(clean_name)
    df_final = pd.merge(df, df_amenities_filtered, on="Project_Key", how="left")
    column_names = df_final.columns.tolist()
    for i, col in enumerate(column_names, 1):
        print(f"{i}. {col}")
    df_final.drop(columns=["Project_Key"], inplace=True)
    print(f"Total columns after filtering and merging: {len(df_final.columns)}")
    print(f"Total row count preserved: {len(df_final)}")
    df_final.to_csv(paths.file("URA_sale_data.csv"), index=False)
