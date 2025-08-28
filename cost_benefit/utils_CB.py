from pathlib import Path
import pandas as pd
from tabulate import tabulate
import copy
from heapq import merge
from math import cos
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from scipy.interpolate import interp1d
from scipy import sparse
from scipy.stats import genextreme

from climada.hazard import Hazard, Centroids
from climada.entity import Exposures
from climada.entity import ImpactFunc
from climada.entity.measures.cost_income import CostIncome


# For a Python script (.py file):
BASE_DIR = Path.cwd()  # folder where the script is located
INPUT_DIR = BASE_DIR / "Data" # The where we store CB specific input data
OUTPUT_DIR = BASE_DIR / "Results" # The where we store CB specific output data
# Define the present year and the future year
PRESENT_YEAR = 2025
FUTURE_YEAR = 2050

# Set estimates (USD per tonne)
DR_TYPICAL_PRICE_TONNE_USD = {
    "Coffee": 2580,  # 20% premium for specialty coffee
    "Cacao":  3760,   # ≈ matches coffee’s $/ha at 0.5 t/ha
    "Banana": 330,
}

DR_TYPICAL_YIELD = { # tonnes per hectare per year
    "Coffee": 0.73,
    "Cacao":  0.50,
    "Banana": 22.64,
}

DR_TYPICAL_PLANTS_PER_HA = { # plants per hectare
    "Coffee": 3000,
    "Cacao":  1000,
    "Banana": 2000,
}

# Set the yiled and price for other crops (USD per tonne)
## The "Region_key": "DR" deoent matter just place it here
FX_RD_USD = 50.0  # RD$ -> USD, Price/tonne = usd_per_kg * 1000 / FX
# 1) DR fruit table (kg/plant + RD$/kg) -> Tonnes/plant + Price/tonnes (USD)
YIELD_PRICE_USD = [
    {"Region_key": "DR", "Scientific name": "Citrus spp.",      "Kg/plant": 40, "usd_per_kg": 40/ FX_RD_USD},
    {"Region_key": "DR", "Scientific name": "Persea americana", "Kg/plant": 50, "usd_per_kg": 30/ FX_RD_USD},
    {"Region_key": "DR", "Scientific name": "Pouteria sapota",  "Kg/plant": 40, "usd_per_kg": 90/ FX_RD_USD},
    {"Region_key": "DR", "Scientific name": "Castanea spp.",    "Kg/plant": 70, "usd_per_kg": 80/ FX_RD_USD},
]

# species we care about Scientific names, species names
TARGETS = {
    "Coffea arabica":   ("Coffee (main crop)", "Coffee"),
    "Theobroma cacao":  ("Cacao (main crop)",  "Cacao"),
    "Musa spp.":        ("Banana",             "Banana"),
}

 # Manual USD costs per tree (RD$ nursery + planting; maintenance = avg of yearly maint.)
COSTS_DICT_USD = {
    "Inga spp.": {
        "Region_key": "DR",
        "Planting cost (per tree)": 1.50,  # (50 + 25) / 50
        "Maintenance cost (per tree)": 0.88,  # (20+30+50+75)/4 / 50
    },
    "Gliricidia sepium": {
        "Region_key": "DR",
        "Planting cost (per tree)": 1.30,  # (30 + 35) / 50
        "Maintenance cost (per tree)": 0.75,  # (20+30+50+50)/4 / 50
    },
    "Citrus spp.": {
        "Region_key": "DR",
        "Planting cost (per tree)": 1.10,  # (30 + 25) / 50
        "Maintenance cost (per tree)": 0.55,  # (20+30+30+30)/4 / 50
    },
    "Persea americana": {
        "Region_key": "DR",
        "Planting cost (per tree)": 1.10,  # (30 + 25) / 50
        "Maintenance cost (per tree)": 0.55,  # same profile
    },
    "Pouteria sapota": {
        "Region_key": "DR",
        "Planting cost (per tree)": 1.50,  # (50 + 25) / 50
        "Maintenance cost (per tree)": 0.55,  # same profile
    },
    "Castanea spp.": {
        "Region_key": "DR",
        "Planting cost (per tree)": 6.80,  # (315 + 25) / 50
        "Maintenance cost (per tree)": 0.55,  # (20+30+30+30)/4 / 50
    },
}

# Set a default planting and maintenance cost for role = secondary crop if missing
DEF_PLANT_COST_SECONDARY = 1.0  # USD per tree
DEF_MAIN_COST_SECONDARY = 0.5  # USD per tree

#%% Functio to add teh costsa and prices to the pregenerated agroforestry_systems
def write_adjusted( excel_file, OUTPUT_DIR = OUTPUT_DIR, price_mult=1.0, plant_cost_mult=1.0, maint_cost_mult=1.0):
    """
    Read the agrosystem excel file, adjust it to ensure all main species are present,
    backfill missing species in other sheets, and add economic data (yield, price, costs).
    Save the adjusted 'present' sheet to a new Excel file in the OUTPUT_DIR.
    Parameters:
    - excel_file: Path to the input Excel file with canopy compositions.
    - OUTPUT_DIR: Directory to save the output Excel file.
    - price_mult: Multiplier for the price of secondary species (e.g., fruit shade).
    - cost_mult: Multiplier for the planting cost of all species.
    - maint_mult: Multiplier for the maintenance cost of all species.
    """

    excel_file = Path(excel_file)
    print(f"Using input excel file: {excel_file}")
    prefix = excel_file.stem

    # Create output directory if it doesn't exist
    OUTPUT_DIR = OUTPUT_DIR / prefix
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # get the number of sheets in the Excel excel_file apart from the 'Current' sheet
    canopy_comps_zelie = pd.ExcelFile(excel_file).sheet_names

    # Modify the DataFrame to get the site_id and rename columns
    def modify_canopy_crop_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Generate unique site_id
        df["Plot ID"] = df.groupby(["Latitude", "Longitude"]).ngroup()

        # Chnage ti string
        df["Plot ID"] = df["Plot ID"].astype(str)

        return df

    # Create a dictionary to hold the canopy composition data
    canopy_crop_zelie_dict = {}
    for sheet in canopy_comps_zelie:
        canopy_crop_zelie_dict[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
        canopy_crop_zelie_dict[sheet] = modify_canopy_crop_df(canopy_crop_zelie_dict[sheet])
        print(f"Loaded sheet: {sheet} with {len(canopy_crop_zelie_dict[sheet])} rows.")
        print(tabulate(canopy_crop_zelie_dict[sheet], headers='keys', tablefmt='psql'))

    # Print the loaded DataFrames
    for sheet in canopy_crop_zelie_dict:
        print(f"Sheet: {sheet}")
        print(tabulate(canopy_crop_zelie_dict[sheet].head(), headers='keys', tablefmt='psql'))


    # Copy the Zélie present DataFrame to adjust it
    canopy_crop_zelie_dict_adjusted = copy.deepcopy(canopy_crop_zelie_dict)

    # Generate unique site_id
    for sheet, df in canopy_crop_zelie_dict_adjusted.items():
        df["site_id"] = df.groupby(["Latitude", "Longitude"]).ngroup()


    def ensure_main_species_and_update_plants(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # ensure numeric for checks
        out["Yield (t/ha/year)"] = pd.to_numeric(out["Yield (t/ha/year)"], errors="coerce")
        out["Plants/ha"] = pd.to_numeric(out["Plants/ha"], errors="coerce")

        rows_to_add = []

        # group by Plot ID + Region as requested
        for (plot_id, region), g in out.groupby(["Plot ID", "Region"], dropna=False):
            base = g.iloc[0].to_dict()  # copy plot metadata

            for sci_name, (common_label, key) in TARGETS.items():
                present = g[g["Scientific name"] == sci_name]

                if not present.empty:
                    # update Plants/ha only where yield is known
                    idx = present.index[present["Yield (t/ha/year)"].notna()]
                    if len(idx):
                        out.loc[idx, "Plants/ha"] = DR_TYPICAL_PLANTS_PER_HA[key]
                else:
                    # add a new row for this plot with Plants/ha = 0
                    new_row = {col: base.get(col, np.nan) for col in out.columns}
                    new_row["Scientific name"] = sci_name
                    if "Species" in new_row:
                        new_row["Species"] = common_label
                    if "Plants/ha" in new_row:
                        new_row["Plants/ha"] = 0
                    if "Yield (t/ha/year)" in new_row:
                        new_row["Yield (t/ha/year)"] = np.nan
                    rows_to_add.append(new_row)

        if rows_to_add:
            out = pd.concat([out, pd.DataFrame(rows_to_add)], ignore_index=True)

        # Sort the DataFrame by Plot ID and Region
        out.sort_values(by=["Plot ID", "Region", "Scientific name"], inplace=True)

        return out

    # Do it for all sheets in the Zélie dictionary
    for sheet, df in canopy_crop_zelie_dict_adjusted.items():
        canopy_crop_zelie_dict_adjusted[sheet] = ensure_main_species_and_update_plants(df)
        # Print the adjusted DataFrame for the sheet
        print(f"Adjusted Zélie DataFrame for sheet '{sheet}':\n{tabulate(canopy_crop_zelie_dict_adjusted[sheet], headers='keys', tablefmt='psql')}")
        print(f"Adjusted Zélie DataFrame for sheet '{sheet}' has {len(canopy_crop_zelie_dict_adjusted[sheet])} rows.")


    # For all sheets except 'present', only store the columns
    unique_cols = ["site_id", "Species", "Scientific name", "Per-tree shading (%)"]

    # 1) Stack all sheets, tagging each row with its sheet idx
    stacked = pd.concat(
        [df[unique_cols].assign(Source=idx)
        for idx, (_, df) in enumerate(canopy_crop_zelie_dict_adjusted.items())],
        ignore_index=True
    )

    # 2) Prefer rows that HAVE shade; within those, prefer the smallest sheet idx
    stacked["_shade_missing"] = stacked["Per-tree shading (%)"].isna()
    stacked = stacked.sort_values(
        ["site_id", "Species", "Scientific name", "_shade_missing", "Source"],
        ascending=[True, True, True, True, True]   # non-NaN first (False < True), then lowest idx
    )

    # 3) Keep the first occurrence per (site_id, Species, Scientific name)
    df_unique_species = (
        stacked
        .drop_duplicates(subset=["site_id", "Species", "Scientific name"], keep="first")
        .drop(columns=["_shade_missing"])
        .reset_index(drop=True)
    )[["site_id", "Species", "Scientific name", "Per-tree shading (%)", "Source"]]

    # If you want to see it:
    # from tabulate import tabulate
    print(tabulate(df_unique_species, headers='keys', tablefmt='psql'))

    # Drop the 'Source' column if not needed
    df_unique_species = df_unique_species.drop(columns=["Source"])
    print(f"Unique species DataFrame has {len(df_unique_species)} rows.")


    # For all sheets except 'present', only store the columns 
    store_columns = ["Plot ID", "Latitude", "Longitude", "Region", "System", "Plot size (ha)", "Species", "Scientific name", "Plants/ha"]
    for sheet in canopy_crop_zelie_dict_adjusted:
        if sheet != "present":
            canopy_crop_zelie_dict_adjusted[sheet] = canopy_crop_zelie_dict_adjusted[sheet][store_columns]

    # For each sheet in the Zélies data, add the region key
    for sheet in canopy_crop_zelie_dict_adjusted:
        canopy_crop_zelie_dict_adjusted[sheet]["Region_key"] = canopy_crop_zelie_dict_adjusted[sheet]["Region"].str.split(" - ", n=1).str[0].str.strip()


    # Print the adjusted DataFrames
    for sheet, df in canopy_crop_zelie_dict_adjusted.items():
        print(f"Sheet: {sheet}")
        print(tabulate(df, headers='keys', tablefmt='psql'))


    def backfill_species_any_sheet(
        sheet_df: pd.DataFrame,
        df_unique_species: pd.DataFrame,
        present_df_for_site_map: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Ensure each site (by site_id) in `sheet_df` contains all (Species, Scientific name)
        pairs from `df_unique_species` for that site. Append missing rows with Plants/ha = 0.
        If 'Per-tree shading (%)' exists in sheet_df, copy it from df_unique_species.

        site_id mapping:
        1) by exact 'Plot ID' to present
        2) if still missing, by numeric suffix of Plot ID (e.g., 001, 002, 003)
        """
        out = sheet_df.copy()

        needed_min = {"Plot ID", "Region", "System", "Species", "Scientific name", "Plants/ha"}
        missing = needed_min - set(out.columns)
        if missing:
            raise ValueError(f"Sheet is missing required columns: {missing}")

        out["Plants/ha"] = pd.to_numeric(out["Plants/ha"], errors="coerce")

        need_unique = {"site_id","Species","Scientific name","Per-tree shading (%)"}
        if not need_unique.issubset(df_unique_species.columns):
            raise ValueError("df_unique_species must have: site_id, Species, Scientific name, Per-tree shading (%)")

        # --- build site_id map from present ---
        pres = present_df_for_site_map[["Plot ID","site_id"]].drop_duplicates().copy()
        pres["_suffix"] = pres["Plot ID"].str.extract(r"(\d+)$", expand=False)

        added_temp_site = False
        if "site_id" not in out.columns:
            out = out.merge(pres[["Plot ID","site_id"]], on="Plot ID", how="left")
            added_temp_site = True

        # if still missing site_id, map by numeric suffix
        if out["site_id"].isna().any():
            out["_suffix"] = out["Plot ID"].str.extract(r"(\d+)$", expand=False)
            out = out.merge(
                pres[["_suffix","site_id"]].rename(columns={"site_id":"site_id_by_suffix"}),
                on="_suffix",
                how="left"
            )
            out["site_id"] = out["site_id"].fillna(out["site_id_by_suffix"])
            out = out.drop(columns=[c for c in ["_suffix","site_id_by_suffix"] if c in out.columns])

        # If we still have no site_id for a row, we can’t backfill it
        existing = set(zip(out["site_id"], out["Species"], out["Scientific name"]))

        new_rows = []
        for sid, grp in df_unique_species.groupby("site_id", dropna=False):
            if pd.isna(sid):
                continue
            base_rows = out[out["site_id"] == sid]
            if base_rows.empty:
                continue

            base = base_rows.iloc[0].to_dict()
            for _, r in grp.iterrows():
                key = (sid, r["Species"], r["Scientific name"])
                if key in existing:
                    continue

                new_row = {col: base.get(col, np.nan) for col in out.columns}
                new_row["site_id"] = sid
                new_row["Species"] = r["Species"]
                new_row["Scientific name"] = r["Scientific name"]
                new_row["Plants/ha"] = 0
                if "Per-tree shading (%)" in out.columns:
                    new_row["Per-tree shading (%)"] = r["Per-tree shading (%)"]
                new_rows.append(new_row)

        if new_rows:
            out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)

        if added_temp_site:
            out = out.drop(columns=["site_id"])

        sort_cols = [c for c in ["Plot ID","site_id","Species","Scientific name"] if c in out.columns]
        return out.sort_values(sort_cols).reset_index(drop=True)


    # Keep 'present' with full columns (including site_id) so we can map site_id by Plot ID.
    present_full = canopy_crop_zelie_dict_adjusted["present"]

    for name, df in canopy_crop_zelie_dict_adjusted.items():
        canopy_crop_zelie_dict_adjusted[name] = backfill_species_any_sheet(
            sheet_df=df,
            df_unique_species=df_unique_species,
            present_df_for_site_map=present_full
        )
        # Print the adjusted sheets
        print(f"Adjusted Zélie DataFrame for sheet '{name}':\n{tabulate(canopy_crop_zelie_dict_adjusted[name], headers='keys', tablefmt='psql')}")
        print(f"Adjusted Zélie DataFrame for sheet '{name}' has {len(canopy_crop_zelie_dict_adjusted[name])} rows.")


    # Print the adjusted DataFrames
    for sheet in canopy_crop_zelie_dict_adjusted:
        print(f"Sheet: {sheet}")
        print(tabulate(canopy_crop_zelie_dict_adjusted[sheet], headers='keys', tablefmt='psql'))


    # 1) Define main crops (everything else defaults to "Secondary")
    main_species = ["Coffea arabica", "Theobroma cacao"]

    # 2) Collect all unique species across the Zélie dict
    all_species = set()
    for df in canopy_crop_zelie_dict_adjusted.values():
        all_species.update(df["Scientific name"].unique())

    # 3) Build role mapping dynamically
    role_records = []
    for sp in sorted(all_species):
        role = "Main" if sp in main_species else "Secondary"
        role_records.append({"Scientific name": sp, "Role": role})

    role_df = pd.DataFrame(role_records)


    # 4) Merge role info into each sheet
    for sheet, df in canopy_crop_zelie_dict_adjusted.items():
        canopy_crop_zelie_dict_adjusted[sheet] = df.merge(role_df, on="Scientific name", how="left")

    # Print the adjusted DataFrames
    zelie_present_df = canopy_crop_zelie_dict_adjusted['present']

    zelie_present_df.drop(columns=["Yield (t/ha/year)"], inplace=True, errors='ignore')


    df1 = pd.DataFrame(YIELD_PRICE_USD)
    df1["Tonnes/plant"] = df1["Kg/plant"] / 1000.0
    df1["Price/tonnes (USD)"] = (df1["usd_per_kg"] * 1000.0).round(2)
    df1 = df1.drop(columns=["usd_per_kg"])[["Region_key","Scientific name","Kg/plant","Tonnes/plant","Price/tonnes (USD)"]]


    # Update your 'typ' list (Coffee already has 1800)
    typ = [
        {"Region_key":"DR","Scientific name":"Coffea arabica","Yield (t/ha/yr)":DR_TYPICAL_YIELD["Coffee"],"Plants/ha":DR_TYPICAL_PLANTS_PER_HA["Coffee"],"Price/tonnes (USD)":DR_TYPICAL_PRICE_TONNE_USD["Coffee"]},
        {"Region_key":"DR","Scientific name":"Theobroma cacao","Yield (t/ha/yr)":DR_TYPICAL_YIELD["Cacao"],"Plants/ha":DR_TYPICAL_PLANTS_PER_HA["Cacao"],"Price/tonnes (USD)":DR_TYPICAL_PRICE_TONNE_USD["Cacao"]},
        {"Region_key":"DR","Scientific name":"Musa spp.","Yield (t/ha/yr)": DR_TYPICAL_YIELD["Banana"],"Plants/ha":DR_TYPICAL_PLANTS_PER_HA["Banana"],"Price/tonnes (USD)":DR_TYPICAL_PRICE_TONNE_USD["Banana"]},
    ]
    df2 = pd.DataFrame(typ)
    df2["Tonnes/plant"] = (df2["Yield (t/ha/yr)"] / df2["Plants/ha"]).round(6)
    df2["Kg/plant"] = (df2["Tonnes/plant"] * 1000).round(3)
    df2 = df2[["Region_key","Scientific name","Kg/plant","Tonnes/plant","Price/tonnes (USD)"]]

    # 3) Combine
    yield_price_df = pd.concat([df1, df2], ignore_index=True)

    # 4) Duplicate Citrus genus values to species-level for Zélie’s rows
    species_cost_mapping = {
        "Citrus aurantium": "Citrus spp.",
        "Citrus sinensis":  "Citrus spp.",
    }
    # 3) Duplicate rows for mapped species
    def duplicate_species(df, species_cost_mapping):
        """
        Duplicate rows in costs_df_expanded for species in species_cost_mapping.
        Each original species will have its mapped name replaced with the original name.
        """
        expanded_df = df.copy()
        for original_name, mapped_name in species_cost_mapping.items():
            if mapped_name in expanded_df["Scientific name"].values:
                row_to_copy = expanded_df[expanded_df["Scientific name"] == mapped_name].copy()
                row_to_copy["Scientific name"] = original_name
                expanded_df = pd.concat([expanded_df, row_to_copy], ignore_index=True)
        return expanded_df

    yield_price_df = duplicate_species(yield_price_df, species_cost_mapping)

    # (Optional) drop the genus Citrus spp. row if you only want species-level:
    # yield_price_df = yield_price_df[yield_price_df["Scientific name"] != "Citrus spp."]

    # Drop the 'Region_key' column if not needed
    yield_price_df = yield_price_df.drop(columns=["Region_key"], errors='ignore')

    yield_price_df = yield_price_df.sort_values(["Scientific name"]).reset_index(drop=True)
    #print(tabulate(yield_price_df, headers="keys", tablefmt="psql"))


    # -> DataFrame
    costs_df = (
        pd.DataFrame.from_dict(COSTS_DICT_USD, orient="index")
        .reset_index()
        .rename(columns={"index": "Scientific name"})
    )[["Region_key", "Scientific name", "Planting cost (per tree)", "Maintenance cost (per tree)"]]

    # 3) Duplicate rows for mapped species
    def duplicate_species(df, species_cost_mapping):
        """
        Duplicate rows in costs_df_expanded for species in species_cost_mapping.
        Each original species will have its mapped name replaced with the original name.
        """
        expanded_df = df.copy()
        for original_name, mapped_name in species_cost_mapping.items():
            if mapped_name in expanded_df["Scientific name"].values:
                row_to_copy = expanded_df[expanded_df["Scientific name"] == mapped_name].copy()
                row_to_copy["Scientific name"] = original_name
                expanded_df = pd.concat([expanded_df, row_to_copy], ignore_index=True)
        return expanded_df

    # Drop the 'Region_key' column if not needed
    costs_df = costs_df.drop(columns=["Region_key"], errors='ignore')

    #print("Expanded costs DataFrame:")
    costs_df = duplicate_species(costs_df, species_cost_mapping)
    #print(tabulate(costs_df, headers='keys', tablefmt='psql'))

    # Copy the Zélie present DataFrame to adjust it
    zelie_present_adjusted_df = zelie_present_df.copy()

    # Add the region key to the DataFrame
    #zelie_present_adjusted_df["Region_key"] = zelie_present_adjusted_df["Region"].str.split(" - ", n=1).str[0].str.strip()

    # Drop the yield
    zelie_present_df.drop(columns=["Yield (t/ha/year)"], inplace=True, errors='ignore')

    # Merge yield_price_df with zelie_present_adjusted_df
    zelie_present_adjusted_df = zelie_present_adjusted_df.merge(
        yield_price_df,
        on=[ "Scientific name"],
        how="left",
        suffixes=("", "_yield_price")
    )

    # Calculate the new yield
    zelie_present_adjusted_df["Yield (t/ha/year)"] = (
        zelie_present_adjusted_df["Tonnes/plant"] * zelie_present_adjusted_df["Plants/ha"]
    )

    # Add the costs_df_expanded to zelie_present_adjusted_df
    zelie_present_adjusted_df = zelie_present_adjusted_df.merge(costs_df, on=[ "Scientific name"], how="left")

    # Reorder columns to match the desired output
    new_order_columns = ["Plot ID",
        "Region",
        "Latitude",
        "Longitude",
        "System",
        "Plot size (ha)",
        "Species",
        "Scientific name",
        "Role",
        "Plants/ha",
        "Kg/plant",
        "Tonnes/plant",
        "Yield (t/ha/year)",
        "Price/tonnes (USD)",
        "Per-tree shading (%)",
        "Planting cost (per tree)",
        "Maintenance cost (per tree)"
    ]
    zelie_present_adjusted_df = zelie_present_adjusted_df[new_order_columns]

    # Columns to check for missing values
    check_cols = [
        "Planting cost (per tree)",
        "Maintenance cost (per tree)",
        "Tonnes/plant",
        "Price/tonnes (USD)"
    ]

    # Filter for rows with NaN in any of the check columns
    df = zelie_present_adjusted_df.copy()
    missing_df = df[df[check_cols].isna().any(axis=1)]

    # Keep only Scientific name and Region, drop duplicates
    unique_missing = missing_df[["Scientific name", "Yield (t/ha/year)"] + check_cols].drop_duplicates()
    #print(tabulate(unique_missing, headers='keys', tablefmt='psql'))


    # df is your merged Zélie table (the one you showed last)
    df = zelie_present_adjusted_df.copy()

    # Make sure inputs are numeric
    df["Plants/ha"] = pd.to_numeric(df["Plants/ha"], errors="coerce")
    df["Tonnes/plant"] = pd.to_numeric(df["Tonnes/plant"], errors="coerce")
    df["Yield (t/ha/year)"] = pd.to_numeric(df["Yield (t/ha/year)"], errors="coerce")

    # Candidate yield per ha
    candidate = df["Plants/ha"] * df["Tonnes/plant"]

    # Fill only where Yield (t/ha/year) is NaN
    mask = df["Yield (t/ha/year)"].isna() & candidate.notna()
    df.loc[mask, "Yield (t/ha/year)"] = candidate[mask]

    # (optional) round to 2–3 decimals
    df["Yield (t/ha/year)"] = df["Yield (t/ha/year)"].round(3)

    # Update the DataFrame with the new Yield (t/ha/year)
    zelie_present_adjusted_df = df


    # Print the adjusted DataFrame
    def adjust_secondary_econ(
        df: pd.DataFrame,
        price_mult: float = 1.0,       # e.g., 0.8 lowers prices by 20%
        plant_cost_mult: float = 1.0,  # e.g., 0.9 lowers planting cost by 10%
        maint_cost_mult: float = 1.0,  # e.g., 0.75 lowers maintenance by 25%
    ) -> pd.DataFrame:
        """
        Return a copy of df where rows with Role == 'Secondary' have their
        price and per-tree costs scaled by the given multipliers.
        (NaNs remain NaN.)
        """
        out = df.copy()
        sec = out["Role"].eq("Secondary")

        if "Price/tonnes (USD)" in out:
            out.loc[sec, "Price/tonnes (USD)"] = out.loc[sec, "Price/tonnes (USD)"] * price_mult
        if "Planting cost (per tree)" in out:
            out.loc[sec, "Planting cost (per tree)"] = out.loc[sec, "Planting cost (per tree)"] * plant_cost_mult
        if "Maintenance cost (per tree)" in out:
            out.loc[sec, "Maintenance cost (per tree)"] = out.loc[sec, "Maintenance cost (per tree)"] * maint_cost_mult

        return out


    # 1) Make fruit shade less dominant and bump O&M for all secondary trees
    zelie_present_adjusted_incl_price_df = adjust_secondary_econ(
        zelie_present_adjusted_df,
        price_mult=price_mult,                      # 70% of previous price for Secondary (fruit shade)
        plant_cost_mult=plant_cost_mult,                 # 0% chnage planting cost
        maint_cost_mult=maint_cost_mult,                 # 0% chnage maintenance
    )

    excel_dict = copy.deepcopy(canopy_crop_zelie_dict_adjusted)

    # Update the 'present' sheet with the adjusted DataFrame
    excel_dict["present"] = zelie_present_adjusted_incl_price_df

    # # Switch cacaco to coffe but have the same present canopy compostion
    if "cacao_to_coffee" in excel_dict:
        print("Warning: 'cacao_to_coffee' already exists in excel_dict and will be overwritten.")
        df_new_cacao = canopy_crop_zelie_dict_adjusted['present'][store_columns + ["Role"]].copy()
        print(tabulate(df_new_cacao, headers='keys', tablefmt='psql'))

        # Update the Scientific name for Cacao by setting Plants/ha to 0
        df_new_cacao.loc[df_new_cacao['Scientific name'] == 'Theobroma cacao', 'Plants/ha'] = 0
        # Update the Scientific name for Coffe by setting 
        df_new_cacao.loc[df_new_cacao['Scientific name'] == 'Coffea arabica', 'Plants/ha'] = DR_TYPICAL_PLANTS_PER_HA["Coffee"]

        print(tabulate(df_new_cacao, headers='keys', tablefmt='psql'))

        # put back to the excel_dict
        excel_dict['cacao_to_coffee'] = df_new_cacao

    elif "coffee_to_cacao" in excel_dict:
        print("Warning: 'coffee_to_cacao' already exists in excel_dict and will be overwritten.")
        df_new_coffee = canopy_crop_zelie_dict_adjusted['present'][store_columns + ["Role"]].copy()
        print(tabulate(df_new_coffee, headers='keys', tablefmt='psql'))

        # Update the Scientific name for Cacao by setting Plants/ha to 0
        df_new_coffee.loc[df_new_coffee['Scientific name'] == 'Coffea arabica', 'Plants/ha'] = 0
        # Update the Scientific name for Coffe by setting 
        df_new_coffee.loc[df_new_coffee['Scientific name'] == 'Theobroma cacao', 'Plants/ha'] = DR_TYPICAL_PLANTS_PER_HA["Cacao"]

        print(tabulate(df_new_coffee, headers='keys', tablefmt='psql'))

        # put back to the excel_dict
        excel_dict['coffee_to_cacao'] = df_new_coffee


    # ## Save excel version 
    # Save the adjusted DataFrame to an Excel file
    print(f"Input file: {excel_file} with type {type(excel_file)}")

    # Update excel_dict presne so that each secondary species has default costs if missing
    # Default planting cost
    excel_dict['present'].loc[
    (excel_dict['present']['Role'] == 'Secondary') & (excel_dict['present']['Planting cost (per tree)'].isna()),
    'Planting cost (per tree)'
    ] = DEF_PLANT_COST_SECONDARY
    # Default maintenance cost
    excel_dict['present'].loc[
        (excel_dict['present']['Role'] == 'Secondary') & (excel_dict['present']['Maintenance cost (per tree)'].isna()),
        'Maintenance cost (per tree)'
    ] = DEF_MAIN_COST_SECONDARY

    # Safe way with pathlib
    file_name = excel_file.stem + "_adjusted_canopy_crop_composition.xlsx"
    output_file = OUTPUT_DIR / file_name
    print(f"Saving adjusted canopy composition to: {output_file}")
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for sheet_name, df in excel_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Saved sheet: {sheet_name} with {len(df)} rows.")

    return output_file


#%%%%%%%%%% SECTION 1 – Load and Prepare Canopy and Crop Data  

def harmonize_plot_size(canopy_crop_dict, reference_sheet=None, ignore_plot_size=False):
    """
    Ensure 'Plot size (ha)' is identical across all sheets for the same site_id,
    using the reference sheet as the truth. If ignore_plot_size=False, mismatches
    are overwritten to match the reference.
    """
    if reference_sheet is None:
        reference_sheet = list(canopy_crop_dict.keys())[0]

    ref_df = canopy_crop_dict[reference_sheet]
    # Build reference maps by site_id
    ref_plot_size = (ref_df[["site_id", "Plot size (ha)"]]
                     .drop_duplicates()
                     .set_index("site_id")["Plot size (ha)"])

    # (optional) if you also want to enforce lat/long mapping consistency
    ref_geo = (ref_df[["site_id", "latitude", "longitude"]]
               .drop_duplicates()
               .set_index("site_id"))

    print(f"Reference sheet: {reference_sheet}")

    for sheet, df in canopy_crop_dict.items():
        # Skip the reference itself
        if sheet == reference_sheet:
            print(f"✅ {sheet}: using as reference (no changes).")
            continue

        d = df.copy()

        # Check site mapping consistency first (lat/lon vs reference)
        cur_geo = (d[["site_id", "latitude", "longitude"]]
                   .drop_duplicates()
                   .set_index("site_id"))
        # Align on common site_ids to avoid false alarms
        common_sites = ref_geo.index.intersection(cur_geo.index)
        geo_mismatch = not ref_geo.loc[common_sites].equals(cur_geo.loc[common_sites])
        if geo_mismatch:
            print(f"⚠️  {sheet}: latitude/longitude mapping differs from reference for some site_id(s).")

        # Map reference plot size
        d["_Plot size ref (ha)"] = d["site_id"].map(ref_plot_size)

        # Build masks
        mask_missing = d["Plot size (ha)"].isna() & d["_Plot size ref (ha)"].notna()
        mask_diff    = d["Plot size (ha)"].notna() & d["_Plot size ref (ha)"].notna() & \
                       (d["Plot size (ha)"] != d["_Plot size ref (ha)"])

        n_missing = int(mask_missing.sum())
        n_diff    = int(mask_diff.sum())

        if n_missing or n_diff:
            print(f"🔧 {sheet}: {n_missing} missing and {n_diff} differing 'Plot size (ha)' rows vs reference.")
            if not ignore_plot_size:
                d.loc[mask_missing | mask_diff, "Plot size (ha)"] = d.loc[mask_missing | mask_diff, "_Plot size ref (ha)"]
                print(f"   → Harmonized to reference 'Plot size (ha)' for those rows.")
            else:
                print(f"   → IGNORE_PLOT_SIZE=True: leaving values as-is.")
        else:
            print(f"✅ {sheet}: 'Plot size (ha)' already matches reference for all site_id(s).")

        # Clean up helper column and write back
        canopy_crop_dict[sheet] = d.drop(columns=["_Plot size ref (ha)"])

    return canopy_crop_dict

# Function to calculate tree shade per site
def calc_tree_shade_per_site(df: pd.DataFrame,
                            present_df: pd.DataFrame,
                            shade_tree_density: int = 144) -> pd.DataFrame:
    """
    Add per-tree shading, shade contribution, and shade per site.
    
    Args:
        df (pd.DataFrame): Canopy composition DataFrame to update.
        present_df (pd.DataFrame): Reference sheet (usually 'present') with Per-tree shading (%).
        shade_tree_density (int): Maximum number of trees per hectare for normalization.

    Returns:
        pd.DataFrame: Modified DataFrame with tree_shade and related columns.
    """
    df = df.copy()

    # 1. Merge shading values if not already present
    if 'Per-tree shading (%)' not in df.columns:
        matching_cols = ['site_id', 'Scientific name']
        df = df.merge(
            present_df[matching_cols + ['Per-tree shading (%)']],
            on=matching_cols,
            how='left'
        )

    # 2. Calculate
    df['Per-tree shading (%) scaled'] = df['Per-tree shading (%)']/shade_tree_density
    df['Shade contribution (%)'] = df['Per-tree shading (%) scaled'] * df['Plants/ha']
    df['Tree Shade (%)'] = df.groupby("site_id")['Shade contribution (%)'].transform("sum")
    df['tree_shade'] = df['Tree Shade (%)']

    return df


# Modify the DataFrame to get the site_id and rename columns
def modify_canopy_crop_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Rename columns
    df.rename(columns={
        "Latitude": "latitude",
        "Longitude": "longitude"
    }, inplace=True)

    # Generate unique site_id
    df["site_id"] = df.groupby(["latitude", "longitude"]).ngroup()

    return df

# Check consistent site_id mapping
def check_consistent_site_id_mapping(canopy_crop_dict, reference_sheet=None):
    """
    Check that all DataFrames in canopy_crop_dict have the same mapping of (latitude, longitude) to site_id
    as the reference_sheet. Print warnings if inconsistencies are found.
    """
    # Get the reference mapping.
    if reference_sheet is None:
        reference_sheet = list(canopy_crop_dict.keys())[0]

    reference_df = canopy_crop_dict[reference_sheet]
    reference_mapping = reference_df[["latitude", "longitude", "site_id"]].drop_duplicates().sort_values(by=["latitude", "longitude"]).reset_index(drop=True)

    # Compare with each other sheet
    for sheet, df in canopy_crop_dict.items():
        if sheet == reference_sheet:
            continue
        current_mapping = df[["latitude", "longitude", "site_id"]].drop_duplicates().sort_values(by=["latitude", "longitude"]).reset_index(drop=True)
        
        # Compare the two mappings
        if not reference_mapping.equals(current_mapping):
            print(f"⚠️ Inconsistent site_id mapping in sheet: {sheet}")
        else:
            print(f"✅ Consistent site_id mapping in sheet: {sheet}")

# Add columns for plants difference
def add_plants_diff_column(baseline_df: pd.DataFrame, alternative_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - 'Added Plants/ha'    = alt - baseline
      - 'Previous Plants/ha' = baseline minus removals (never below 0)
    Match on ['site_id', 'Scientific name'].
    """
    merged = alternative_df.merge(
        baseline_df[["site_id", "Scientific name", "Plants/ha"]],
        on=["site_id", "Scientific name"],
        how="left",
        suffixes=("", "_baseline"),
    )

    # ensure numeric
    for c in ["Plants/ha", "Plants/ha_baseline"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    merged["Added Plants/ha"] = merged["Plants/ha"] - merged["Plants/ha_baseline"]

    merged["Previous Plants/ha"] = (
        merged["Plants/ha_baseline"]
            .where(merged["Added Plants/ha"] >= 0,
                   merged["Plants/ha_baseline"] + merged["Added Plants/ha"])
            .clip(lower=0)
    )

    return merged


#%%%%%%%%%% SECTION 2 – Exposure  

def make_exposure_from_canopy_df(canopy_df: pd.DataFrame, value_unit: str = "USD") -> Exposures:
    """
    Create an Exposures object from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing exposure data.
        value_unit (str): Unit of value for the exposures.

    Returns:
        Exposures: An Exposures object with the provided data.
    """

    # Store the modified DataFrame for 'present' as a reference
    exp_df = canopy_df.copy()

    # 1. Calculate production and the value of the crops
    exp_df["Yield (t/ha/year)"] = exp_df["Plants/ha"] * exp_df["Tonnes/plant"] # Assuming 'Tonnes/plant' is a column in the DataFrame
    exp_df["Production (t)"] = exp_df["Plot size (ha)"] * exp_df["Yield (t/ha/year)"]
    exp_df["value"] = exp_df["Production (t)"]*exp_df["Price/tonnes (USD)"]

    # Ensure the DataFrame has the required columns
    required_cols = ['site_id', 'latitude', 'longitude', 'tree_shade', 'Production (t)', 'value']
    if not all(col in exp_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain the following columns: {required_cols}")
    
    # Set any rows with NaN in the value column to zero
    exp_df['value'].fillna(0, inplace=True)

    # Create Exposures object
    exp = Exposures(exp_df)
    exp.value_unit = value_unit
    exp.check()
    
    return exp

def make_cacao_shade_curve(n_points=101,
                           peak_shade=45,    # literature: optimal ~25–45%
                           peak_yield=100,   # normalized
                           yield_at_0=80,    # reduced yield without shade
                           yield_at_100=55   # stronger penalty at full shade
                          ):
    """
    Construct a simple quadratic shade–yield curve for cacao.
    
    Anchors:
    - Yield at 0% shade ~70%
    - Maximum yield (100%) around 40% shade
    - Yield declines to ~45% at 100% shade
    
    Returns:
        DataFrame with columns: Shade (%) and Yield (%)
    """
    shade = np.linspace(0, 100, n_points)
    
    # Fit quadratic through anchor points
    X = np.array([
        [0**2, 0, 1],
        [peak_shade**2, peak_shade, 1],
        [100**2, 100, 1]
    ])
    y = np.array([yield_at_0, peak_yield, yield_at_100])
    a, b, c = np.linalg.solve(X, y)

    yield_vals = a*shade**2 + b*shade + c
    yield_vals = np.clip(yield_vals, 0, peak_yield)

    return pd.DataFrame({'Shade (%)': shade, 'Yield (%)': yield_vals})

# Estimate new yield based on shade change
def estimate_new_yield(
    current_shade: float,
    current_yield: float,
    new_shade: float,
    shade_yield_df: pd.DataFrame
) -> float:
    """
    Estimate new yield based on change in shade, using relative scaling
    from the provided empirical shade–yield curve DataFrame.

    Args:
        current_shade (float): Current shade (%) [0–100]
        current_yield (float): Current observed yield (any unit)
        new_shade (float): Target new shade (%)
        shade_yield_df (pd.DataFrame): DataFrame with columns 'Shade (%)' and 'Yield (%)'

    Returns:
        float: Estimated new yield
    """
    interpolate_yield = interp1d(
        shade_yield_df["Shade (%)"],
        shade_yield_df["Yield (%)"],
        kind='linear',
        fill_value="extrapolate"
    )
    current_yield_pct = interpolate_yield(current_shade)
    new_yield_pct = interpolate_yield(new_shade)
    
    return current_yield * (new_yield_pct / current_yield_pct)


# Adjust yield based on new tree shade
def adjust_yield_based_on_new_tree_shade(
    prev_crop_df: pd.DataFrame,
    new_crop_df: pd.DataFrame,
    spec_to_fcn: dict  # {str or list[str]: function}
) -> pd.DataFrame:
    """
    Adjust yield in new_crop_df based on shade and scientific name-specific rules.

    Args:
        prev_crop_df (pd.DataFrame): DataFrame with previous shade and yield.
        new_crop_df (pd.DataFrame): DataFrame to adjust.
        spec_to_fcn (dict): Dictionary where keys are scientific name(s) (str or list),
                            and values are functions that estimate new yield.

    Returns:
        pd.DataFrame: DataFrame with adjusted yield values.
    """
    df = new_crop_df.copy()

    # Merge previous yield and shade
    matching_cols = ['site_id', 'Scientific name']
    df = df.merge(
        prev_crop_df[matching_cols + ['tree_shade', 
                                      'Tonnes/plant',
                                      'Price/tonnes (USD)']],
        on=matching_cols,
        how='left',
        suffixes=('', '_previous')
    )

    df['Tonnes/plant_previous'] = df['Tonnes/plant']

    # Flatten the dictionary to {individual name: function}
    name_to_function = {}
    for names, func in spec_to_fcn.items():
        if isinstance(names, str):
            names = [names]
        for name in names:
            name_to_function[name] = func

    # Apply yield adjustment row-wise
    def compute_new_yield(row):
        name = row['Scientific name']
        func = name_to_function.get(name)
        if func and pd.notna(row['tree_shade_previous']) and pd.notna(row['tree_shade']) and pd.notna(row['Tonnes/plant_previous']):
            return func(
                current_shade=row['tree_shade_previous'],
                current_yield=row['Tonnes/plant_previous'],
                new_shade=row['tree_shade']
            )
        return row['Tonnes/plant_previous']

    df['Tonnes/plant'] = df.apply(compute_new_yield, axis=1)

    return df

# Make exposures from canopy DataFrame
def make_exposures(canopy_df, exposure_present=None, spec_to_fcn={}, value_unit="USD"):
    """
    Create an Exposures object from a canopy DataFrame.

    Args:
        canopy_df (pd.DataFrame): DataFrame containing canopy data.
        base_exposures (Exposures): Base exposures to use for the new exposures.
        value_unit (str): Unit of value for the exposures.
        spec_to_fcn (dict): Dictionary mapping scientific names to yield adjustment functions.

    Returns:
        Exposures: An Exposures object with the provided data.
    """
    
    if exposure_present is None:
        # If no base exposures are provided, create a new Exposures object
        exp = make_exposure_from_canopy_df(canopy_df, value_unit=value_unit)
    else:
        # Use the base exposures and update the DataFrame
        new_canopy_df = canopy_df.copy()
        prev_canopy_df = exposure_present.gdf.copy()
        updated_df = adjust_yield_based_on_new_tree_shade(
            prev_crop_df=prev_canopy_df,
            new_crop_df=new_canopy_df,
            spec_to_fcn=spec_to_fcn
        )
        exp = make_exposure_from_canopy_df(updated_df, value_unit=value_unit)
            
    return exp


def shift_peak_by_interpolation(df, mu_new):
    # Original x, y
    x = df["Shade (%)"].to_numpy(dtype=float)
    y = df["Yield (%)"].to_numpy(dtype=float)

    # Detect old peak location
    mu_old = x[np.argmax(y)]

    # Split original curve at old peak
    left_mask  = x <= mu_old
    right_mask = x >= mu_old

    # Progress variables on each side (0→1 toward the peak)
    p_left  = (x[left_mask] / mu_old)
    p_right = (x[right_mask] - mu_old) / (100.0 - mu_old)

    # Interpolation functions
    def y_left_of(p):
        return np.interp(p, p_left, y[left_mask])
    def y_right_of(p):
        return np.interp(p, p_right, y[right_mask])

    # Build new curve
    y_new = np.empty_like(x, dtype=float)

    # Left of new peak
    left_new = x <= mu_new
    pL = np.divide(x[left_new], mu_new, out=np.zeros_like(x[left_new]), where=mu_new>0)
    y_new[left_new] = y_left_of(np.clip(pL, 0, 1))

    # Right of new peak
    right_new = ~left_new
    denom = (100.0 - mu_new) if mu_new < 100 else np.inf
    pR = np.divide(x[right_new] - mu_new, denom, out=np.zeros_like(x[right_new]), where=denom>0)
    y_new[right_new] = y_right_of(np.clip(pR, 0, 1))

    out = df.copy()
    out["Yield (%)"] = y_new
    return out


#%%%%%%%%%% SECTION 3 – Hazard


def _bootstrap_annual_fields(annual_da: xr.DataArray, n_years=100, start_year=2101, seed=None):
    """Resample annual FIELDS with replacement (preserves spatial patterns)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, annual_da.sizes["year"], size=n_years)
    out = annual_da.isel(year=idx)
    return out.assign_coords(year=np.arange(start_year, start_year + n_years))

def _gev_sample_field(annual_da: xr.DataArray, n_years=100, start_year=2101,
                      invert=False, coupled=False, seed=None):
    """
    Fit GEV per grid and sample n_years (use for annual MAX, or MIN with invert=True).
    coupled=True uses the same quantile across all grid cells per synthetic year (adds coherence).
    """
    rng = np.random.default_rng(seed)
    Y = annual_da
    stacked = Y.stack(grid=("lat","lon"))
    lat, lon = Y["lat"], Y["lon"]

    # Fit params per grid cell
    params = []
    for i in range(stacked.sizes["grid"]):
        ts = stacked.isel(grid=i).values
        ts = ts[~np.isnan(ts)]
        if len(ts) < 10:
            params.append((np.nan, np.nan, np.nan))
            continue
        if invert: ts = -ts
        c, loc, scale = genextreme.fit(ts)
        params.append((c, loc, scale))
    params = np.array(params)

    ngrid = stacked.sizes["grid"]
    # Draw quantiles
    U = rng.random((n_years, 1 if coupled else ngrid))
    U = np.clip(U, 1e-10, 1-1e-10)
    if coupled:
        U = np.repeat(U, ngrid, axis=1)

    samples = np.full((n_years, ngrid), np.nan)
    valid = ~np.isnan(params[:,0])
    if np.any(valid):
        c, loc, scale = params[valid,0], params[valid,1], params[valid,2]
        samples[:, valid] = genextreme.ppf(U[:, valid], c, loc=loc, scale=scale)
    if invert: samples = -samples

    out = samples.reshape(n_years, len(lat), len(lon))
    return xr.DataArray(out,
                        coords={"year": np.arange(start_year, start_year+n_years),
                                "lat": lat, "lon": lon},
                        dims=["year","lat","lon"])



def create_hazard_from_nc_files(
    nc_files,
    varname,
    haz_type="HAZARD",
    units="unit",
    agg="max",
    months=None,                 # e.g. [5,6,7,8,9,10,11]
    nan_replacement=None,        # e.g. 0.0 to fill; None keeps NaNs (then convert later)
    synth=None,                  # None | "bootstrap" | "gev"
    synth_kwargs=None            # dict, e.g. {"n_years":200,"start_year":2101,"invert":False,"coupled":True,"seed":42}
):
    """
    Build a CLIMADA Hazard from NetCDFs, with optional longer synthetic annual series.
    - months is honored (subset BEFORE annual aggregation).
    - agg in {'max','sum','mean'}.
    - synth:
        * None: use observed annual series
        * 'bootstrap': resample annual fields (best for means; preserves spatial patterns)
        * 'gev': fit per-cell GEV and sample (best for block MAX/MIN)
    """
    if synth_kwargs is None:
        synth_kwargs = {}

    print(f"📂 Loading {len(nc_files)} files for '{varname}'...")
    ds_list = [xr.open_dataset(str(f))[varname] for f in nc_files]
    data = xr.concat(ds_list, dim="time")

    if months:
        print(f"📆 Filtering to months: {months}")
        data = data.sel(time=data['time.month'].isin(months))

    print(f"🧮 Aggregating by year: '{agg}'")
    grouped = data.groupby("time.year")
    if agg == "max":
        data_yr = grouped.max("time")
    elif agg == "sum":
        data_yr = grouped.sum("time")
    elif agg == "mean":
        data_yr = grouped.mean("time")
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    # Synthetic extension (optional)
    if synth is None:
        annual_da = data_yr
    elif synth == "bootstrap":
        print("🔁 Synthetic years via BOOTSTRAP of annual fields.")
        annual_da = _bootstrap_annual_fields(data_yr, **synth_kwargs)
    elif synth == "gev":
        if agg != "max":
            print("⚠️ GEV is typically for block maxima/minima; agg!='max' used.")
        print("📈 Synthetic years via GEV sampling (per grid).")
        annual_da = _gev_sample_field(data_yr, **synth_kwargs)
    else:
        raise ValueError("synth must be None, 'bootstrap', or 'gev'.")

    print("📌 Reshaping to [year, lat*lon] ...")
    data_flat = annual_da.stack(site=("lat","lon")).transpose("year","site")
    data_arr = data_flat.values

    if nan_replacement is not None:
        print(f"🔄 Replacing NaNs with {nan_replacement}...")
        data_arr = np.nan_to_num(data_arr, nan=nan_replacement)

    print("🧱 Building sparse intensity ...")
    intensity = sparse.csr_matrix(data_arr)

    print("🗺️ Centroids ...")
    latlon = data_flat.site.to_index().to_frame(index=False)
    cents = Centroids(lat=latlon["lat"].values, lon=latlon["lon"].values)

    years = annual_da["year"].values
    hazard = Hazard(
        haz_type=haz_type,
        intensity=intensity,
        fraction=intensity.copy().astype(bool),
        centroids=cents,
        units=units,
        event_id=np.arange(len(years)),
        frequency=np.ones(len(years))/len(years),
        date=np.array([pd.Timestamp(f"{y}-01-01").toordinal() for y in years]),
        event_name=[f"year_{y}" for y in years]
    )
    return hazard


#%%%%%%%%%% SECTION 4 – Impact functions

def impact_func_from_kath(
    file_path: str,
    intensity_col: str,
    yield_col: str,
    only_yield_losses: bool = True,
    haz_type: str = "VPD",
    unit: str = "kPa",
    if_id: int = 1,
    name: str = "Impact Function from CSV"
) -> ImpactFunc:
    """
    Create a CLIMADA ImpactFunc directly from a CSV file with intensity and yield(-log) data.

    Parameters:
    - file_path: path to CSV
    - intensity_col: column name for hazard intensity (e.g. 'Mean_VPD_kPa')
    - yield_col: column name for yield or log-yield (e.g. 'Effect_Central')
    - is_log: True if yield_col is in log scale
    - only_yield_losses: whether to filter for only negative yield impacts
    - haz_type: CLIMADA hazard type string (e.g. "VPD")
    - unit: unit for intensity (e.g. "kPa", "°C")
    - if_id: Impact function ID
    - name: Name for the impact function

    Returns:
    - ImpactFunc
    """
    df = pd.read_csv(file_path)

    if yield_col not in df.columns or intensity_col not in df.columns:
        raise ValueError("Specified columns not found in the CSV.")

    x = df[intensity_col].values
    y = df[yield_col].values

    # Optional filtering
    if only_yield_losses:
        mask = y < 0
        x, y = x[mask], y[mask]

    # Convert to linear scale if needed
    yield_vals = np.exp(y)

    # Normalize to max yield (best-case baseline)
    baseline = np.max(yield_vals)
    mdd = 1 - (yield_vals / baseline)
    mdd = np.clip(mdd, 0, 1)

    paa = np.ones_like(mdd)

    # Build and return
    impf = ImpactFunc(
        id=if_id,
        name=name,
        haz_type=haz_type,
        intensity=x,
        mdd=mdd,
        paa=paa,
        intensity_unit=unit
    )
    impf.check()
    return impf

# Apply a a function to extra polate the impact function to full damage
def extend_to_full_damage(impf: ImpactFunc, x_full: float | None = None) -> ImpactFunc:
    """
    Extend any ImpactFunc so that MDD reaches 1.0 (100% damage).
    
    Parameters
    ----------
    impf : ImpactFunc
        The impact function to extend (not modified in-place).
    x_full : float or None
        Intensity at which to set MDD=1.
        If None, extrapolate linearly using the slope of the last segment.
    
    Returns
    -------
    ImpactFunc
        A new ImpactFunc with extended arrays.
    """
    impf_new = copy.deepcopy(impf)

    x = np.array(impf_new.intensity, dtype=float)
    mdd = np.array(impf_new.mdd, dtype=float)
    paa = np.array(impf_new.paa, dtype=float)

    # If already reaches full damage, nothing to do
    if np.isclose(mdd[-1], 1.0):
        return impf_new

    # Decide endpoint
    if x_full is None:
        # Linear slope from last two points
        dx = max(x[-1] - x[-2], 1e-9)
        slope = (mdd[-1] - mdd[-2]) / dx
        if slope <= 0:
            slope = 1e-6  # tiny slope to allow reaching 1
        x_full = x[-1] + (1.0 - mdd[-1]) / slope

    # Append endpoint
    x_ext = np.append(x, x_full)
    mdd_ext = np.append(mdd, 1.0)
    paa_ext = np.append(paa, 1.0)

    impf_new.intensity = x_ext
    impf_new.mdd = mdd_ext
    impf_new.paa = paa_ext
    impf_new.check()
    return impf_new

# Create an impact function for Cacao based on Tmax
def cacao_impf_tmax(
    T: np.ndarray = np.linspace(20, 45, 100),               # intensity grid (°C), e.g. seasonal mean/max during flowering
    T_opt: float = 32.0,        # optimum photosynthesis (literature: 31–33 °C)
    T50: float = 36.0,          # ~50% relative yield around 36 °C
    T_maxloss: float = 38.0,    # near-total loss beyond 37–38 °C
    if_id: int = 2, name="Cacao | Tmax", unit="°C"
) -> ImpactFunc:
    T = np.asarray(T, float)
    mdd = np.zeros_like(T, float)
    # No damage up to T_opt
    mdd[T <= T_opt] = 0.0
    # Linear ramp T_opt -> T50
    sel = (T > T_opt) & (T <= T50)
    mdd[sel] = (T[sel]-T_opt)/(T50-T_opt) * 0.5
    # Ramp T50 -> T_maxloss
    sel = (T > T50) & (T <= T_maxloss)
    mdd[sel] = 0.5 + (T[sel]-T50)/(T_maxloss-T50) * 0.5
    mdd[T > T_maxloss] = 1.0
    order = np.argsort(T)
    return ImpactFunc(
        id=if_id, name=name, haz_type="Tmax",
        intensity=T[order],
        mdd=np.maximum.accumulate(mdd[order]),
        paa=np.ones_like(T), intensity_unit=unit
    )


# Create an impact function for Cacao based on VPD
def cacao_impf_vpd(
    V: np.ndarray = np.linspace(0.5, 3.0, 100),             # intensity grid (kPa) for the relevant season
    V50: float = 1.8,           # ~50% relative yield (slightly higher, cacao tolerates some stress)
    k: float = 3.0,             # slope/steepness
    vpd_floor: float = 0.8,     # below this, no damage (typical under shade, canopy cooling)
    if_id: int = 3, name="Cacao | VPD", unit="kPa"
) -> ImpactFunc:
    V = np.asarray(V, float)
    # Relative yield r(V) = 1 / (1 + exp(k*(V - V50)))
    r = 1.0 / (1.0 + np.exp(k*(V - V50)))
    # Flatten pre-stress range (capped at r=1)
    r[V <= vpd_floor] = 1.0
    mdd = np.clip(1.0 - r, 0.0, 1.0)
    order = np.argsort(V)
    return ImpactFunc(
        id=if_id, name=name, haz_type="VPD",
        intensity=V[order],
        mdd=np.maximum.accumulate(mdd[order]),
        paa=np.ones_like(V), intensity_unit=unit
    )


# Generate custom impact function IDs based on canopy DataFrame
def generate_custom_impact_function_IDs_by_canopy_df(
    impf_default_df: pd.DataFrame,
    canopy_df: pd.DataFrame,
    hazard_type: str,
    ignore_impf_ids: list = [0] # Default to ignoring ID 0 (no impact) if not specified
):
    """
    Generate custom impact function IDs based on shade and assign new IDs.

    Args:
        impf_default_df (pd.DataFrame): Columns ['Scientific name', 'impf_id_default']
        canopy_df (pd.DataFrame): Columns ['site_id', 'Scientific name', 'tree_shade']
        imp_fun_set (ImpactFuncSet): Base impact function set to copy and expand.
        adjust_fcn (function): Function to adjust the base impact function using tree_shade.
        hazard_type (str): Hazard type (e.g. "TR").
        ignore_impf_ids (list): List of impf IDs to skip (default: [0] = no impact)

    Returns:
        Tuple[pd.DataFrame, ImpactFuncSet]: 
            - Updated DataFrame with new impact function column `impf_{hazard_type}`
            - Modified ImpactFuncSet with newly added functions
    """

    # get the site_id and scientific name from the canopy_df
    df_site_canopy = canopy_df[['site_id', 'Scientific name']].drop_duplicates()

    # Merge canopy and default impact function assignments
    df_impf_id_map = df_site_canopy.merge(impf_default_df, on='Scientific name', how='left')

    # Copy the impact function set
    #imp_fun_set_new = copy.deepcopy(imp_fun_set)

    # Get next available ID
    next_id = impf_default_df['impf_id_default'].max() + 1

    # For caching and reuse
    species_func_cache = {}

    # Store new IDs
    new_impf_ids = []

    for _, row in df_impf_id_map.iterrows():
        site_id = row['site_id']
        species_name = row['Scientific name']
        base_impf_id = row['impf_id_default']

        if pd.isna(base_impf_id) or base_impf_id in ignore_impf_ids:
            # Use original base ID (e.g. 0 = no impact)
            new_impf_ids.append(base_impf_id)
            continue

        # Use a cache key (species, base ID) to avoid duplication
        key = (site_id, species_name, base_impf_id)

        if key in species_func_cache:
            new_impf_ids.append(species_func_cache[key])
            continue

        # Get base function and generate adjusted one
        species_func_cache[key] = next_id
        new_impf_ids.append(next_id)
        next_id += 1

    # Assign new column to the DataFrame
    df_impf_id_map = df_impf_id_map.copy()
    impf_col_name = f"impf_{hazard_type}"
    df_impf_id_map[impf_col_name] = new_impf_ids


    return df_impf_id_map

# Updated function definition
def helper_create_adj_func(max_cooling=4, plateau_at=70, sensitivity=None):
    """
    Creates an adjustment function for shade effects on the intensity of a hazard.
    Args:
        max_cooling (float): Maximum cooling effect (°C or kPa).
        plateau_at (float): shade percentage at which the cooling effect plateaus.
        sensitivity (float, optional): Sensitivity factor for VPD reduction. If None, no
            sensitivity adjustment is applied.
    Returns:
        function: A function that takes a hazard intensity and shade array
                  and returns the adjusted hazard intensity.
    """
    slope = -max_cooling / plateau_at

    def adjustment(intensity, tree_shade):
        """Applies adjustment to hazard intensity based on shade."""
        if not isinstance(tree_shade, np.ndarray):
            tree_shade = np.array(tree_shade)
        canopy = np.clip(tree_shade, 0, 100)
        temp_reduction = np.where(
            canopy <= plateau_at,
            slope * canopy,
            -max_cooling
        )
        if sensitivity is not None:
            reduction = -sensitivity * temp_reduction
        else:
            reduction = temp_reduction
        return intensity + reduction

    return adjustment


def adjust_impf_by_canopy(impf, adjust_fcn, new_id, current_canopy=50, new_name=None):
    """
    Adjust an impact function based on a canopy adjustment function.

    Parameters:
    -----------
    impf : ImpactFuncSet or compatible object
        Original impact function to be adjusted.
    adjust_fcn : function
        Function that modifies intensity based on shade: f(intensity, canopy).
    new_id : int
        ID for the new adjusted impact function.
    current_canopy : float
        shade percentage to apply (default is 50).

    Returns:
    --------
    adj_impf : ImpactFunc object
        A copy of the original impact function with adjusted intensity and MDR.
    """
    # Copy the original impact function
    adj_impf = copy.deepcopy(impf)

    # Full original intensity range
    orig_intensity = impf.intensity
    orig_min, orig_max = orig_intensity.min(), orig_intensity.max()

    # Use fine spacing to assess max shift
    test_intensity = np.linspace(orig_min, orig_max, 500)
    adjusted = adjust_fcn(test_intensity, current_canopy)
    shift = adjusted - test_intensity
    max_shift = np.abs(shift).max()

    # Extend range to ensure coverage
    n_points = 50
    intensity_range = np.linspace(orig_min - max_shift, orig_max + max_shift, n_points)


    # Check paa logic
    if np.all(impf.paa == 1):
        adj_intensity = adjust_fcn(intensity_range, current_canopy)
        adjusted_mdr = impf.calc_mdr(adj_intensity)

        # Update impact function
        adj_impf.intensity = intensity_range
        adj_impf.mdd = adjusted_mdr
        adj_impf.paa = np.ones_like(intensity_range)
        adj_impf.id = new_id
        if new_name is not None:
            adj_impf.name = new_name

    else:
        print('⚠️ Warning: impf.paa is not all ones, adjustment skipped.')

    return adj_impf


#%%%% %%%%%%%% SECTION 5 – COst and Income

def make_cashflows_costs(canopy_df: pd.DataFrame,
                         costs_df: pd.DataFrame,
                         start_year: int = PRESENT_YEAR,
                         FUTURE_YEAR: int = FUTURE_YEAR,
                         disc_rate=None):
    """
    Build yearly cost cashflows (USD) with columns: year | Previous | Added | total.
    If `disc_rate` (object with .net_present_value(start_year, end_year, cashflows)) is given,
    returns (df, npv). Otherwise returns df.
    """

    # Calculate the number of years
    years = FUTURE_YEAR - start_year + 1

    need_canopy = {"Scientific name", "Previous Plants/ha", "Added Plants/ha"}
    if not need_canopy.issubset(canopy_df.columns):
        raise ValueError(f"canopy_df must contain columns {need_canopy}")

    need_costs = {"Scientific name", "Planting cost (per tree)", "Maintenance cost (per tree)"}
    if not need_costs.issubset(costs_df.columns):
        raise ValueError(f"costs_df must contain columns {need_costs}")

    # Aggregate counts per scientific name
    counts = (canopy_df
              .groupby("Scientific name", as_index=False)[["Previous Plants/ha", "Added Plants/ha"]]
              .sum())

    # Ensure numeric + clip negatives for 'Added'
    counts["Previous Plants/ha"] = pd.to_numeric(counts["Previous Plants/ha"], errors="coerce").fillna(0.0)
    counts["Added Plants/ha"]    = pd.to_numeric(counts["Added Plants/ha"], errors="coerce").fillna(0.0).clip(lower=0)

    # Join costs
    df = counts.merge(costs_df, on="Scientific name", how="left")
    for c in ["Planting cost (per tree)", "Maintenance cost (per tree)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Per-year totals
    previous_per_year     = float((df["Previous Plants/ha"] * df["Maintenance cost (per tree)"]).sum())
    added_maint_per_year  = float((df["Added Plants/ha"]    * df["Maintenance cost (per tree)"]).sum())
    added_oneoff_year1    = float((df["Added Plants/ha"]    * df["Planting cost (per tree)"]).sum())

    years_idx = np.arange(start_year, start_year + years)

    previous = np.full(years, round(previous_per_year, 2))
    added    = np.full(years, round(added_maint_per_year, 2))
    if years > 0:
        added[0] = round(added[0] + added_oneoff_year1, 2)

    out = pd.DataFrame({
        "year": years_idx,
        "Previous": previous,
        "Added": added,
    })
    out["total"] = (out["Previous"] + out["Added"]).round(2)

    # Calculate NPV
    npv_dict = {}
    for col in ["Previous", "Added", "total"]:
        if disc_rate is None:
                npv_dict[col] = out[col].sum()
        else:
            npv_dict[col] = disc_rate.net_present_value(int(out["year"].min()),
                                            int(out["year"].max()),
                                            out[col].to_numpy())
    return out, npv_dict


# make a function to create the cashflows from the canopy_df and costs_df
def make_cost_income_from_canopy_df(
    canopy_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    column: str = 'Added', # 'Added' or 'total'
) -> pd.DataFrame:
    """
    Create cashflows DataFrame from canopy and costs DataFrames.
    Args:
        canopy_df (pd.DataFrame): DataFrame with canopy data.
        costs_df (pd.DataFrame): DataFrame with costs data.
        years (int): Number of years for cashflows.
        start_year (int): Starting year for cashflows.

    Returns:
        pd.DataFrame: DataFrame with cashflows.
    """
    cashflows_df, _ = make_cashflows_costs(
        canopy_df=canopy_df,
        costs_df=costs_df
    )

    # Extract initial and periodic costs
    init_cost = cashflows_df[column].iloc[0] if not cashflows_df.empty else 0.0
    periodic_cost = cashflows_df[column].iloc[1:].mean() if len(cashflows_df) > 1 else 0.0

    # Create a CostIncome object
    cost_income = CostIncome(
        init_cost=init_cost,  # Initial cost from the first row
        periodic_cost=periodic_cost,  # Periodic cost from the mean of subsequent rows
    )
    return cost_income


#%%%%%%%%%% SECTION 8 - Cost and Benefit Analysis

# --- same helper as before ---
def _resolve_col(df, name):
    """Match column allowing for hyphen/en-dash/em-dash and thin spaces."""
    want = name.replace("—", "-").replace("–", "-").replace("\u2009", " ").strip()
    for col in df.columns:
        norm = col.replace("—", "-").replace("–", "-").replace("\u2009", " ").strip()
        if norm == want:
            return col
    raise ValueError(f"Column '{name}' not found. Available: {list(df.columns)}")

def plot_normalized_annual_revenue(
    df,
    role='Main',                 # 'Total' | 'Main' | 'Secondary'
    baseline_comp='present',     # composition to normalize against
    show_labels=False,
    normalize_to='today',        # 'today' | 'future' | 'separate'
):
    # --- Colors to match the attribution plot ---
    LINE_TODAY  = "dimgray"   # present = dark grey
    LINE_FUTURE = "black"     # future  = black

    # Columns (robust to dash variants)
    col_today  = _resolve_col(df, f"Avg. Revenue – {role} (today)")
    col_future = _resolve_col(df, f"Avg. Revenue – {role} (future)")
    col_comp   = _resolve_col(df, "Composition")
    col_canopy = _resolve_col(df, "Avg. shade")

    # Baseline row
    base = df.loc[df[col_comp] == baseline_comp]
    if base.empty:
        raise ValueError(f"Baseline composition '{baseline_comp}' not found.")
    base = base.iloc[0]
    base_today   = float(base[col_today])
    base_future  = float(base[col_future])
    base_canopy  = float(base[col_canopy])

    # Denominators / tags (for legend text)
    if normalize_to == 'today':
        denom_today  = base_today
        denom_future = base_today
        tag_today, tag_future = "today", "today"
    elif normalize_to == 'future':
        denom_today  = base_future
        denom_future = base_future
        tag_today, tag_future = "future", "future"
    elif normalize_to == 'separate':
        denom_today  = base_today
        denom_future = base_future
        tag_today, tag_future = "today", "future"
    else:
        raise ValueError("normalize_to must be 'today', 'future', or 'separate'.")

    if denom_today == 0 or denom_future == 0:
        raise ValueError("Baseline denominator is zero; cannot normalize.")

    # Prepare & normalize (% change wrt denominators)
    d = df.copy().sort_values(col_canopy)
    d["norm_today"]  = (d[col_today]  / denom_today  - 1.0) * 100.0
    d["norm_future"] = (d[col_future] / denom_future - 1.0) * 100.0

    # Baseline markers (what each series equals at the baseline composition)
    y_base_today  = (base_today  / denom_today  - 1.0) * 100.0
    y_base_future = (base_future / denom_future - 1.0) * 100.0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Short, clear legend labels
    lbl_today  = f"Today (norm: {baseline_comp} @ {tag_today})"
    lbl_future = f"Future (norm: {baseline_comp} @ {tag_future})"

    ax.plot(d[col_canopy], d["norm_today"],  marker="o", label=lbl_today,  color=LINE_TODAY)
    ax.plot(d[col_canopy], d["norm_future"], marker="o", label=lbl_future, color=LINE_FUTURE)

    if show_labels:
        for _, r in d.iterrows():
            ax.text(r[col_canopy], r["norm_today"],  r[col_comp],
                    fontsize=8, ha="right", va="bottom", color=LINE_TODAY)
            ax.text(r[col_canopy], r["norm_future"], r[col_comp],
                    fontsize=8, ha="left",  va="bottom", color=LINE_FUTURE)

    # Baseline composition rings (same style; two points may differ in y)
    ax.scatter([base_canopy], [y_base_today],  s=160, facecolors="none",
               edgecolors="grey", linewidths=2, zorder=5)
    ax.scatter([base_canopy], [y_base_future], s=160, facecolors="none",
               edgecolors="grey", linewidths=2, zorder=5)

    # Baseline horizontal line at 0%
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1)

    # Axes & title (more pedagogical)
    ax.set_xlabel("Average shade (%)")
    ax.set_ylabel(f"Change in average annual revenue vs baseline ({role}) [%]")
    baseline_tag = f"{tag_today}" if normalize_to != 'separate' else "today & future"
    ax.set_title(
        f"Normalized Annual Revenue vs Shade — {role}\n"
        f"Baseline: '{baseline_comp}' (normalization reference: {baseline_tag})"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.9)

    # Legend helpers (boxed, below plot — like the other function)
    zero_line = Line2D([0], [0], color='grey', linestyle='--', linewidth=1,
                       label="No change vs baseline (0%)")
    base_ring = Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor='none', markeredgecolor='grey',
                       markersize=10, label=f"Baseline composition: {baseline_comp}")

    handles, labels = ax.get_legend_handles_labels()
    handles += [zero_line, base_ring]
    labels  += ["No change vs baseline (0%)", f"Baseline composition: {baseline_comp}"]

    leg = fig.legend(
        handles, labels,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True, fancybox=True, framealpha=0.95,
        edgecolor="0.5", facecolor="white",
        borderpad=0.8, handlelength=2
    )
    leg.get_frame().set_linewidth(1.2)

    # leave room for the legend box
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    return fig, ax

def derive_gain_risk_from_est(
    est_values_df: pd.DataFrame,
    baseline: str = "present",
    segment: str = "Total",           # "Total" | "Main" | "Secondary"
) -> pd.DataFrame:
    """
    Derive:
      - g(c',c), r(c',c), decision ratios,
      - present & future risk-first attributions (with future risk via R by default).

    Notes:
      - ΔY_present uses raw Yp: Yp(c') - Yp(c)
      - Δrisk_future uses R: Yp(c) * (R(c') - R(c))
      - Δgain_* is always the residual: ΔY_* - Δrisk_*
    """

    seg_tok = {"total": "Total", "main": "Main", "secondary": "Secondary"}[
        segment.strip().lower()
    ]
    col_y_today  = f"Avg. Revenue – {seg_tok} (today)"   # Yp(x)
    col_y_future = f"Avg. Revenue – {seg_tok} (future)"  # Yf(x)
    col_aalp     = f"AAL % – {seg_tok} (today)"
    col_aalf     = f"AAL % – {seg_tok} (future)"

    df = est_values_df.copy()
    df["Segment"] = seg_tok

    # Inputs
    df["Yp"] = df[col_y_today]
    df["Yf"] = df[col_y_future]
    df["AALp_frac"] = df[col_aalp] / 100.0
    df["AALf_frac"] = df[col_aalf] / 100.0

    # Retention R(x) = (1 - AALf) / (1 - AALp)
    denom = (1 - df["AALp_frac"])
    df["R"] = np.where(denom > 0, (1 - df["AALf_frac"]) / denom, np.nan)

    # Baseline values (c)
    base = df.loc[df["Composition"] == baseline]
    if base.empty:
        raise ValueError(f"Baseline '{baseline}' not found.")
    base = base.iloc[0]
    Yp_c, Yf_c = base["Yp"], base["Yf"]
    AALp_c, AALf_c, R_c = base["AALp_frac"], base["AALf_frac"], base["R"]

    # Ratios and decisions
    df["g(c,c′)"] = df["Yp"] / Yp_c                     # gain ratio
    df["r(c,c′)"] = df["R"] / R_c                       # risk ratio
    df["ρ_future = Yf/Yf_baseline"] = df["Yf"] / Yf_c   # equals g*r if Yf=Yp*(1-AALf)
    df["Beneficial_vs_future_baseline"] = df["ρ_future = Yf/Yf_baseline"] > 1

    # --- Present attribution (using present AALs) ---
    df["ΔY_present"]    = df["Yp"] - Yp_c
    df["Δrisk_present"] = Yp_c * ((1 - df["AALp_frac"]) - (1 - AALp_c))
    df["Δgain_present"] = df["ΔY_present"] - df["Δrisk_present"]

    # --- Future attribution ---
    df["ΔY_future"] = df["Yf"] - Yf_c
    df["Δrisk_future"] = Yp_c * ((1 - df["AALf_frac"]) - (1 - AALf_c))
    df["Δgain_future"] = df["ΔY_future"] - df["Δrisk_future"]

    keep_cols = [
        "Composition","Segment",
        *(["Avg. shade"] if "Avg. shade" in df.columns else []),
        *(["Costs (NPV)"] if "Costs (NPV)" in df.columns else []),
        "Yp","Yf","AALp_frac","AALf_frac","R",
        "g(c,c′)","r(c,c′)",
        "ρ_future = Yf/Yf_baseline","Beneficial_vs_future_baseline",
        "ΔY_present","Δrisk_present","Δgain_present",
        "ΔY_future","Δrisk_future","Δgain_future",
    ]
    return df[keep_cols]

def plot_yield_attribution_grouped(
    est_values_df,
    baseline="present",
    segment="Main",        # "Total" | "Main" | "Secondary"
    view="future",         # "present" | "future"  (used in mode="attribution")
    sort_by=None,          # None | "delta" | "risk" | "gain" (ignored in mode="aal")
    relative=False,        # False -> absolute deltas; True -> % of baseline (only for mode="attribution")
    mode="attribution",    # "attribution" (risk/gain bars + total line) | "aal" (ΔYield% lines + ΔAAL bars)
):
    df_all = est_values_df.copy()
    figsize = (10, 6)

    # Colors
    RISK_COLOR  = "#d62728"   # red
    GAIN_COLOR  = "#2ca02c"   # green
    LINE_TODAY  = "dimgray"   # dark grey
    LINE_FUTURE = "black"

    def _resolve_col(df, name):
        want = name.replace("—", "-").replace("–", "-").replace("\u2009", " ").strip()
        for col in df.columns:
            norm = col.replace("—", "-").replace("–", "-").replace("\u2009", " ").strip()
            if norm == want:
                return col
        raise ValueError(f"Column '{name}' not found. Available: {list(df.columns)}")

    if mode == "attribution":
        df = derive_gain_risk_from_est(df_all, baseline=baseline, segment=segment)

        if view.lower() == "future":
            d_total, d_risk, d_gain = "ΔY_future", "Δrisk_future", "Δgain_future"
            base_den_col = "Yf"
            title_tag = "Future"
            baseline_expl = "evaluated under future climate"
            total_line_color = LINE_FUTURE
        else:
            d_total, d_risk, d_gain = "ΔY_present", "Δrisk_present", "Δgain_present"
            base_den_col = "Yp"
            title_tag = "Present"
            baseline_expl = "evaluated under today’s climate"
            total_line_color = LINE_TODAY

        if sort_by == "delta":
            df = df.sort_values(d_total)
        elif sort_by == "risk":
            df = df.sort_values(d_risk)
        elif sort_by == "gain":
            df = df.sort_values(d_gain)

        if relative:
            base_row = df.loc[df["Composition"] == baseline]
            if base_row.empty:
                raise ValueError(f"Baseline '{baseline}' not found in results.")
            denom = float(base_row.iloc[0][base_den_col])
            if denom == 0 or np.isnan(denom):
                raise ValueError(f"Baseline denominator for '{base_den_col}' is zero/NaN.")

            df["_risk_plot"]  = (df[d_risk]  / denom) * 100.0
            df["_gain_plot"]  = (df[d_gain]  / denom) * 100.0
            df["_total_plot"] = (df[d_total] / denom) * 100.0

            ylab  = f"Revenue change vs baseline ({segment}) [%]"
            risk_lbl  = "Attribution: change in vulnerability (hazard risk)"
            gain_lbl  = "Attribution: change due to other factors (incl. canopy-to-yield, increased # of plants etc.)"
            total_lbl = "Annual revenue change (relative, %)"
        else:
            df["_risk_plot"]  = df[d_risk]
            df["_gain_plot"]  = df[d_gain]
            df["_total_plot"] = df[d_total]

            ylab  = f"Revenue change vs baseline ({segment}) [currency]"
            risk_lbl  = "Attribution: change in climate hazard risk"
            gain_lbl  = "Attribution: other factors"
            total_lbl = "Annual revenue change (absolute)"

        x = np.arange(len(df))
        width = 0.35
        off   = width / 2

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - off, df["_risk_plot"].fillna(0).values, width,
               label=risk_lbl, color=RISK_COLOR)
        ax.bar(x + off, df["_gain_plot"].fillna(0).values, width,
               label=gain_lbl, color=GAIN_COLOR)
        ax.plot(x, df["_total_plot"].values, marker="o", linewidth=2,
                color=total_line_color, label=total_lbl)

        ax.axhline(0, linestyle="--", linewidth=1, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Composition"].tolist(), rotation=45, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(f"Annual Revenue Change and Attribution — (Crop: {segment}, Scenario: {title_tag})")
        ax.grid(axis='y', linestyle=':', alpha=0.9)

        baseline_label  = f"Baseline: {baseline} ({baseline_expl})"
        baseline_handle = Line2D([0], [0], marker='o', linestyle='None',
                                 markerfacecolor='none', markeredgecolor='grey',
                                 markersize=10, label=baseline_label)

        handles, labels = ax.get_legend_handles_labels()
        handles += [baseline_handle]
        labels  += [baseline_label]
        leg = fig.legend(
            handles, labels,
            loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.00),
            frameon=True, fancybox=True, framealpha=0.95,
            edgecolor="0.5", facecolor="white",
            borderpad=0.8, handlelength=2
        )
        leg.get_frame().set_linewidth(1.2)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        return fig, ax

    elif mode == "aal":
        df = df_all.copy()

        col_comp   = _resolve_col(df, "Composition")
        col_y_t    = _resolve_col(df, f"Avg. Revenue – {segment} (today)")
        col_y_f    = _resolve_col(df, f"Avg. Revenue – {segment} (future)")
        col_a_t    = _resolve_col(df, f"AAL % – {segment} (today)")
        col_a_f    = _resolve_col(df, f"AAL % – {segment} (future)")

        base = df.loc[df[col_comp] == baseline]
        if base.empty:
            raise ValueError(f"Baseline composition '{baseline}' not found.")
        base = base.iloc[0]

        base_y_t = float(base[col_y_t])
        base_y_f = float(base[col_y_f])
        base_a_t = float(base[col_a_t])
        base_a_f = float(base[col_a_f])
        if base_y_t == 0 or base_y_f == 0:
            raise ValueError("Baseline revenue denominator is zero; cannot normalize.")

        df["_yield_today_pct"]  = (df[col_y_t] / base_y_t - 1.0) * 100.0
        df["_yield_future_pct"] = (df[col_y_f] / base_y_f - 1.0) * 100.0
        df["_aal_today_pp"]     = df[col_a_t] - base_a_t
        df["_aal_future_pp"]    = df[col_a_f] - base_a_f

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(df))
        w = 0.35

        # Risk as bars (red for today, green for future)
        ax.bar(x - w/2, df["_aal_today_pp"].values,  width=w,
               label="ΔAAL (pp) today",  color=RISK_COLOR, alpha=0.95)
        ax.bar(x + w/2, df["_aal_future_pp"].values, width=w,
               label="ΔAAL (pp) future", color=GAIN_COLOR, alpha=0.95)

        # Yield change as lines (dark grey today, black future)
        ax.plot(x, df["_yield_today_pct"].values,  marker="o", linewidth=2,
                color=LINE_TODAY,  label="ΔYield % (today)")
        ax.plot(x, df["_yield_future_pct"].values, marker="o", linewidth=2,
                color=LINE_FUTURE, label="ΔYield % (future)")

        ax.axhline(0, linestyle="--", linewidth=1, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels(df[col_comp].tolist(), rotation=45, ha="right")
        ax.set_ylabel("ΔYield (%) and ΔAAL (pp) vs baseline")
        ax.set_title(f"Yield & Risk Changes vs Baseline — Role: {segment}")
        ax.grid(axis='y', linestyle=':', alpha=0.9)

        baseline_label  = f"Baseline reference — {baseline}"
        baseline_handle = Line2D([0], [0], marker='o', linestyle='None',
                                 markerfacecolor='none', markeredgecolor='grey',
                                 markersize=10, label=baseline_label)

        handles, labels = ax.get_legend_handles_labels()
        handles += [baseline_handle]
        labels  += [baseline_label]
        leg = fig.legend(
            handles, labels,
            loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.02),
            frameon=True, fancybox=True, framealpha=0.95,
            edgecolor="0.5", facecolor="white",
            borderpad=0.8, handlelength=2
        )
        leg.get_frame().set_linewidth(1.2)
        fig.tight_layout(rect=[0, 0.2, 1, 1])
        return fig, ax

    else:
        raise ValueError("mode must be 'attribution' or 'aal'")


def plot_value_stacked_simple(
    exp_dict,
    group_col='Species',      # e.g. 'Role' or 'Species'
    day='today',
    subset=None,              # list of group_col values to show (others -> "Other" or dropped)
    include_other=True,       # if True, non-subset groups collapse to "Other"; if False, they’re dropped
    as_percent=False,         # if True, each bar sums to 100%
    title=None,
):
    """
    Stacked bars by `group_col` across compositions (exp_dict[day].keys()).
    Minimal API. Use `subset` to control which groups appear in the stack.

    Parameters
    ----------
    exp_dict : dict
        exp_dict[day][comp] must have a .gdf (or be a DataFrame) with columns:
        [group_col, 'value'].
    group_col : str
        Column to stack (e.g., 'Role', 'Species').
    day : str
        'today' or any key present in exp_dict.
    subset : list[str] | None
        Which values of `group_col` to display. If None, use all.
    include_other : bool
        If True and subset is provided, aggregate all non-subset groups under 'Other'.
        If False, drop non-subset groups entirely.
    as_percent : bool
        Plot each bar as shares (100%) if True, else absolute values.
    title : str | None
        Optional plot title.
    figsize : tuple
        Figure size.
    """
    comps = list(exp_dict[day].keys())
    frames = []

    for comp in comps:
        df = exp_dict[day][comp].gdf if hasattr(exp_dict[day][comp], "gdf") else exp_dict[day][comp]
        if group_col not in df.columns or 'value' not in df.columns:
            raise ValueError(f"Missing column(s) in '{comp}': need '{group_col}' and 'value'")

        tmp = df.groupby(group_col, as_index=False)['value'].sum()
        tmp['comp'] = comp
        frames.append(tmp)

    tall = pd.concat(frames, ignore_index=True)
    tall['comp'] = pd.Categorical(tall['comp'], categories=comps, ordered=True)

    # Apply subset logic
    if subset is not None:
        subset = set(subset)
        if include_other:
            tall.loc[~tall[group_col].isin(subset), group_col] = 'Other'
        else:
            tall = tall[tall[group_col].isin(subset)]

    # Pivot to wide
    wide = (tall.pivot_table(index='comp', columns=group_col, values='value', aggfunc='sum')
                 .fillna(0))

    # Order stacks by total height
    wide = wide[wide.sum(axis=0).sort_values(ascending=False).index]

    # Normalize to 100% if requested
    if as_percent:
        row_sums = wide.sum(axis=1).replace(0, np.nan)
        wide = (wide.div(row_sums, axis=0).fillna(0) * 100)

    ax = wide.plot(kind='bar', stacked=True, figsize=(9, 5))
    ax.set_xlabel('Composition / Scenario')
    # add exp_dict[day][comp].value_unit if available if not as_percent
    ax.set_ylabel(f"{'Share (%)' if as_percent else f'value ({exp_dict[day][comp].value_unit})'}")
    if as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_ylim(0, 100)

    if title:
        ax.set_title(title)

    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    # Return the axes and wide DataFrame and figure for further use
    return ax, wide, ax.figure


def plot_cost_vs_benefit(
    df_cb_metrics,
    baseline="present",
    role="total",          # "total" (default), "main", or "secondary"
    label_points=True,
    iso_bcr=(0, 1),        # isolines: Benefit = k * Cost
    highlight_rule=None,   # custom highlight function
    figsize=(8, 5),
):
    d = df_cb_metrics.copy()

    # Map role → correct columns
    role = role.lower()
    if role == "total":
        benefit_col, bcr_col, ylabel, title_suffix = "Benefit", "BCR", "Benefit vs baseline (NPV)", ""
    elif role == "main":
        benefit_col, bcr_col, ylabel, title_suffix = "Benefit (Main)", "BCR (Main)", "Benefit (Main crop)", " — Main crop"
    elif role == "secondary":
        benefit_col, bcr_col, ylabel, title_suffix = "Benefit (Secondary)", "BCR (Secondary)", "Benefit (Secondary crops)", " — Secondary crops"
    else:
        raise ValueError("role must be one of: 'total', 'main', 'secondary'")

    # Extract x/y values
    x = d["Costs (NPV)"].values
    y = d[benefit_col].values
    bcr = d[bcr_col].values
    names = d["Composition"].astype(str).values

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter all compositions
    ax.scatter(x, y, s=60, alpha=0.9, label="Compositions")

    # Highlight rule
    if highlight_rule is None:
        highlight_rule = lambda r: (r[bcr_col] >= 1) and (r[benefit_col] > 0)

    mask_eff = np.array([highlight_rule(r) for _, r in d.iterrows()])
    ax.scatter(
        x[mask_eff], y[mask_eff], s=90,
        facecolors='none', edgecolors='k', linewidths=1.5,
        label="Cost-efficient (BCR ≥ 1 & Benefit > 0)"
    )

    # Mark baseline
    if (d["Composition"] == baseline).any():
        i0 = int(d.index[d["Composition"] == baseline][0])
        ax.scatter([x[i0]], [y[i0]], s=160, marker="*",
                   edgecolors="k", facecolors="#FFD166", linewidths=0.8,
                   label=f"Baseline: {baseline}")

    # BCR isolines
    x_line = np.linspace(0, max(x)*1.05 if len(x) else 1, 200)
    iso_handles, iso_labels = [], []
    for k in iso_bcr:
        lbl = "No benefit (BCR = 0)" if k == 0 else ("Break-even (BCR = 1)" if k == 1 else f"BCR = {k:g}")
        h, = ax.plot(x_line, k * x_line, linestyle="--", linewidth=1, color="grey", label=lbl)
        iso_handles.append(h); iso_labels.append(lbl)

    # Labels
    if label_points:
        for xi, yi, nm in zip(x, y, names):
            ax.annotate(nm, (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Total costs (NPV)")
    ax.set_ylabel(ylabel)
    ax.set_title("Cost–Benefit (NPV) with BCR isolines" + title_suffix)
    ax.grid(True, linestyle=":", linewidth=0.8)

    # Legend
    h_plot, l_plot = ax.get_legend_handles_labels()
    handles, labels = h_plot + iso_handles, l_plot + iso_labels
    leg = fig.legend(
        handles, labels,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True, fancybox=True, framealpha=0.95,
        edgecolor="0.5", facecolor="white", borderpad=0.8, handlelength=2
    )
    leg.get_frame().set_linewidth(1.2)
    fig.tight_layout(rect=[0, 0.18, 1, 1])

    return fig, ax


def plot_cb_roles_with_bcr(
    df_cb_metrics: pd.DataFrame,
    df_npv: pd.DataFrame = None,          # provide if role NPVs aren't in df_cb_metrics
    baseline: str = "present",
    figsize=(12,5),
    sort_by: str = None,                  # None | "BCR" | "Benefit" | "Cost"
):
    """
    Left y-axis: grouped bars per composition
        - Net benefit change (ΔRevenue_total − ΔCost)  ← emphasized, separate bar
        - Change in revenue — main crop
        - Change in revenue — secondary/shade trees
        - Change in costs (planting + maintenance)
    Right y-axis: Benefit–Cost Ratio (BCR) as a line.
    All deltas are relative to the baseline composition.
    """
    d = df_cb_metrics.copy()

    # Ensure role NPVs exist
    role_cols = ["Revenue (NPV) – Main", "Revenue (NPV) – Secondary"]
    if any(c not in d.columns for c in role_cols):
        if df_npv is None:
            raise ValueError("Role NPVs missing. Pass df_npv with 'Revenue (NPV) – Main/Secondary'.")
        d = d.merge(df_npv[["Composition"] + role_cols], on="Composition", how="left")

    # Baseline
    b = d.loc[d["Composition"] == baseline]
    if b.empty:
        raise ValueError(f"Baseline '{baseline}' not found.")
    b = b.iloc[0]

    # Deltas vs baseline
    d["ΔRev Main (NPV)"]      = d["Revenue (NPV) – Main"]      - b["Revenue (NPV) – Main"]
    d["ΔRev Secondary (NPV)"] = d["Revenue (NPV) – Secondary"] - b["Revenue (NPV) – Secondary"]
    d["ΔCost (NPV)"]          = d["Costs (NPV)"]               - b["Costs (NPV)"]
    d["ΔBenefit (NPV)"]       = d["ΔRev Main (NPV)"] + d["ΔRev Secondary (NPV)"] - d["ΔCost (NPV)"]

    if "BCR" not in d.columns and {"Benefit","Costs (NPV)"} <= set(d.columns):
        d["BCR"] = d["Benefit"] / d["Costs (NPV)"]

    # Optional sort
    if sort_by == "BCR":
        d = d.sort_values("BCR")
    elif sort_by == "Benefit" and "Benefit" in d.columns:
        d = d.sort_values("Benefit")
    elif sort_by == "Cost":
        d = d.sort_values("ΔCost (NPV)")

    # Positions
    x = np.arange(len(d))
    w = 0.22
    off_total = -1.5 * w
    off_main  = -0.5 * w
    off_seco  =  0.5 * w
    off_cost  =  1.5 * w

    fig, ax = plt.subplots(figsize=figsize)

    # --- Emphasized net benefit (its own bar) ---
    bar_net = ax.bar(
        x + off_total, d["ΔBenefit (NPV)"].values, width=w,
        facecolor="none", edgecolor="black", linewidth=1.8, hatch="////",
        label="Net benefit change (ΔRevenue_total − ΔCost)"
    )

    # --- Role revenues and cost (grouped) ---
    bar_main = ax.bar(x + off_main, d["ΔRev Main (NPV)"].fillna(0).values, width=w,
                      label="Change in revenue — main crop")
    bar_sec  = ax.bar(x + off_seco, d["ΔRev Secondary (NPV)"].fillna(0).values, width=w,
                      label="Change in revenue — secondary (shade trees)")
    bar_cost = ax.bar(x + off_cost, d["ΔCost (NPV)"].fillna(0).values, width=w,
                      label="Change in costs (planting + maintenance)")

    ax.axhline(0, linestyle="--", linewidth=1, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(d["Composition"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Change vs baseline (NPV)")
    ax.set_title(f"Benefits by role and costs vs baseline ({baseline}) with Benefit–Cost Ratio")

    # Right axis: BCR line + reference at 1
    ax2 = ax.twinx()
    line_bcr, = ax2.plot(x, d["BCR"].values, marker="o", linewidth=2, color="purple",
                         label="Benefit–Cost Ratio (total)")
    line_bcr1 = ax2.axhline(1.0, linestyle=":", linewidth=3, color="purple",
                            label="Breakeven (BCR = 1) ")
    ax2.set_ylabel("BCR (Benefit / Cost)")

    # --- Legend (boxed, below the plot) ---
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = h1 + h2
    labels  = l1 + l2

    leg = fig.legend(
        handles, labels,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.06),   # move up/down as needed
        frameon=True, fancybox=True, framealpha=0.95,
        edgecolor="0.5", facecolor="white", borderpad=0.8, handlelength=2
    )
    leg.get_frame().set_linewidth(1.2)

    # Leave room at bottom so the legend isn't clipped
    fig.tight_layout(rect=[0, 0.18, 1, 1])

    # Expand x-limits so outer bars aren’t clipped
    ax.set_xlim(x.min() - 2*w, x.max() + 2*w)

    return fig, ax, ax2