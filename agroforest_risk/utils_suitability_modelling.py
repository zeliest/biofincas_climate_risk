from pygbif import occurrences as gbif
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import geopandas as gpd
import time
import os
from shapely.geometry import Point
import xarray as xr
import geopandas as gpd


def clean_species_name(name: str) -> str:
    """
    Standardise species name & fix 'Coffee' to 'Coffea'.
    """
    name = name.replace('_', ' ').replace('×', '').strip()
    parts = name.split()
    if len(parts) >= 2:
        genus, species = parts[0].capitalize(), parts[1].lower()
        if genus == "Coffee":
            genus = "Coffea"
        return f"{genus} {species}"
    return name

def fetch_gbif_species(species: str, wkt_bbox: str, year_min=1960, max_records=None, delay=0.5) -> pd.DataFrame:
    """
    Fetch GBIF records >year_min for a species inside bbox.
    """
    all_results, offset, batch_size = [], 0, 300

    while True:
        res = gbif.search(
            scientificName=species,
            hasCoordinate=True,
            geometry=wkt_bbox,
            limit=batch_size,
            offset=offset
        )
        results = res.get("results", [])
        if not results:
            break

        filtered = [r for r in results if r.get("year") and r["year"] > year_min]
        all_results.extend(filtered)
        offset += batch_size

        print(f"  📦 {offset} total | {len(all_results)} kept >{year_min} for {species}")
        if max_records and offset >= max_records:
            break
        time.sleep(delay)

    return pd.json_normalize(all_results)

def load_existing(path="occurrences_raw.parquet"):
    """
    Load existing GBIF data if available.
    """
    if os.path.exists(path):
        df = pd.read_parquet(path)
        fetched = set(df['species_query'].unique()) if not df.empty else set()
        print(f"📄 Loaded {len(fetched)} species from {path}")
    else:
        df, fetched = pd.DataFrame(), set()
        print(f"📄 No existing data at {path}, starting fresh.")
    return df, fetched



def generate_background_points_on_land(bbox, n_points=5000, seed=42):
    lon_min, lon_max, lat_min, lat_max = bbox
    np.random.seed(seed)

    # Load Natural Earth land polygons
    land = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    land = land[land['geometry'].type.isin(['Polygon', 'MultiPolygon'])]

    # Oversample so we have enough points after filtering
    bg_lons = np.random.uniform(lon_min, lon_max, n_points * 3)
    bg_lats = np.random.uniform(lat_min, lat_max, n_points * 3)
    points = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(bg_lons, bg_lats)],
        crs="EPSG:4326"
    )

    # Keep only points on land
    points_on_land = gpd.sjoin(points, land, how="inner", predicate='intersects')

    # Limit to requested number
    points_on_land = points_on_land.head(n_points)

    # Return lon/lat arrays
    return points_on_land.geometry.x.values, points_on_land.geometry.y.values

def build_training_dataset(
    presence_df,
    bio_stack,
    bbox,
    selected_predictors,
    background_ratio=5,
    max_background=5000,
    seed=42
):
    """
    Batch extraction using KDTree, with optional max_background cap.
    Now generates background points only on land.
    """
    lon_min, lon_max, lat_min, lat_max = bbox

    # 🎯 Filter presence points
    presence_in_bbox = presence_df[
        (presence_df['lon'] >= lon_min) &
        (presence_df['lon'] <= lon_max) &
        (presence_df['lat'] >= lat_min) &
        (presence_df['lat'] <= lat_max)
    ].copy()

    print(f"✅ Found {len(presence_in_bbox)} presence points in study area.")

    # 📋 Background points (land only)
    n_background = len(presence_in_bbox) * background_ratio
    bg_lons, bg_lats = generate_background_points_on_land(bbox, n_points=n_background, seed=seed)

    # 📋 Combine presence & background
    all_lons = np.concatenate([presence_in_bbox.lon.values, bg_lons])
    all_lats = np.concatenate([presence_in_bbox.lat.values, bg_lats])
    labels = np.array([1]*len(presence_in_bbox) + [0]*len(bg_lons))
    species = list(presence_in_bbox.species) + ["background"]*len(bg_lons)

    print(f"🚀 Querying predictors for {len(all_lons)} points…")
    predictors_array = extract_predictors_batch(all_lons, all_lats, bio_stack, selected_predictors)

    df = pd.DataFrame(predictors_array, columns=selected_predictors)
    df["lon"] = all_lons
    df["lat"] = all_lats
    df["presence"] = labels
    df["species"] = species

    # Drop invalid rows (e.g., NaN predictors)
    df = df[np.all(np.isfinite(df[selected_predictors]), axis=1)].reset_index(drop=True)

    # Cap background points if too many
    bg = df[df.presence == 0].copy()
    pres = df[df.presence == 1].copy()
    if len(bg) > max_background:
        bg = bg.sample(n=max_background, random_state=seed)

    df_final = pd.concat([pres, bg], ignore_index=True)

    print(f"🎉 Final dataset: {len(df_final)} records.")
    print(df_final["presence"].value_counts())
    print(df_final["species"].value_counts())

    return df_final




def extract_predictors_batch(lons, lats, bio_stack, predictors):
    """
    Extract predictor values from bio_stack for many points at once.
    Uses nearest-neighbor lookup on lon/lat grid.
    """
    # Detect coordinate names
    lon_name = "x" if "x" in bio_stack.dims else "lon"
    lat_name = "y" if "y" in bio_stack.dims else "lat"

    extracted = []
    for var in predictors:
        # Select nearest grid cell for each point
        vals = [
            bio_stack[var]
            .sel({lon_name: float(lon), lat_name: float(lat)}, method="nearest")
            .item()
            for lon, lat in zip(lons, lats)
        ]
        extracted.append(vals)

    # Stack into (n_points, n_predictors)
    return np.column_stack(extracted)


def fetch_gbif_species(species: str, wkt_bbox: str, year_min=1960, max_records=None, delay=0.5) -> pd.DataFrame:
    """
    Fetch GBIF records newer than year_min for a species inside bbox.
    """
    all_results, offset, batch_size = [], 0, 300

    while True:
        res = gbif.search(
            scientificName=species,
            hasCoordinate=True,
            geometry=wkt_bbox,
            limit=batch_size,
            offset=offset
        )
        results = res.get("results", [])
        if not results:
            break

        filtered = [r for r in results if r.get("year") and r["year"] >= year_min]
        all_results.extend(filtered)
        offset += batch_size

        print(f"  📦 {offset} total | {len(all_results)} kept ≥{year_min} for {species}")
        if max_records and offset >= max_records:
            break
        time.sleep(delay)

    return pd.json_normalize(all_results)


def get_gbif_occurrences_for_species(species_list, wkt_bbox, year_min=1960):
    """
    Fetch GBIF occurrence records for species within bbox, filtering by year_min.
    """
    all_records = []
    for species in species_list:
        try:
            print(f"🌱 Fetching GBIF: {species}")
            records = fetch_gbif_species(species, wkt_bbox, year_min=year_min)
            if not records.empty:
                records["species_query"] = clean_species_name(species)
                all_records.append(records)
            else:
                print(f"⚠️ No records ≥{year_min} for: {species}")
        except Exception as e:
            print(f"❌ Failed for {species}: {e}")

    if all_records:
        return pd.concat(all_records, ignore_index=True)
    else:
        return pd.DataFrame(columns=["species_query", "decimalLongitude", "decimalLatitude"])


