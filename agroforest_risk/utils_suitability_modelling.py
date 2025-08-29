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
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix


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


def clean_occurrences(df, max_isolate_distance_deg=0.5, thinning_deg=2.5/60):
    """
    Clean occurrence points per species by:
    1. Removing isolated points farther than `max_isolate_distance_deg` from all others.
    2. Applying spatial thinning so that only 1 point is kept per grid cell of size `thinning_deg`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: species (or species_query), decimalLongitude, decimalLatitude.
    max_isolate_distance_deg : float
        Maximum allowed distance (in degrees) for a point to have a neighbour. 
        Points farther than this from all others are removed.
    thinning_deg : float
        Grid cell size for thinning, in degrees. Default 2.5 arc‑min (~0.0416667°).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame of occurrences.
    """

    def remove_isolated_points(species_df):
        coords = species_df[["decimalLongitude", "decimalLatitude"]].to_numpy()
        keep_idx = []
        for i, c in enumerate(coords):
            dists = np.sqrt((coords[:,0] - c[0])**2 + (coords[:,1] - c[1])**2)
            dists[i] = np.inf  # ignore self
            if np.any(dists < max_isolate_distance_deg):
                keep_idx.append(i)
        return species_df.iloc[keep_idx]

    def thin_points(species_df):
        # Assign grid cell indices
        grid_x = np.floor(species_df["decimalLongitude"] / thinning_deg)
        grid_y = np.floor(species_df["decimalLatitude"] / thinning_deg)
        species_df = species_df.assign(grid_x=grid_x, grid_y=grid_y)
        # Randomly sample 1 point per grid cell
        thinned = species_df.groupby(["grid_x", "grid_y"], group_keys=False).apply(
            lambda g: g.sample(1, random_state=42)
        )
        return thinned.drop(columns=["grid_x", "grid_y"])

    cleaned_species_list = []
    for species, group in df.groupby("species_query" if "species_query" in df.columns else "species"):
        # 1️⃣ Remove extreme isolates
        non_isolated = remove_isolated_points(group)
        # 2️⃣ Apply spatial thinning
        thinned = thin_points(non_isolated)
        print(f"{species}: {len(group)} → {len(thinned)} after cleaning")
        cleaned_species_list.append(thinned)

    return pd.concat(cleaned_species_list, ignore_index=True)

def drop_correlated(df, predictors, threshold=0.85):
    corr = df[predictors].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    keep = [p for p in predictors if p not in to_drop]
    return keep, to_drop

def calculate_vif(df, predictors):
    X = df[predictors].dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = predictors
    vif_data["VIF"] = [sm.OLS(X[col], sm.add_constant(X.drop(columns=[col]))).fit().rsquared for col in X.columns]
    vif_data["VIF"] = 1 / (1 - vif_data["VIF"])
    return vif_data

def drop_high_vif(df, predictors, threshold=10):
    vif_data = calculate_vif(df, predictors)
    to_drop = vif_data.loc[vif_data["VIF"] > threshold, "feature"].tolist()
    keep = [p for p in predictors if p not in to_drop]
    return keep, to_drop, vif_data

def max_sens_spec_threshold(y_true, y_prob):
    thresholds = np.linspace(0, 1, 101)
    best_thresh = 0.5
    best_score = -1
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if tp + fn > 0 else 0
        spec = tn / (tn + fp) if tn + fp > 0 else 0
        score = sens + spec
        if score > best_score:
            best_score = score
            best_thresh = thr
    return best_thresh

