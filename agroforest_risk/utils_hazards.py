import os
import glob
import xarray as xr
import numpy as np
from scipy import sparse
from scipy.stats import genextreme
from climada.hazard import Hazard, Centroids
from climada.util.constants import DEF_CRS
from climate_indices.indices import spei, Distribution
from climate_indices.compute import Periodicity


def clip_to_region(ds, lat_bounds, lon_bounds):
    return ds.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))

from pathlib import Path
import xarray as xr
import glob

def _glob_tc(path, var):
    """
    Find TerraClimate files for variable `var` under `path`, handling both:
      - historical: TerraClimate_<var>_*.nc
      - scenarios:  TerraClimate_<scenario>_<var>_*.nc  (e.g., plus2C/plus4C)
    Returns a sorted list of Paths.
    """
    p = Path(path)
    patterns = [
        f"TerraClimate_{var}_*.nc",         # historical
        f"TerraClimate_*_{var}_*.nc",       # scenarios (catch-all)
    ]
    files = []
    for pat in patterns:
        files.extend(p.glob(pat))
    files = sorted(set(files))
    return files

def _open_da(files, var):
    """Open multiple NetCDFs and return DataArray `var` with coords ordered (time, lat, lon) if present."""
    if not files:
        raise FileNotFoundError(f"No files found for variable '{var}'.")
    ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords")
    da = ds[var]
    # Ensure a consistent axis order when dims exist
    dims = list(da.dims)
    order = [d for d in ["time", "lat", "lon", "y", "x"] if d in dims]
    if order:
        da = da.transpose(*order)
    return da

def get_aggregates(path, variables=("tmean", "ppt", "ppt_monthly", "pet_monthly")):
    """
    Load TerraClimate variables from `path`.

    Parameters
    ----------
    path : str or Path
        Directory containing NetCDFs (historical or scenario).
    variables : iterable
        Any of: "tmean", "ppt", "ppt_monthly", "pet_monthly".

    Returns
    -------
    dict
        var name -> xarray.DataArray
    """
    results = {}

    # --- tmean ---
    if "tmean" in variables:
        tmean_files = _glob_tc(path, "tmean")
        if tmean_files:
            results["tmean"] = _open_da(tmean_files, "tmean")
        else:
            # compute from tmax / tmin if direct tmean is absent
            tmax_files = _glob_tc(path, "tmax")
            tmin_files = _glob_tc(path, "tmin")
            if not tmax_files or not tmin_files:
                raise FileNotFoundError(
                    "tmean not found and cannot compute from tmax/tmin (missing one of them)."
                )
            tmax = _open_da(tmax_files, "tmax")
            tmin = _open_da(tmin_files, "tmin")
            tmean = (tmax + tmin) / 2
            tmean.name = "tmean"
            results["tmean"] = tmean

    # --- ppt (monthly totals) ---
    if "ppt" in variables:
        ppt_files = _glob_tc(path, "ppt")
        results["ppt"] = _open_da(ppt_files, "ppt")

    # --- monthly series (ppt/pet) ---
    if "ppt_monthly" in variables:
        ppt_files = _glob_tc(path, "ppt")
        results["ppt_monthly"] = _open_da(ppt_files, "ppt")

    if "pet_monthly" in variables:
        pet_files = _glob_tc(path, "pet")
        results["pet_monthly"] = _open_da(pet_files, "pet")

    return results


    
def compute_spei_3(ppt, pet, calib_start=1981, calib_end=2010):
    # Ensure expected dims and monthly continuity
    clim = (ppt - pet).transpose("time", "lat", "lon")
    ppt  = ppt.transpose("time", "lat", "lon")
    pet  = pet.transpose("time", "lat", "lon")

    # Basic check
    assert clim.indexes["time"].inferred_type == "datetime64", "Time must be datetime"
    # (Optionally check for monthly frequency / missing months)

    out = np.full(clim.shape, np.nan)
    start_year = int(clim.time.dt.year.values[0])
    n_months = clim.sizes["time"]

    for i in range(clim.lat.size):
        for j in range(clim.lon.size):
            ppt_series = ppt[:, i, j].values.astype(float)
            pet_series = pet[:, i, j].values.astype(float)

            # Skip empty cells
            if np.isnan(ppt_series).all() or np.isnan(pet_series).all():
                continue

            # Gentler gap fill (per series)
            if np.isnan(ppt_series).any():
                m = np.isfinite(ppt_series)
                ppt_series[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), ppt_series[m])
            if np.isnan(pet_series).any():
                m = np.isfinite(pet_series)
                pet_series[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), pet_series[m])

            try:
                spei_series = spei(
                    precips_mm=ppt_series,
                    pet_mm=pet_series,
                    scale=3,
                    distribution=Distribution.loglogistic,     # ✅ SPEI uses log-logistic
                    periodicity=Periodicity.monthly,
                    data_start_year=start_year,
                    calibration_year_initial=calib_start,      # ✅ fixed baseline
                    calibration_year_final=calib_end
                )
                out[:, i, j] = spei_series
            except Exception as e:
                print(f"⚠️ Grid cell ({i},{j}) failed: {e}")
                continue

    return xr.DataArray(
        out,
        coords={"time": clim.time, "lat": clim.lat, "lon": clim.lon},
        dims=["time", "lat", "lon"],
        name="SPEI3"
    )
    
def generate_gev_sample_field(dataarray, n_years=100, start_year=2101, invert=False):
    synthetic_years = np.arange(start_year, start_year + n_years)

    def _fit_and_sample(ts):
        ts = ts[~np.isnan(ts)]
        if len(ts) < 10:
            return np.full(n_years, np.nan)
        try:
            ts = -ts if invert else ts
            shape, loc, scale = genextreme.fit(ts)
            sampled = genextreme.rvs(shape, loc=loc, scale=scale, size=n_years)
            return -sampled if invert else sampled
        except Exception:
            return np.full(n_years, np.nan)

    stacked = dataarray.stack(grid=("lat", "lon"))
    sampled_values = np.array([
        _fit_and_sample(stacked.isel(grid=i).values)
        for i in range(stacked.sizes["grid"])
    ])
    reshaped = sampled_values.reshape((dataarray.sizes["lat"], dataarray.sizes["lon"], n_years))
    reshaped = np.transpose(reshaped, (2, 0, 1))

    return xr.DataArray(
        reshaped,
        coords={"year": synthetic_years, "lat": dataarray.lat, "lon": dataarray.lon},
        dims=["year", "lat", "lon"]
    )

def create_hazard_from_array(dataarray, haz_type, units, start_year=2101):
    years = np.arange(start_year, start_year + dataarray.sizes["year"])
    lat_vals = dataarray.lat.values
    lon_vals = dataarray.lon.values
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    intensity_data = np.vstack([
        dataarray.sel(year=yr).values.reshape(1, -1)
        for yr in years
    ])
    intensity_data = np.nan_to_num(intensity_data, nan=0)

    intensity = sparse.csr_matrix(intensity_data)
    fraction = intensity.copy()
    fraction.data.fill(1)

    centroids = Centroids(
        lat=lat_grid.ravel(),
        lon=lon_grid.ravel(),
        crs=DEF_CRS
    )

    return Hazard(
        haz_type=haz_type,
        intensity=intensity,
        fraction=fraction,
        centroids=centroids,
        units=units,
        event_id=years.tolist(),
        event_name=[f"{haz_type}_{y}" for y in years],
        date=np.array([np.datetime64(f"{y}-07-01").astype("datetime64[D]").astype(int) for y in years]),
        orig=np.zeros(len(years), dtype=bool),
        frequency=np.ones(len(years)) / len(years)
    )



import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from climada.hazard import Hazard

def load_suitability_hazards(files, sample=False):
    """
    Load multiple suitability parquet files into a dict of Hazard objects.

    Parameters
    ----------
    files : list[str or Path]
        List of parquet file paths. Filenames should follow format like
        'scenario_Species_name.parquet' (e.g. 'now_Coffea_arabica.parquet').
    sample : bool, default=False
        If True, generate a hazard by sampling from N(mean, std) using the
        'suitability' and 'suitability_std' columns.

    Returns
    -------
    dict
        Nested dict: hazards[scenario][species] = Hazard
    """
    hazards = {}

    for f in files:
        f = Path(f)
        df = pd.read_parquet(f)

        # Parse scenario and species from filename
        parts = f.stem.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Filename must be 'scenario_species.parquet', got {f.stem}")
        scenario, species = parts
        species = species.replace("_", " ")

        # Prepare grids
        lons = np.sort(df["lon"].unique())
        lats = np.sort(df["lat"].unique())[::-1]
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Choose which column to use
        if sample and "suitability_std" in df.columns:
            values = df["suitability"] + np.random.randn(len(df)) * df["suitability_std"]
            metric = "sample"
        else:
            values = df["suitability"]
            metric = "mean"

        # Fill grid
        Z = np.full_like(lon_grid, np.nan, dtype=np.float32)
        for (_, row), val in zip(df.iterrows(), values):
            i = np.where(lats == row["lat"])[0][0]
            j = np.where(lons == row["lon"])[0][0]
            Z[i, j] = val

        # Add event dimension
        Z = Z[np.newaxis, :, :]
        ds = xr.Dataset(
            {"intensity": (("event", "latitude", "longitude"), Z)},
            coords={
                "event": [0],
                "latitude": lats,
                "longitude": lons,
                "time": pd.to_datetime(["2000-01-01"]),
            },
        )

        haz = Hazard.from_xarray_raster(
            data=ds,
            hazard_type=species,
            intensity_unit="suitability_score",
            intensity="intensity",
        )
        haz.event_id = np.array([0])
        haz.event_name = [f"{species}_{scenario}_{metric}"]

        # Store in dict
        hazards.setdefault(scenario, {})[species] = haz

    return hazards
