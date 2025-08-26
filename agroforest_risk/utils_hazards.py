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

def get_aggregates(path, variables=("tmean", "ppt", "ppt_monthly", "pet_monthly")):
    """
    Load a subset of TerraClimate variables.

    Parameters
    ----------
    path : Path
        Directory containing NetCDFs.
    variables : list/tuple
        Which variables to load. Options: "tmean", "ppt", "ppt_monthly", "pet_monthly".

    Returns
    -------
    dict
        Dictionary mapping variable name -> xarray.DataArray
    """
    results = {}

    if "tmean" in variables:
        import glob
        tmean_files = glob.glob(f"{path}/TerraClimate_tmean_*.nc")

        if tmean_files:  # Case 1: tmean files exist
            ds_tmean = xr.open_mfdataset(tmean_files, combine="by_coords")
            results["tmean"] = ds_tmean.tmean
        else:            # Case 2: compute from tmax + tmin
            print("⚠️ No tmean files found — computing from tmax and tmin...")
            ds_tmax = xr.open_mfdataset(f"{path}/TerraClimate_tmax_*.nc", combine="by_coords")
            ds_tmin = xr.open_mfdataset(f"{path}/TerraClimate_tmin_*.nc", combine="by_coords")
            tmean = (ds_tmax.tmax + ds_tmin.tmin) / 2
            tmean.name = "tmean"
            results["tmean"] = tmean

    if "ppt" in variables:
        ds_ppt = xr.open_mfdataset(f"{path}/TerraClimate_ppt_*.nc", combine="by_coords")
        results["ppt"] = ds_ppt.ppt

    if "ppt_monthly" in variables or "pet_monthly" in variables:
        ds_ppt = xr.open_mfdataset(f"{path}/TerraClimate_ppt_*.nc", combine="by_coords")
        ds_pet = xr.open_mfdataset(f"{path}/TerraClimate_pet_*.nc", combine="by_coords")
        if "ppt_monthly" in variables:
            results["ppt_monthly"] = ds_ppt.ppt
        if "pet_monthly" in variables:
            results["pet_monthly"] = ds_pet.pet

    return results


    
def compute_spei_3(ppt, pet):
    clim = ppt - pet
    out = np.full(clim.shape, np.nan)
    start_year = int(clim.time.dt.year.values[0])
    n_months = clim.sizes["time"]

    for i in range(clim.lat.size):
        for j in range(clim.lon.size):
            ppt_series = ppt[:, i, j].values.astype(float)
            pet_series = pet[:, i, j].values.astype(float)

            if np.isnan(ppt_series).all() or np.isnan(pet_series).all():
                continue
            if np.isnan(ppt_series).any():
                ppt_series[np.isnan(ppt_series)] = np.nanmean(ppt_series)
            if np.isnan(pet_series).any():
                pet_series[np.isnan(pet_series)] = np.nanmean(pet_series)

            try:
                spei_series = spei(
                    precips_mm=ppt_series,
                    pet_mm=pet_series,
                    scale=3,
                    distribution=Distribution.gamma,
                    periodicity=Periodicity.monthly,
                    data_start_year=start_year,
                    calibration_year_initial=start_year,
                    calibration_year_final=start_year + (n_months // 12) - 1
                )
                out[:, i, j] = spei_series
            except Exception as e:
                print(f"⚠️ Grid cell ({i},{j}) failed: {e}")
                continue

    return xr.DataArray(
        out,
        coords={"time": clim.time, "lat": clim.lat, "lon": clim.lon},
        dims=["time", "lat", "lon"]
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
