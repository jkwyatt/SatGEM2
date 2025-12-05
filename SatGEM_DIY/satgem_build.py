import copernicusmarine
import xarray as xr
from tqdm import tqdm
import numpy as np
import os
import csv
from datetime import datetime, timedelta
from pathlib import Path  # NEW

import urllib.request
import tempfile

def resolve_path(path):
    """
    Accepts a local path or a URL.
    If URL → downloads to a temporary file and returns the local filename.
    If local path → returns Path(path).
    """
    path = str(path)

    # If it's a URL, download it
    if path.startswith("http://") or path.startswith("https://"):
        tmp_dir = Path(tempfile.gettempdir())
        filename = tmp_dir / Path(path).name.split("?")[0]  # clean '?download=1'
        print(f"Downloading {path} → {filename}")

        urllib.request.urlretrieve(path, filename)
        return filename

    # Otherwise treat as local file path
    return Path(path)


# ============================================
# Path setup: make things relative to the repo
# ============================================
# Folder that contains THIS script
HERE = Path(__file__).resolve().parent

# If this script lives in the top-level of the repo, use:
REPO_ROOT = HERE

# If this script lives in a subfolder (e.g. SatGEM2/satGEM_DIY/build_satGEM.py),
# then use the parent:
# REPO_ROOT = HERE.parent

# Default locations *inside the repo* (edit if you use different names)
DEFAULT_CSV_PATH = REPO_ROOT / "saved_adt_rel.csv"        # or REPO_ROOT / "data" / "saved_adt_rel.csv"
DEFAULT_TS_DIR = REPO_ROOT / "ts_gem_fields"              # folder with GEM_{lon}.nc
DEFAULT_GAMMA_DIR = REPO_ROOT / "gamma_gem_fields"        # folder with GEM_{lon}_gamma_n.nc
DEFAULT_COMBINED_GEM_PATH = REPO_ROOT / "GEM_combined.nc"


# ============================================
# Helper: load slope / intercept for a longitude
# ============================================
def load_data_for_longitude(csv_path, target_longitude):
    """
    Reads a CSV with columns: longitude, slope, intercept
    and returns (slope, intercept) for the given target_longitude.
    """
    csv_path = Path(csv_path)  # ensure it's a Path

    with csv_path.open('r') as csv_file:
        csv_reader = csv.reader(csv_file)
        # Skip the header row
        next(csv_reader)
        for row in csv_reader:
            if len(row) < 3:
                continue
            longitude, slope, intercept = row
            if float(longitude) == target_longitude:
                try:
                    slope = float(slope)
                    intercept = float(intercept)
                    return slope, intercept
                except ValueError:
                    # If conversion fails, return None for both slope and intercept
                    return None, None
    # If no matching longitude found
    return None, None


# ============================================
# Helper: parse dates (single date or range)
# ============================================
def parse_dates(dates):
    """
    Accepts:
      - "2020-01-01"
      - ("2020-01-01", "2020-01-10")
    Returns (start_str, end_str) suitable for Copernicus:
      [start, end) in daily resolution.
    """
    if isinstance(dates, str):
        start = datetime.strptime(dates, "%Y-%m-%d")
        end = start + timedelta(days=1)
    elif isinstance(dates, (list, tuple)) and len(dates) == 2:
        start = datetime.strptime(dates[0], "%Y-%m-%d")
        end = datetime.strptime(dates[1], "%Y-%m-%d") + timedelta(days=1)
    else:
        raise ValueError("dates must be 'YYYY-MM-DD' or ('YYYY-MM-DD', 'YYYY-MM-DD').")

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ============================================
# Main function: build SatGEM fields
# ============================================
def build_satGEM_fields(
    dates,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    csv_path=None,
    combined_gem_path=None,
    dataset_id="c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
):
    """
    Build SatGEM ts and gamma fields for a given date or date range,
    longitude range, and latitude range, using a single combined GEM file.

    Parameters
    ----------
    dates : str or (str, str)
        Single date: "2020-01-01"
        or date range: ("2020-01-01", "2020-01-10")
    lon_min, lon_max : float
        Longitude range in degrees (can be -180..180 or 0..360).
    lat_min, lat_max : float
        Latitude range in degrees.
    csv_path : str or Path, optional
        Path to CSV with (longitude, slope, intercept).
        Defaults to CSV inside the repo.
    combined_gem_path : str or Path, optional
        Path to GEM_combined.nc (all longitudes concatenated).
        Defaults to GEM_combined.nc in the repo root.
    dataset_id : str
        Copernicus Marine SSH dataset ID.

    Returns
    -------
    satGEM_ts_field : xr.Dataset
        TS fields on the SSH grid (includes time dimension if multiple days).
    satGEM_gamma_field : xr.DataArray
        gamma_n on the SSH grid, regridded to a common pressure axis.
    """

    # ----------------------------------------
    # 0) Resolve default paths if not provided
    # ----------------------------------------
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH
    if combined_gem_path is None:
        combined_gem_path = DEFAULT_COMBINED_GEM_PATH

    csv_path = resolve_path(csv_path)
    combined_gem_path = resolve_path(combined_gem_path)

    # ----------------------------------------
    # 1) Build slopes_intercepts dict (0–359)
    # ----------------------------------------
    slopes_intercepts = {}
    for lon in range(-180, 180):
        adjusted_lon = (lon + 360) % 360  # 0..359
        slopes_intercepts[adjusted_lon] = load_data_for_longitude(csv_path, lon)

    # ----------------------------------------
    # 2) Parse dates for CopernicusMarine
    # ----------------------------------------
    start_str, end_str = parse_dates(dates)

    # ----------------------------------------
    # 3) Open SSH from Copernicus for date range
    # ----------------------------------------
    ssh = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=["adt"],
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=start_str,
        end_datetime=end_str,
    )

    # Put longitude in [0, 360] and sort
    ssh = ssh.assign_coords(
        longitude=((ssh.longitude + 360) % 360)
    ).sortby("longitude")

    # ----------------------------------------
    # 4) Define longitude range in 0–360
    # ----------------------------------------
    def to_0360(lon):
        return (lon + 360) % 360

    lon_min_0360 = to_0360(lon_min)
    lon_max_0360 = to_0360(lon_max)

    def lon_in_range(lon_val):
        """Check if lon_val (0–360) lies within [lon_min, lon_max], allowing wrap."""
        if lon_min_0360 <= lon_max_0360:
            return (lon_val >= lon_min_0360) and (lon_val <= lon_max_0360)
        else:
            # wrap-around case, e.g. lon_min=350, lon_max=10
            return (lon_val >= lon_min_0360) or (lon_val <= lon_max_0360)

    # ----------------------------------------
    # 5) Open and subset combined GEM file
    # ----------------------------------------
    if not combined_gem_path.exists():
        raise FileNotFoundError(f"Combined GEM file not found: {combined_gem_path}")

    gem_combined = xr.open_dataset(combined_gem_path)

    # Ensure GEM longitudes are also 0..360 and sorted
    gem_combined = gem_combined.assign_coords(
        longitude=((gem_combined.longitude + 360) % 360)
    ).sortby("longitude")

    # Subset GEM longitudes to requested range (with wrap handling)
    gem_lons = gem_combined.longitude.values
    mask = np.array([lon_in_range(l) for l in gem_lons])
    gem_combined = gem_combined.isel(longitude=mask)

    # ----------------------------------------
    # 6) Main loop over longitude steps
    #    (time is handled implicitly by xarray)
    # ----------------------------------------
    satGEM_field = []
    satGEM_gamma_field_list = []

    for lon_step in tqdm(ssh.longitude.values, desc="Processing longitudes"):

        # Skip exactly 0 if you still want that behaviour
        if lon_step == 0:
            continue

        # Restrict to requested longitude range
        if not lon_in_range(lon_step):
            continue

        # SSH ±0.125° around current longitude (keeps time & lat dims)
        ssh_insitu_bb = ssh.sel(
            longitude=slice(lon_step - 0.125, lon_step + 0.125)
        ).adt

        # If everything is NaN here, skip
        if np.all(np.isnan(ssh_insitu_bb)):
            continue

        # Integer GEM longitude in 0..359
        gem_lon_int = int(np.floor(lon_step)) % 360

        # Get slope/intercept for this integer longitude
        slope, intercept = slopes_intercepts.get(gem_lon_int, (None, None))
        if slope is None or intercept is None or slope == 0:
            continue

        # If this longitude is not present in GEM_combined, skip
        if gem_lon_int not in gem_combined.longitude.values:
            continue

        # Slice the GEM fields for this longitude
        ts_gamma_field = gem_combined.sel(longitude=gem_lon_int)
        # Drop the scalar longitude coord to avoid conflict with SSH indexer dims
        ts_gamma_field = ts_gamma_field.reset_coords("longitude", drop=True)

        # ----------------------------------------
        # 7) Map dyn_m -> SSH using linear fit
        # ----------------------------------------
        ssh_GEM = (ts_gamma_field.dyn_m - intercept) / slope

        # Put everything on 'ssh' instead of 'dyn_m'
        t_s_field_ssh = (
            ts_gamma_field
            .assign_coords(ssh=ssh_GEM)
            .swap_dims({'dyn_m': 'ssh'})
            .drop_vars('dyn_m')
        )
 
        # ----------------------------------------
        # 8) Interpolate GEM onto observed SSH
        # ----------------------------------------
        satGEM_ts_field_lon = t_s_field_ssh.sel(ssh=ssh_insitu_bb, method='nearest')
        satGEM_ts_field_lon = satGEM_ts_field_lon.where(~np.isnan(ssh_insitu_bb), np.nan)

        satGEM_field.append(satGEM_ts_field_lon)

    if not satGEM_field:
        raise RuntimeError("No SatGEM fields were created – check date / lon / lat ranges and data availability.")

    # ----------------------------------------
    # 9) Concatenate TS fields along longitude
    # ----------------------------------------
    satGEM_ts_field = xr.concat(satGEM_field, dim='longitude')


    return satGEM_ts_field
