import os
import numpy as np
import xarray as xr
from tqdm.auto import tqdm
import copernicusmarine
from pathlib import Path
import urllib.request
import tempfile

# -------------------------------------------------------------------
# Helper: resolve local path OR URL (Zenodo) to a local file
# -------------------------------------------------------------------
def resolve_path(path):
    """
    Accepts either:
      - local filepath
      - HTTP/HTTPS URL (e.g. Zenodo link)
    Returns a local Path to the actual file.
    """
    path = str(path)

    # URL case
    if path.startswith("http://") or path.startswith("https://"):
        tmp_dir = Path(tempfile.gettempdir())
        cleaned_name = Path(path).name.split("?")[0]  # remove ?download=1 etc.
        local_file = tmp_dir / cleaned_name

        if not local_file.exists():
            print(f"Downloading {path} → {local_file}")
            urllib.request.urlretrieve(path, local_file)
        else:
            print(f"Using cached file: {local_file}")

        return local_file

    # Local file
    return Path(path)


# -------------------------------------------------------------------
# Small helpers (pure, no hard-coded paths)
# -------------------------------------------------------------------
def _to_180(lon):
    """Convert [0, 360) to [-180, 180]."""
    lon = float(lon)
    return lon - 360.0 if lon > 180 else lon


def _wrap180_int(k):
    """Wrap integer longitude to [-180, 180] with 180 included."""
    if k > 180:
        k -= 360
    if k < -180:
        k += 360
    return int(k)


def _subset_vars(GEM, vars_to_keep):
    """Keep only requested data variables that exist in this GEM dataset."""
    if vars_to_keep is None:
        return GEM
    present = [v for v in vars_to_keep if v in GEM.data_vars]
    if len(present) == 0:
        # nothing matches; just return as-is to avoid empty datasets
        return GEM
    return GEM[present]


def _open_gem_as_adt_nointerp(lon_int, gem_all, lut_all, vars_to_keep, month_da=None):
    """
    Select one longitude slice from a combined GEM dataset and
    use the corresponding LUT (single file with dims like dyn_m, longitude)
    to swap 'dyn_m' -> 'adt'.

    Parameters
    ----------
    lon_int : int
        Integer longitude ([-180, 180]) for this "post".
    gem_all : xr.Dataset
        Combined GEM dataset with longitude, month, dyn_m, ...
    lut_all : xr.Dataset
        Single LUT dataset with the dyn_m ↔ adt relationship for ALL longitudes.
        Expected to have at least dims ('dyn_m', 'longitude') and variables
        'dyn_m' and 'adt'.
    vars_to_keep : list of str or None
        Variables to retain from GEM.
    month_da : xr.DataArray or None
        Optional month indexer (1–12) with a 'time' dimension.
    """
    # Basic checks
    if "longitude" not in gem_all.coords or "longitude" not in lut_all.coords:
        return None

    # Select this longitude from GEM (exact integer match)
    try:
        GEM = gem_all.sel(longitude=lon_int)
    except KeyError:
        return None

    # Select this longitude from LUT
    try:
        LUT = lut_all.sel(longitude=lon_int)
    except KeyError:
        return None

    dyn_m = LUT["dyn_m"]
    adt1d = LUT["adt"]

    # Handle case where LUT has one more dyn_m point than GEM
    if "dyn_m" in GEM.dims and dyn_m.size > GEM.dims["dyn_m"]:
        dyn_m = dyn_m.isel(dyn_m=slice(0, GEM.dims["dyn_m"]))
        adt1d = adt1d.isel(dyn_m=slice(0, GEM.dims["dyn_m"]))

    # Make sure dyn_m, adt1d are 1D along 'dyn_m'
    if dyn_m.ndim != 1:
        # collapse any extra dims except 'dyn_m' (if they exist)
        for extra_dim in dyn_m.dims:
            if extra_dim != "dyn_m":
                dyn_m = dyn_m.isel({extra_dim: 0})
                adt1d = adt1d.isel({extra_dim: 0})

    # Assign dyn_m coord and swap to adt dimension
    GEM = GEM.assign_coords(dyn_m=("dyn_m", dyn_m.data))
    GEM = GEM.assign_coords(adt=("dyn_m", adt1d.data)).swap_dims({"dyn_m": "adt"})

    # ---- month selection ----
    if "month" in GEM.dims:
        if month_da is None:
            # default: always January
            GEM = GEM.sel(month=1)
        else:
            # advanced indexing: month_da has a 'time' dim, GEM has 'month';
            # xarray broadcasts and replaces 'month' with 'time'.
            GEM = GEM.sel(month=month_da)

    # subset variables
    GEM = _subset_vars(GEM, vars_to_keep)

    return GEM



# -------------------------------------------------------------------
# Main public function
# -------------------------------------------------------------------
def create_bgc_satGEM(
    dates,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    gem_path,
    lut_path,   # <--- was lut_dir
    dataset_id="c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
    vars_to_keep=None,
):
    ...
    # ---------------------------------------------------------------
    # 0. Open the combined GEM file (local path OR Zenodo URL)
    # ---------------------------------------------------------------
    gem_path = resolve_path(gem_path)
    gem_all = xr.open_dataset(gem_path)

    # Ensure longitudes are in [-180, 180] to match k = floor(lon_180)
    if "longitude" in gem_all.coords:
        gem_all = gem_all.assign_coords(
            longitude=np.vectorize(_to_180)(gem_all.longitude.values)
        ).sortby("longitude")

    # ---------------------------------------------------------------
    # 0b. Open the single LUT file (local or URL)
    # ---------------------------------------------------------------
    lut_path = resolve_path(lut_path)
    lut_all = xr.open_dataset(lut_path)

    # Make sure LUT longitudes are also in [-180, 180]
    if "longitude" in lut_all.coords:
        lut_all = lut_all.assign_coords(
            longitude=np.vectorize(_to_180)(lut_all.longitude.values)
        ).sortby("longitude")

    # ---------------------------------------------------------------
    # 1. Get SSH from Copernicus over requested region / time
    # ---------------------------------------------------------------
    ssh = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=["adt"],
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    ds_SO = ssh["adt"] # DataArray: time × lat × lon

    # ---------------------------------------------------------------
    # 2. Group longitude columns into "posts" k = floor(lon_180)
    # ---------------------------------------------------------------
    cop_lons = ds_SO.longitude.values
    cop_lons_180 = np.vectorize(_to_180)(cop_lons)
    lon_keys = np.floor(cop_lons_180).astype(int)  # k = floor(lon_180)
    unique_keys = np.unique(lon_keys)

    tiles = []

    # ---------------------------------------------------------------
    # 3. Loop over each longitude "post"
    # ---------------------------------------------------------------
    for k in tqdm(unique_keys, desc="Building BGC SatGEM"):
        cols_mask = lon_keys == k
        if not cols_mask.any():
            continue

        print(f"Processing lon-post {k}° ...")

        lon_subset = cop_lons[cols_mask]
        # Subset SSH for this group of longitudes
        cop_tile = ds_SO.sel(longitude=xr.DataArray(lon_subset, dims="longitude"))

        # If SSH has a time dimension, build a month indexer (1–12) from it.
        month_da = None
        if "time" in cop_tile.dims:
            month_da = cop_tile["time"].dt.month

                # Open the three posts, with month selection if available
        Gc = _open_gem_as_adt_nointerp(
            k, gem_all, lut_all, vars_to_keep, month_da=month_da
        )
        Gw = _open_gem_as_adt_nointerp(
            _wrap180_int(k - 1), gem_all, lut_all, vars_to_keep, month_da=month_da
        )
        Ge = _open_gem_as_adt_nointerp(
            _wrap180_int(k + 1), gem_all, lut_all, vars_to_keep, month_da=month_da
        )

        if Gc is None and Gw is None and Ge is None:
            print(f"  -> Skipping {k} (no usable GEMs)")
            continue

        # -----------------------------------------------------------
        # 4. Sample each available post at Copernicus ADT (nearest)
        # -----------------------------------------------------------
        samples = []
        weights = []

        if Gw is not None:
            S_w = Gw.sel(adt=cop_tile, method="nearest")
            samples.append(S_w)
            weights.append(0.25)
        if Gc is not None:
            S_c = Gc.sel(adt=cop_tile, method="nearest")
            samples.append(S_c)
            weights.append(0.5)
        if Ge is not None:
            S_e = Ge.sel(adt=cop_tile, method="nearest")
            samples.append(S_e)
            weights.append(0.25)

        w = np.array(weights, float)
        w /= w.sum()  # renormalize if any neighbor missing

        stacked = xr.concat(samples, dim="__m__")
        sampled = (stacked * xr.DataArray(w, dims="__m__")).sum("__m__")

        # Mask with Copernicus NaNs
        sampled = sampled.where(np.isfinite(cop_tile))

        # Ensure correct lat / lon coords (time comes from cop_tile via .sel)
        sampled = sampled.assign_coords(
            latitude=cop_tile.latitude,
            longitude=cop_tile.longitude,
        )

        tiles.append(sampled)

    # ---------------------------------------------------------------
    # 5. Stitch tiles together in longitude and sort
    # ---------------------------------------------------------------
    if len(tiles) == 0:
        raise ValueError("No valid GEM tiles were produced – check paths and region.")

    satGEM_field = xr.concat(tiles, dim="longitude").sortby("longitude")

    return satGEM_field
