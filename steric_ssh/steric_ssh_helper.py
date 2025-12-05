import numpy as np
import xarray as xr
import copernicusmarine
import subprocess
from pathlib import Path


def download_nc_curl(url, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if not local_path.exists():
        cmd = [
            "curl",
            "-k",        # skip SSL certificate verification
            "-L", url,   # follow redirects
            "-o", str(local_path)
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("Using cached file:", local_path)

    return local_path


def open_grace_local(url, local_dir):
    filename = url.split("/")[-1]
    local_path = download_nc_curl(url, Path(local_dir) / filename)
    return xr.open_dataset(local_path)



# -------------------------------------------------------------------
# Helper: make sure fields use (time, latitude, longitude)
# -------------------------------------------------------------------
def to_latlon(da):
    """
    Ensure DataArray has dims (time, latitude, longitude)
    and only 'latitude' and 'longitude' as spatial coords.
    """
    da = da.copy()

    # Rename dims if needed
    if "lon" in da.dims:
        da = da.rename({"lon": "longitude"})
    if "lat" in da.dims:
        da = da.rename({"lat": "latitude"})

    # Drop any leftover coord variables named 'lat'/'lon'
    for old in ["lat", "lon"]:
        if old in da.coords:
            da = da.drop_vars(old)

    return da


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def compute_steric_ssh(
    grace_dir,
    early_path,
    mid_path,
    mdt_path,
    ssh_baseline_path,
    # spatial domain
    lon_min=0.0,
    lon_max=360.0,
    lat_min=-80.0,
    lat_max=-35.0,
    # time domain
    start_datetime="2020-01-01",
    end_datetime="2025-04-01",
    # names of variables in the path files
    early_var="grace_on_ssh_time",
    mid_var="grace_on_ssh_time",
    mdt_var="adt",
    ssh_baseline_var="adt",
    # GRACE URLs and altimetry dataset
    url_atmospheric="https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_GAD-component.nc",
    url_grace_all="https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
    copernicus_dataset_id="c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1M-m",
):
    """
    Compute steric SLA and steric SSH over a chosen region and time range.

    Returns an xarray.Dataset with:
      - sla          : dynamic SLA (SSH - MDT)
      - steric_sla   : SLA minus GRACE barystatic (i.e. steric contribution)
      - steric_ssh   : steric_sla + baseline SSH
      - barystatic   : GRACE barystatic water height on SLA time axis
    """

    # ----------------------------------------------------------
    # 1) Load GRACE mascons (atmospheric + all corrections)
    # ----------------------------------------------------------
    atmospheric_GRACE = open_grace_local(url_atmospheric, grace_dir).sel(
        lat=slice(lat_min, lat_max)
    )
    barystatic = open_grace_local(url_grace_all, grace_dir).sel(
        lat=slice(lat_min, lat_max)
    )

    # Convert GRACE time from days since 2002-01-01 to DatetimeIndex
    base_time = np.datetime64("2002-01-01")
    atmospheric_GRACE = atmospheric_GRACE.assign_coords(
        time=base_time + atmospheric_GRACE.time.astype("timedelta64[D]")
    )
    barystatic = barystatic.assign_coords(
        time=base_time + barystatic.time.astype("timedelta64[D]")
    )

    # Convert to monthly periods (month start)
    time_index = barystatic.indexes["time"]  # pandas.DatetimeIndex
    new_time = time_index.to_period("M").to_timestamp(how="start")

    barystatic = barystatic.assign_coords(time=("time", new_time.to_numpy()))
    atmospheric_GRACE = atmospheric_GRACE.assign_coords(time=("time", new_time.to_numpy()))

    # Correct barystatic for atmospheric component (cm → m)
    barystatic_corrected = (barystatic.lwe_thickness - atmospheric_GRACE.lwe_thickness) / 100.0
    corr_da = to_latlon(barystatic_corrected)

    # ----------------------------------------------------------
    # 2) Load pre-processed early & mid GRACE-on-SSH-time fields
    # ----------------------------------------------------------
    early_ds = xr.open_dataset(early_path)
    mid_ds   = xr.open_dataset(mid_path)

    early_da = to_latlon(early_ds[early_var])
    mid_da   = to_latlon(mid_ds[mid_var])

    # Full barystatic time period
    barystatic_full_timeperiod = xr.concat(
        [early_da, corr_da, mid_da], dim="time"
    ).sortby("time")

    # ----------------------------------------------------------
    # 3) Load MDT and baseline SSH
    # ----------------------------------------------------------
    mdt_da = xr.open_dataset(mdt_path).__xarray_dataarray_variable__
    mdt_da = to_latlon(mdt_da)

    ssh_baseline_da = xr.open_dataset(ssh_baseline_path)[ssh_baseline_var]
    ssh_baseline_da = to_latlon(ssh_baseline_da)

    # Restrict MDT & baseline to region if needed
    mdt_da = mdt_da.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )
    ssh_baseline_da = ssh_baseline_da.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )

    # ----------------------------------------------------------
    # 4) Load Copernicus monthly SLA
    # ----------------------------------------------------------
    ssh_monthly = copernicusmarine.open_dataset(
        dataset_id=copernicus_dataset_id,
        variables=["sla"],
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    # Put longitudes on [0, 360)
    ssh_monthly = ssh_monthly.assign_coords(
        longitude=((ssh_monthly.longitude + 360) % 360)
    ).sortby("longitude")

    # ----------------------------------------------------------
    # 5) Build SSH, SLA, and align fields
    # ----------------------------------------------------------
    # MDT → same lon/lat grid as SLA
    mdt_da = mdt_da.sel(
        longitude=ssh_monthly.longitude,
        latitude=ssh_monthly.latitude,
    )

    # Baseline SSH: same grid as SLA
    ssh_baseline_da = ssh_baseline_da.sel(
        longitude=ssh_monthly.longitude,
        latitude=ssh_monthly.latitude,
    )

    # Total SSH (dynamic) and SLA relative to baseline
    ssh_full = ssh_monthly.sla + mdt_da
    sla = ssh_full - ssh_baseline_da

    # ----------------------------------------------------------
    # 6) Put GRACE barystatic onto the SLA time axis
    # ----------------------------------------------------------
    # First, restrict GRACE to the SLA time range
    t0 = sla.time.min()
    t1 = sla.time.max()
    barystatic_sub = barystatic_full_timeperiod.sel(time=slice(t0, t1))

    # Reindex GRACE to exactly SLA's time stamps (nearest month)
    # Assumes both are monthly time series
    barystatic_on_sla_time = barystatic_sub.reindex(
        time=sla.time, method="nearest"
    )

    # Align spatially
    barystatic_on_sla_time = barystatic_on_sla_time.sel(
        latitude=sla.latitude,
        longitude=sla.longitude,
    )

    # ----------------------------------------------------------
    # 7) Steric SLA and steric SSH
    # ----------------------------------------------------------
    steric_sla = sla - barystatic_on_sla_time
    steric_ssh = steric_sla + ssh_baseline_da

    # Package into a Dataset
    out = xr.Dataset(
        data_vars=dict(
            steric_ssh=steric_ssh,
        )
    )

    return out
