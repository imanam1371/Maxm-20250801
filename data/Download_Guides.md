# Downloading GPM IMERG **Final Run V07** data

An access to the **GPM IMERG Final Run (V07)** precipitation datasets from NASA GES DISC.

---

## What is IMERG Final V07?
**IMERG (Integrated Multi‑satellitE Retrievals for GPM)** merges precipitation estimates from the GPM constellation into globally gridded fields at **0.1° (~10 km)** resolution with **30‑min, daily, and monthly** products. The **Final Run** is the research‑grade product that applies gauge calibration and late-arriving inputs; it has the highest quality but **largest latency** (typically ~3.5 months). V07 reprocessed the full TRMM–GPM record back to **January 1998** and (as of V07B) provides **near‑global coverage to the poles** except for a few boxes at the highest latitudes.

Key characteristics (Final Run):
- **Spatial resolution:** 0.1° × 0.1°
- **Temporal resolutions:** 30‑min (3IMERGHH), **daily** (3IMERGDF), monthly (3IMERGM)
- **Coverage:** Jan 1998–present; global (with minor polar gaps)
- **Format:** HDF5 (also distributed via NetCDF4/OPeNDAP and GIS/GeoTIFF translations)
- **Core variables:** `precipitation` (a.k.a. `precipitationCal`), `precipitationQualityIndex`, `randomError`, and ancillary quality/flags
- **Units:**
  - Half‑hourly files: mm·hr⁻¹
  - **Daily files (3IMERGDF):** **mm·day⁻¹** 

---

## Prerequisites
1. **Create an Earthdata Login** (free): <https://urs.earthdata.nasa.gov/>
2. **Pre‑authorize the GES DISC application** so command‑line tools (wget/curl) can authenticate:
   - Log in → **Applications** → **Authorized Apps** → **Approve more applications** → search for and **approve** **“NASA GESDISC DATA ARCHIVE”**.
3. (Recommended) Set up standard auth files in your home directory:
   - `~/.netrc` (secure file with Earthdata username & password)
   - `~/.urs_cookies` (created/updated automatically during the first request)
   - `~/.dodsrc` (only if you’ll use OPeNDAP/netCDF tools)

> **Security tip:** set strict permissions on `~/.netrc`.

---

## Target dataset (daily Final Run)
- **GES DISC landing page:** <https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary>
- **What you get:** day‑by‑day grids of precipitation `mm/day` at 0.1° resolution.

Other resolutions:
- Half‑hourly: `GPM_3IMERGHH_07` (mm/hr)
- Monthly: `GPM_3IMERGM_07` (mm/day monthly mean)

---

## Recommended download workflow
You can either (A) generate a **link list** with the GES DISC Subsetter (most robust), or (B) download **direct files** if you already know the URLs.

### A) Subsetter + bulk download (robust)
1. Open the dataset page → **Subset / Get Data**.
2. **Select a time range** (avoid spatial subsetting if you’ve had broken links before).
3. Generate the request and click **Download Links** to save a plain‑text list (e.g., `urls.txt`).
4. Use `wget` or `curl` to download everything in the list.


**Windows (PowerShell or CMD):**
```bat
REM Adjust paths if you used a different drive/folder
wget --load-cookies C:\.urs_cookies ^
     --save-cookies C:\.urs_cookies ^
     --keep-session-cookies ^
     --user
     --ask-password
     --content-disposition ^
     -i urls.txt
```

**Notes**
- V07 daily (`3IMERGDF`) values are daily mean rates in **mm/day** (24× the mean mm/hr over the day), not a straight sum of valid half‑hours. This choice reduces dry bias on days with missing half‑hours.







# Downloading ERA5 Reanalysis Data

A guide to accessing **ERA5** reanalysis datasets from the Copernicus Climate Data Store (CDS) using the **cdsapi** Python client.

---

## What is ERA5?
ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate, providing hourly estimates of atmospheric, land and oceanic variables from 1940 to present.

Key characteristics:
- **Spatial resolution:** ~31 km (0.25° grid)
- **Temporal coverage:** 1940–present
- **Variables:** Surface, single‑level, and pressure‑level parameters (e.g., temperature, wind, humidity, radiation)
- **Format:** NetCDF (.nc) or GRIB (.grib)

---

## Prerequisites

1. **Create a Copernicus Climate Data Store (CDS) account:**  
   <https://cds.climate.copernicus.eu/#!/home> → click **Register** (top‑right).

2. **Install the CDS API client:**  
   ```bash
   pip install cdsapi
   ```

3. **Set up your API key:**  
   - Log in to the CDS website.  
   - Go to your [API key page](https://cds.climate.copernicus.eu/api-how-to).  
   - Copy the URL, key, and UID string.  

4. **Test authentication:**
   ```python
   import cdsapi
   cdsapi.Client()
   ```

---

## Access via the web interface (optional)

1. Visit a dataset page (e.g., ERA5 hourly single levels):  
   <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>
2. Configure variables, years, months, days, times, and area.
3. Select output format (NetCDF recommended for Python workflows).
4. Click **Show API request** to get the exact Python `cdsapi` code for your selection.

---

## Python API download examples

### Example 
```python
import cdsapi
c = cdsapi.Client()
years = range(1960, 1983)

variables = [
    '10m_u_component_of_wind', 
    '10m_v_component_of_wind',
    'geopotential',
    'surface_pressure',
    'convective_available_potential_energy',
    '2m_temperature',
    'sea_surface_temperature',
    'total_column_water_vapour',
    '2m_dewpoint_temperature',
    'volumetric_soil_water_layer_1',
    'total_column_cloud_liquid_water',
    'total_column_cloud_ice_water',
    'vertical_integral_of_eastward_water_vapour_flux',
    'vertical_integral_of_northward_water_vapour_flux',
    'toa_incident_solar_radiation',
    'convective_inhibition',
    'k_index',
    'mean_surface_latent_heat_flux',
    'mean_surface_sensible_heat_flux',
    'mean_vertically_integrated_moisture_divergence',
    'total_precipitation',
    'total_totals_index'
    ]

for y in years:
    for var in variables:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': var,
                'year': str(y),
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [49, 4, 43, 19]
            },
            f'path\\single_level_{var}_{y}.nc'
        )
```
## Variables used in extended workflows

ERA5 variables include:

| Short name | Units        | Description |
|------------|--------------|-------------|
| u10        | m/s          | Zonal wind at 10 m |
| v10        | m/s          | Meridional wind at 10 m |
| d2m        | K            | 2 m dewpoint temperature |
| t2m        | K            | 2 m air temperature |
| cape       | J/kg         | Convective available potential energy |
| kx         | K            | K‑index (thunderstorm potential) |
| mslhf      | W/m²         | Surface latent heat flux |
| msshf      | W/m²         | Surface sensible heat flux |
| mvimd      | kg/m²/s      | Moisture divergence |
| sp         | Pa           | Surface pressure |
| tisr       | J/m²         | TOA incident solar radiation |
| tclw       | kg/m²        | Cloud liquid water |
| tciw       | kg/m²        | Cloud ice water |
| tcwv       | kg/m²        | Total column water vapour |
| totalx     | K            | Total Totals index |
| viewv      | kg/m/s       | Eastward water vapour transport |
| vinwv      | kg/m/s       | Northward water vapour transport |
| swvl1      | m³/m³        | Volumetric soil water layer 1 |
| t (500 hPa)| K            | Temperature at 500 hPa |
| u (500 hPa)| m/s          | Zonal wind at 500 hPa |
| v (500 hPa)| m/s          | Meridional wind at 500 hPa |
| w          | Pa/s         | Vertical velocity |
| vo         | s⁻¹          | Vorticity |
| tp         | mm           | Total precipitation (re‑mapped) |

> Note: Pressure level variables require using `reanalysis-era5-pressure-levels` instead of `single-levels` in `c.retrieve()`.

---

## Citation
If you use ERA5, cite as:  
Hersbach, H., et al. (2020): The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. [doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803)

---

# Downloading ETOPO 2022 Global DEM (30 arc‑second Ice‑Surface NetCDF)

A science‑focused guide to retrieving the **ETOPO 2022** Ice‑Surface elevation dataset at 30‑arc‑second resolution from NOAA's NCEI, specifically the `ETOPO_2022_v1_30s_N90W180_surface.nc` global file.

---

## What is ETOPO 2022?

**ETOPO 2022** is NOAA’s latest global digital elevation model combining topography and bathymetry, distributed at high resolutions (15", 30", 60") in multiple surface definitions and formats (GeoTIFF and NetCDF).

The dataset provides continuous global coverage from –90° to +90° latitude and –180° to +180° longitude.

---

## Types of ETOPO 2022 Grids

ETOPO 2022 provides multiple "surface definitions" that differ in how ice-covered regions and sea surface are treated:

1. **Ice‑Surface (`*_surface.nc`)**  
   Elevation at the *top of the ice* in ice-covered regions (e.g., Greenland, Antarctica). Over the ocean, represents the mean sea surface. This is most suitable for atmospheric or climate models that interact with the actual upper surface seen by the atmosphere.

2. **Bedrock (`*_bed.nc`)**  
   Elevation of the *land or seabed* beneath the ice. This is useful for glaciology, hydrology, and subglacial studies, as it removes the ice thickness from the elevation.

3. **Geoid (`*_geoid.nc`)**  
   Height of the EGM2008 geoid relative to the WGS84 ellipsoid. This is not an elevation surface but the reference gravitational equipotential surface used to define "mean sea level".

---

## File Naming & Versions

Files follow the pattern:

```
ETOPO_2022_v<version>_<resolution>s_<Hem><Lat><HemLon><Lon>_<suffix>.nc
```

Example:  
`ETOPO_2022_v1_30s_N90W180_surface.nc` = Version 1, 30 arc‑second resolution, tile from N90 to W180, ice‑surface elevation.

---

## Download Methods

### 1. Direct download (browser)
- Visit: [NOAA NCEI ETOPO 2022 page](https://www.ncei.noaa.gov/products/etopo-global-relief-model)
- Locate **30 arc‑second Ice Surface** NetCDF link.
- Click to download or right‑click and **Copy Link Address**.

### 2. Command line with `wget`
Example:
```bash
wget "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/30s/30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc"
```

---

## File Details
- **Resolution:** 30 arc‑seconds (~1 km at equator)
- **Coverage:** global
- **Format:** netCDF4
- **Variable:** `surface` — elevation (m)

---

## Citation
If you use this dataset, cite as:

NOAA National Centers for Environmental Information. 2022: *ETOPO 2022 30 Arc‑Second Global Relief Model*. NOAA NCEI. DOI: [10.25921/fd45-gt74](https://doi.org/10.25921/fd45-gt74)

---

# HISTALP Climate Dataset — Station Selection & Download

A description of downloading a subset of stations from the **HISTALP** dataset provided by the **Central Institute for Meteorology and Geodynamics (ZAMG)**.

---

## What is HISTALP?

**HISTALP** (Historical Instrumental Climatological Surface Time Series of the Greater Alpine Region) is a dataset of meteorological observations from stations in and around the Alps, with long-term series dating back to the 18th and 19th centuries.

Key features:
- **Coverage:** Greater Alpine Region (~4–19°E, 43–49°N)
- **Variables:** Temperature, precipitation, pressure, sunshine, cloudiness, etc.
- **Temporal resolution:** Monthly (in most cases), with daily for certain variables/stations
- **Data format:** CSV (station-by-station), also available in NetCDF for gridded products

Official website: <https://www.zamg.ac.at/histalp/>

---

## Data Access

Station data can be accessed directly via CSV download.  

---

## Selected 50 Stations

From the full HISTALP station list, 50 were randomly selected for analysis:

```
AT_BMU, AT_FRE, AT_KOL, AT_LAD, AT_SPO, AT_WIE,
BA_JAJ, DE_KMF, DE_LHU, DE_ULM, HR_POZ, HR_ZAD, HU_PAP,
IT_BAR, IT_FOR, IT_IMP, IT_MIL, IT_UDI, IT_VAL, SI_LJU,
AT_ADM, SI_CEL, IT_ROV, IT_BRX, DE_OBS, IT_TOM,
IT_BOZ, HU_SZO, HR_KVA, DE_STU, DE_ROS, IT_PAR,
BA_BIH, AT_WAI, AT_TAM, AT_SSB, AT_KAL, IT_VEN,
AT_SAL, AT_RAU, AT_OFK, AT_LAG, AT_KRE, AT_KOR,
AT_BST, AT_BRE, AT_FLA, AT_INN, AT_VIL, AT_RET
```
---

## Notes
- **Missing values:** encoded as `-999` in HISTALP CSVs — handle appropriately in processing.
---
