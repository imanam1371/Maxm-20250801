
# ML‑IMEX (Machine Learning IMERG backward‑EXtended)

> **Goal:** Reproduce the dataset and analyses described in the paper “A Machine Learning‑Based Backward Extension of IMERG Daily Precipitation over the Greater Alpine Region (GAR)” by building an XGBoost model that learns from ERA5 predictors to reconstruct IMERG‑like daily precipitation, selecting features with SHAP, and validating against HISTALP/CHAPTER.  
> **Spatial domain:** 4.05°E–18.95°E, 43.05°N–48.95°N (GAR) on a 0.1° grid (≈10 km).  
> **Temporal coverage for training/evaluation:** 2001‑01‑01 to 2020‑12‑31 (leap days removed).  
> **Historical reconstruction:** 1960‑01‑01 to 2000‑12‑31.

---
## 0) Environment setup
---

## 1) Data acquisition
Please refer to Download_Guides.md in the data folder.
### 1.1 ERA5 (single levels + pressure levels)
### 1.2 IMERG Final Run V07 (0.1°, 30‑min → daily)
**Quality Index:** Acquire the QI (V06/QId proxy if needed) and compute the **daily mean QIh**, then keep only days/pixels where **QI ≥ 0.6**.
### 1.3 ETOPO 2022 (for elevation)
### 1.4 HISTALP stations (for validation)
### 1.5 CHAPTER (optional validation)
---

## 2) Preprocessing & grid harmonization
Please refer to CDO_processing.md in the data folder.
### 2.1 Aggregate and harmonize time
### 2.2 Put everything on the IMERG 0.1° grid
- **Nearest‑neighbour** for all ERA5 predictors (except precipitation).
- **Conservative remapping** for ERA5 precipitation (tp → ppt‑era5) to preserve integrals.
### 2.3 Compute derived variables
We compute the four extra predictors defined in the Appendix:
- **q2m** from `d2m` and `sp` 
- **CAMR** ≈ `tcwv / (sp/9.81)`
- **Balk shear (BS)** from wind at 925 and 500 hPa
- **Water Vapor Transport speed (WVT)** = `upslope_wind × q2m`; with `upslope_wind = U10 * dh/dx + V10 * dh/dy`

---

## 3) Imports all libraries, Datasets
### 3.1 Imports all libraries

```python
#Data_Handling_Libraries
import os                                                      # File system operations
import json                                                    # Working with JSON files
import pandas as pd                                            # Tabular data handling
import numpy as np                                             # Numerical operations
import xarray as xr                                            # Multi-dimensional labeled data
import dask                                                    # Parallel computing and lazy evaluation
import netCDF4 as nc                                           # Reading netCDF files (common in climate science)

#Machine_Learning_Libraries
from sklearn.metrics import mean_squared_error, r2_score       # Model evaluation metrics
from sklearn.model_selection import train_test_split           # Splitting data
from xgboost import XGBRegressor                               # XGBoost regressor
import shap                                                    # SHAP values for model explainability
import joblib                                                  # Model saving/loading

#Visualization_Libraries
import matplotlib.pyplot as plt                                # Plotting library
from matplotlib import ticker                                  # Customizing axis ticks
import matplotlib.colors as mcolors                            # Handling color maps
import seaborn as sns                                          # Statistical data visualization
 
#Geospatial_Tools
import cartopy.feature as cfeature                             # Map features like coastlines, borders
import cartopy.crs as ccrs                                     # Coordinate reference systems
import rioxarray as rxr                                        # Spatial raster data handling with xarray
from rasterio.merge import merge                               # Merge multiple raster files
from rasterio.plot import show                                 # Plotting raster data
from geopy.distance import geodesic                            # Compute geographic distance

#Utility_Libraries
from tqdm import tqdm                                          # Progress bars for loops
import random                                                  # Random number generation
```

---
### 3.2 Import Data

**What this does:**  
- Sets the working path and the year range.  
- Toggles the SHAP analysis flag to determine which variables to include.

```python
<h1>Import Data</h1>
path = 'C:/Iman/Maxm'  #path = '/leonardo_work/mBI23_AmbCo_1/igoudarz'
iyear = list(range(2001, 2021))
#Are you going to do SHAP Analysis?
shap = "NO"  # or "YES"

if shap == "NO":
    var_surface_input_list = ['total_column_cloud_ice_water_daymean','total_precipitation_daycum','camr_daymean']
else:
    var_surface_input_list = [
         'total_column_cloud_ice_water_daymean',
         'total_precipitation_daycum',
         'camr_daymean',
         
         '10m_u_component_of_wind_daymean',
         '10m_v_component_of_wind_daymean',
         '2m_dewpoint_temperature_daymean',
         '2m_temperature_daymax',
         'convective_available_potential_energy_daymax',
         'geopotential_daymean',
         'k_index_daymean',
         'mean_surface_latent_heat_flux_daymean',
         'mean_surface_sensible_heat_flux_daymean',
         'mean_vertically_integrated_moisture_divergence_daymean',
         'sea_surface_temperature_daymean',
         'surface_pressure_daymean',
         'toa_incident_solar_radiation_daymean',
         'total_column_cloud_liquid_water_daymean',
         'total_column_water_vapour_daymean',
         'total_totals_index_daymean',
         'vertical_integral_of_eastward_water_vapour_flux_daymean',
         'vertical_integral_of_northward_water_vapour_flux_daymean',
         'volumetric_soil_water_layer_1_daymean',
         'temperature_daymean',
         'u_component_of_wind_daymean',
         'v_component_of_wind_daymean',
         'vertical_velocity_daymean',
         'vorticity_daymean',
         'q2m_daymean',
         'bs_daymax',
         'WVT_daymean']
```

### 3.3 ERA5 (2001–2021)

**What this does:**  
- Reads each selected variable for each year, drops `bnds` dimension, and merges by variable and across variables.  
- Loads conservative-remapped ERA5 precipitation, converts from meters to **mm**, and merges with the main dataset.  
- Applies basic cleaning (non-negative precipitation and tciw), rounds coordinates, and renames/scales additional fields.  
- Adds units metadata and optional SHAP-specific renaming for variable keys.

```python
# List to store the merged datasets for each variable
merged_datasets = []
for var in var_surface_input_list:
    datasets = []
    for year in iyear:
        # Open the dataset for the current variable and year
        ds = xr.open_dataset(f"{path}/ERA5_NearestN/{var}_{year}_NearestN.nc")
        ds = ds.isel(bnds=0)
        datasets.append(ds)
    # Merge all datasets for the current variable
    merged_var_dataset = xr.merge(datasets)
    merged_datasets.append(merged_var_dataset)
# Combine all merged variable datasets into a single xarray.Dataset
era5_Imerg = xr.merge(merged_datasets)
#######################################################################
#Importing era5 conservative remapping dataset
merged_datasets = []

datasets = []
for year in iyear:
    # Open the dataset for the current variable and year
    ds = xr.open_dataset(f"{path}/ERA5_consmap/total_precipitation_daycum_{year}_consmap.nc")
    ds = ds.isel(bnds=0)
    datasets.append(ds)

# Merge all datasets for the current variable
merged_var_dataset = xr.merge(datasets)
merged_datasets.append(merged_var_dataset)

# Combine all merged variable datasets into a single xarray.Dataset
era5_ppt_era5 = xr.merge(merged_datasets)
era5_ppt_era5 = era5_ppt_era5.rename({'tp': 'ppt_era5'})
era5_ppt_era5['ppt_era5']=era5_ppt_era5['ppt_era5']*1000 
#######################################################################
#merge with main DS
era5_Imerg = xr.merge([era5_Imerg, era5_ppt_era5], compat='override')
era5_Imerg = era5_Imerg.drop_vars('time_bnds')
#exclude data less than 0
era5_Imerg["ppt_era5"] = xr.where(era5_Imerg["ppt_era5"] > 0, era5_Imerg["ppt_era5"], 0)
era5_Imerg["tciw"] = xr.where(era5_Imerg["tciw"] > 0, era5_Imerg["tciw"], 0)
#######################################################################
#furthur modifications
era5_Imerg['lat'] = era5_Imerg['lat'].round(2).astype(float)
era5_Imerg['lon'] = era5_Imerg['lon'].round(2).astype(float)

era5_Imerg = era5_Imerg.rename({'tp': 'tp_NearestN'})
era5_Imerg['tp_NearestN'] = era5_Imerg['tp_NearestN'] * 1000

if shap != "NO":
    era5_Imerg = era5_Imerg.rename({'z': 'geopotential/9.81'})  
    era5_Imerg['geopotential/9.81'] = era5_Imerg['geopotential/9.81'] / 9.81  

    era5_Imerg = era5_Imerg.rename({'p71.162': 'viewvf'})  
    era5_Imerg = era5_Imerg.rename({'p72.162': 'vinwvf'})  
    pass  

#Units
era5_Imerg["ppt_era5"].attrs["units"] = "mm" 
era5_Imerg["camr"].attrs["units"] = "unitless"
era5_Imerg["tciw"].attrs["units"] = "kg m**-2"

if shap != "NO":
    era5_Imerg["cape"] = xr.where(era5_Imerg["cape"] > 0, era5_Imerg["cape"], 0)  
    era5_Imerg["tp_NearestN"].attrs["units"] = "mm"       
    era5_Imerg["bs"].attrs["units"] = "m s**-1"           
    era5_Imerg["cape"].attrs["units"] = "J kg**-1"
    era5_Imerg["q2m"].attrs["units"] = "kg/kg"            
    era5_Imerg["WVT"].attrs["units"] = "m s**-1"  
    pass  

```

### 3.4 ERA5 Back-in-Time (1960–2000)

**What this does:**  
- Loads two periods (1960–1983, 1984–2000) for selected variables with consistent time coordinate names and merges them.  
- Handles precipitation separately, converts to **mm/day**, and merges periods.  
- Brings in ETOPO elevation, replicates over time for both datasets, masks negatives to 0 (sea level).  
- Removes Feb 29 for a consistent 365-day calendar, rounds coordinates, and tidies attributes.

```python
selected_variables_backintime= ['camr', 'total_column_cloud_ice_water']
############1960-1983################
merged_datasets = []
for var in selected_variables_backintime:
    datasets = []
    for year in range (1960,1984): 
        ds=xr.open_dataset(f"{path}/ERA5_hourly_backintime/{var}_{year}_daymean_NN.nc")
        ds = ds.isel(bnds=0)
        ds = ds.rename({"valid_time": "time"})
        datasets.append(ds)
    # Merge all datasets for the current variable
    merged_var_dataset = xr.merge(datasets)
    merged_datasets.append(merged_var_dataset)
    
# Combine all merged variable datasets into a single xarray.Dataset
era5_backintime_1960_1983 = xr.merge(merged_datasets)
############1983-2000################
merged_datasets = []
for var in selected_variables_backintime:
    datasets = []
    for year in range (1984,2001): 
        ds=xr.open_dataset(f"{path}/ERA5_NearestN/{var}_daymean_{year}_NearestN.nc") #,chunks={"time": 1}
        ds = ds.isel(bnds=0)
        #ds = ds.rename({"valid_time": "time"})
        datasets.append(ds)
    # Merge all datasets for the current variable
    merged_var_dataset = xr.merge(datasets)
    merged_datasets.append(merged_var_dataset)
    
# Combine all merged variable datasets into a single xarray.Dataset
era5_backintime_1984_2000 = xr.merge(merged_datasets)
############1960-2000################
era5_backintime = xr.merge([era5_backintime_1960_1983, era5_backintime_1984_2000]) #, compat='override'
    
############1960-1983################
merged_datasets = []
for var in ['total_precipitation']:
    datasets = []
    for year in range (1960,1984):
        ds=xr.open_dataset(f"{path}/ERA5_hourly_backintime/{var}_{year}_daysum_consmap.nc")
        ds = ds.isel(bnds=0)
        ds = ds.rename({"valid_time": "time"})
        ds = ds.rename({'tp': 'ppt_era5'})
        ds['ppt_era5']=ds['ppt_era5']*1000   #convert to mm/d
        ds["ppt_era5"].attrs["units"] = "mm"
        datasets.append(ds)
    # Merge all datasets for the current variable
    merged_var_dataset = xr.merge(datasets)
    merged_datasets.append(merged_var_dataset)
# Combine all merged variable datasets into a single xarray.Dataset
era5_tp_backintime_1960_1983 = xr.merge(merged_datasets)
############1984-2000################
merged_datasets = []
for var in ['total_precipitation']:
    datasets = []
    for year in range (1984,2001):
        ds=xr.open_dataset(f"{path}/ERA5_consmap/{var}_daycum_{year}_consmap.nc")
        ds = ds.isel(bnds=0)
        #ds = ds.rename({"valid_time": "time"})
        ds = ds.rename({'tp': 'ppt_era5'})
        ds['ppt_era5']=ds['ppt_era5']*1000   #convert to mm/d
        ds["ppt_era5"].attrs["units"] = "mm"
        datasets.append(ds)
    # Merge all datasets for the current variable
    merged_var_dataset = xr.merge(datasets)
    merged_datasets.append(merged_var_dataset)
# Combine all merged variable datasets into a single xarray.Dataset
era5_tp_backintime_1984_2000 = xr.merge(merged_datasets)
############1960-2000################
era5_tp_backintime = xr.merge([era5_tp_backintime_1960_1983, era5_tp_backintime_1984_2000]) #, compat='override'
############IMPORT ETOPO################# 
ds = xr.open_dataset(f"{path}/ETOPO_2022_v1_30s_N90W180_surface_Imerg_conservative_remapped.nc", engine="netcdf4")
# Select the latitude and longitude ranges
ETOPO_consmap_Z = ds.sel(lat=slice(43, 49), lon=slice(4, 19))

expanded_z = xr.concat([ETOPO_consmap_Z['z']] * len(era5_Imerg.time), dim='time')
# Create a new DataArray 
new_data_array = xr.DataArray(
    expanded_z,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': era5_Imerg.coords['time'],
        'lat': era5_Imerg.coords['lat'],
        'lon': era5_Imerg.coords['lon']})

# Assign the new DataArray to the era5 dataset
era5_Imerg['ETOPO_consmap_z'] = new_data_array
era5_Imerg['ETOPO_consmap_z'] = era5_Imerg['ETOPO_consmap_z'].where(era5_Imerg['ETOPO_consmap_z'] >= 0, 0)
############################################################################################################
# Repeat the z data along the time dimension to match the length of era5_data_daily time dimension
expanded_z = xr.concat([ETOPO_consmap_Z['z']] * len(era5_tp_backintime.time), dim='time')
# Create a new DataArray 
new_data_array = xr.DataArray(
    expanded_z,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': era5_tp_backintime.coords['time'],
        'lat': era5_tp_backintime.coords['lat'],
        'lon': era5_tp_backintime.coords['lon']})

# Assign the new DataArray to the era5 backintime dataset
era5_tp_backintime['ETOPO_consmap_z'] = new_data_array
era5_tp_backintime['ETOPO_consmap_z'] = era5_tp_backintime['ETOPO_consmap_z'].where(era5_tp_backintime['ETOPO_consmap_z'] >= 0, 0)
era5_backintime = xr.merge([era5_backintime, era5_tp_backintime]) #, compat='override'
#removing 29th of Feb to have a same data for all years
era5_backintime = era5_backintime.sel(time=~((era5_backintime['time'].dt.month == 2) & (era5_backintime['time'].dt.day == 29)))
era5_backintime["camr"].attrs["units"] = "unitless"
era5_backintime = era5_backintime.drop_vars('valid_time_bnds')
# Round to 6 decimal places
era5_backintime['lat'] = era5_backintime['lat'].round(2).astype(float)
era5_backintime['lon'] = era5_backintime['lon'].round(2).astype(float)

```

### 3.5 IMERG Dataset (2001–2020)

**What this does:**  
- Loads daily IMERG precipitation files lazily using Dask, concatenates by time, and aligns dimensions.  
- Assigns IMERG precipitation into the ERA5 dataset, removes Feb 29, creates a custom 365-day index, and sets units.

```python
dates = pd.date_range(start="2001-01-01", end="2020-12-31", freq="D")
# Enable dask to manage datasets lazily
merged_datasets = []
for date in dates:
    try:
        i = date.strftime("%Y%m%d")  
        # Open the dataset with dask for lazy loading
        ds = xr.open_dataset(f"{path}/IMERG7_CDO/CDO_Precipitation_3B-DAY.MS.MRG.3IMERG.{i}-S000000-E235959.V07B.nc4", 
                             engine="netcdf4", chunks={'time': 1}) 
        ds = ds['precipitation']
        merged_datasets.append(ds)
    except FileNotFoundError:
        print(f"File not found for {i}, skipping...")
# Merge datasets lazily
Imerg_orgres_dask = xr.concat(merged_datasets, dim='time')
Imerg_orgres_dask=Imerg_orgres_dask.transpose('time', 'lat', 'lon')
# Create a new DataArray for train/test indices
era5_Imerg['Imerg_orgres'] = xr.DataArray(
    Imerg_orgres_dask.compute().values,
    dims=['time', 'lat','lon'],
    coords={
        'time': era5_Imerg.coords['time'],
        'lat': era5_Imerg.coords['lon'],
        'lon': era5_Imerg.coords['lat']})

#Other Modifications
#removing 29th of Feb to have a same data for all years
era5_Imerg = era5_Imerg.sel(time=~((era5_Imerg['time'].dt.month == 2) & (era5_Imerg['time'].dt.day == 29)))
# Create a custom list of 365 day indices (or labels)
custom_dayofyear = np.tile(np.arange(1, 366), len(np.unique(era5_Imerg['time'].dt.year.values)))
# Assign this custom day index as a new coordinate
era5_Imerg = era5_Imerg.assign_coords(custom_dayofyear=("time", custom_dayofyear))
#assign the unit to IMERG pp
era5_Imerg["Imerg_orgres"].attrs["units"] = "mm"
```

### 3.6 IMERG Quality Index

**What this does:**  
- Loads hourly IMERG quality index (QI) files, drops time bounds and rounds coordinates.  
- Resamples to daily mean, aligns time with ERA5 (starting 2001), removes Feb 29, creates a 365-day index, and merges into main dataset.

```python
#Import QI data
file_list = [os.path.join(path, "QI_CDO", f) for f in os.listdir(os.path.join(path, "QI_CDO")) if f.endswith('.nc4')]
ds_QI_hour = xr.open_mfdataset(file_list,combine='by_coords',chunks={'time': 24}, parallel=True,cache=False)
############################################################
ds_QI_hour=ds_QI_hour.drop_vars('time_bnds')
ds_QI_hour=ds_QI_hour.compute()
ds_QI_hour['lat'] = ds_QI_hour['lat'].round(2).astype(float)
ds_QI_hour['lon'] = ds_QI_hour['lon'].round(2).astype(float)
##########################################################
#convert hourly to daily
ds_QI_resamp = ds_QI_hour.resample(time='1D').mean()
###########################################################
#starting from 2001 to be compatible with era5_Imerg
ds_QI_resamp = ds_QI_resamp.sel(time=slice('2001', None))
#Transpose 
ds_QI_resamp=ds_QI_resamp.transpose('time', 'lat', 'lon')
#removing 29th of Feb to have a same data for all years
ds_QI_resamp = ds_QI_resamp.sel(time=~((ds_QI_resamp['time'].dt.month == 2) & (ds_QI_resamp['time'].dt.day == 29)))
# Create a custom list of 365 day indices (or labels)
custom_dayofyear = np.tile(np.arange(1, 366), len(np.unique(ds_QI_resamp['time'].dt.year.values)))
# Assign this custom day index as a new coordinate
ds_QI_resamp = ds_QI_resamp.assign_coords(custom_dayofyear=("time", custom_dayofyear))
# round the long, lat a constant number
ds_QI_resamp['lat'] = ds_QI_resamp['lat'].round(2).astype(float)
ds_QI_resamp['lon'] =ds_QI_resamp['lon'].round(2).astype(float)
ds_QI_resamp['time'] = era5_Imerg['time']
#mere resampled QI into our era5-Imerg
era5_Imerg = xr.merge([era5_Imerg, ds_QI_resamp])
```

### 3.7 Additional Modifications

**What this does:**  
- Builds a seasonal mask (winter/spring/summer/fall).  
- Extracts variables for plotting (lon/lat/elevation).

```python
#seasonal labaling
era5_Imerg['Seasonal Mask'] = xr.where(era5_Imerg['time.month'].isin([12, 1, 2]), 'winter',
    xr.where(era5_Imerg['time.month'].isin([3, 4, 5]), 'spring',
        xr.where(era5_Imerg['time.month'].isin([6, 7, 8]), 'summer',
            xr.where(era5_Imerg['time.month'].isin([9, 10, 11]), 'fall', None))))
#########################################################################################################################
#necessary variables for plotting
era5_long = era5_Imerg.variables["lon"][:]
era5_lat  = era5_Imerg.variables["lat"][:]
era5_elev= era5_Imerg['ETOPO_consmap_z'][0,:,:]
```

---

### 3.8 Visualization of Variables

**What this does:**  
- Defines a Cartopy-based plotting helper `map_m_fig` for consistent map styling.  
- Iterates through variables, choosing colormaps and ranges per variable type.  
- Applies a QI-based mask for mean maps and produces extra diagnostics (hexbin, histogram).

```python
#plot colour and size modification for the poster
crs = ccrs.PlateCarree()
custom_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"])


def map_m_fig(fig, m_array, elev, lon, lat, lev_max=3000, lev_range=500, title_fig="title fig", m_unit=r"angular coefficient m $[year^{-1}]", extr=None, chosen_map="RdBu_r", vmin=None, vmax=None):
    ## extremes of colormap
    m_not_nan = np.nan_to_num(m_array)
    if extr is None:
        extr = np.max((np.abs(m_not_nan.min()), m_not_nan.max()))

    # label height
    levels = np.arange(0, lev_max, lev_range)

    ax = fig.add_subplot(projection=crs)
    contour = ax.contour(lon, lat, elev, levels, colors='k', linewidths=0.5)
    ax.add_feature(cfeature.COASTLINE)
    
    title_color = '#000000'  # Title color
    plt.title(title_fig, fontsize=20, fontweight='bold', color=title_color)  # Increased title font size
    
    if vmin is None:
        vmin = -extr
    if vmax is None:
        vmax = extr
    im = plt.pcolormesh(lon, lat, m_not_nan, vmin=vmin, vmax=vmax, cmap=chosen_map)
    
    # Gridlines with larger font size and same color as title
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
    gl.right_labels = gl.top_labels = False
    gl.xlabel_style = {'size': 15, 'color': title_color}
    gl.ylabel_style = {'size': 15, 'color': title_color}

    # Colorbar with larger font size and same color as title
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(m_unit, fontsize=15, color=title_color)
    
    # Set the colorbar background to transparent by setting its frame face color to None
    cbar.outline.set_edgecolor('none')  # Hide the colorbar outline
    
    # Remove the background face color for both the axes and colorbar
    cbar.ax.patch.set_facecolor('none')  # Set the colorbar facecolor to none
    ax.patch.set_facecolor('none')       # Set the axes facecolor to none
    
    # Update colorbar tick labels and ticks to match title color and size
    cbar.ax.yaxis.set_tick_params(labelsize=15, color=title_color)  # Adjust size and color of tick labels
    cbar.ax.yaxis.set_tick_params(which='both', width=2, color=title_color)  # Adjust width and color of ticks
    plt.setp(cbar.ax.get_yticklabels(), color=title_color)  # Set tick labels color to match title
selected_time='2002-11-09'
excluded_vars = ["Rain/No rain threshold","Imerg P with threshold","tp_NearestN","Seasonal Mask"]
included_vars = [var for var in era5_Imerg.data_vars if var not in excluded_vars]

for var in included_vars:
    fig = plt.figure(figsize=(14, 4))
    unit=era5_Imerg[var].attrs.get('units', 'unitless')
    vars_positive = ['totalx', 'tp_NearestN'] #'cape', 'tclw', 'tciw', 'tcwv', 'totalx', 'swvl1', 'tp_NearestN', 'q2m','bs'
    data = era5_Imerg[var].sel(time=selected_time)[0,:,:]
    
    
    if var in vars_positive:
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat, title_fig=f'{var} ({unit})', m_unit='', chosen_map="GnBu", vmin=0)
    
    
    elif var == 'WVT':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat, title_fig=f'{var} (m/s)', m_unit='', chosen_map="GnBu")
    elif var == 'vo':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat, title_fig=f'{var} (1/s)', m_unit='', chosen_map="GnBu")
    elif var == 'w':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat, title_fig=f'{var} (Pa/s)', m_unit='', chosen_map="GnBu")
    elif var == 'u' or var == 'u10' or var == 'v' or var == 'v10':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat, title_fig=f'{var} (m/s)', m_unit='', chosen_map="GnBu")
    elif var == 'vinwvf' or var == 'viewvf':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat, title_fig=f'{var} (kg/m.s)', m_unit='', chosen_map="GnBu")
    elif var == 'mslhf' or var == 'msshf':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat, title_fig=fr'{var} (w/m$^2$)', m_unit='', chosen_map="GnBu")
    elif var == 'mvimd':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat, title_fig=f'{var} (kg/m$^2$.s)', m_unit='', chosen_map="GnBu")

    elif var == 'bs':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=f'{var} (m/s)', m_unit='', chosen_map="GnBu", vmin=0)
        
    elif var == 'q2m':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=f'{var} (kg/kg)', m_unit='', chosen_map="GnBu", vmin=0)
        
    elif var == 'cape':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=f'{var} (j/kg)', m_unit='', chosen_map="GnBu", vmin=0)
        
    elif var == 'swvl1':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=fr'{var} (m$^3$/m$^3$)', m_unit='', chosen_map="GnBu", vmin=0)
        
    elif var == 'tclw' or var == 'tciw' or var == 'tcwv':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=fr'{var} (kg/m$^2$)', m_unit='', chosen_map="GnBu", vmin=0)
        
    elif var == 'd2m' or var == 't2m':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=f'{var} ({unit})', m_unit='', chosen_map="GnBu", vmin=250, vmax=300)

    elif var == 'sp':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=f'{var} ({unit})', m_unit='', chosen_map="GnBu", vmin=80000, vmax=100000)

    elif var == 'tisr':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=fr'{var} (j/m$^2$)', m_unit='', chosen_map="GnBu", vmin=500000, vmax=900000)

    elif var == 't':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=f'{var} ({unit})', m_unit='', chosen_map="GnBu", vmin=250, vmax=280)
        
    elif var == 'Imerg_orgres':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                title_fig=f'IMERG ppt ({unit})', m_unit='', chosen_map="Blues", vmin=0)
    
    elif var == 'ppt_era5':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                title_fig=f'ERA5 ppt ({unit})', m_unit='', chosen_map="Blues", vmin=0)
    
    elif var == 'ETOPO_consmap_z':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                title_fig=f'ETOPO (m)', m_unit='', chosen_map="terrain", vmin=0)
        
    elif var == 'camr':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                title_fig=f'camr (kg/kg)', m_unit='', chosen_map="GnBu", vmin=0)
    
    elif var == 'precipitationQualityIndex':
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                title_fig=f'QI (unitless)', m_unit='', chosen_map="GnBu", vmin=0)
        
    else:
        map_m_fig(fig, data.values, era5_elev, era5_long, era5_lat,
                  title_fig=f'{var} ({unit})', m_unit='', chosen_map="GnBu")
        
#########################################################################################################################
mask= (era5_Imerg['precipitationQualityIndex'] >= 0.6)
    
fig = plt.figure(figsize=(14, 4))
map_m_fig(fig, era5_Imerg["Imerg_orgres"].where(mask,drop=True).mean(dim='time').values, era5_elev, era5_long, era5_lat, title_fig=f'IMERG Precipitation Mean', m_unit=era5_Imerg["Imerg_orgres"].attrs.get('units', 'unitless'), chosen_map="Blues", vmin=0, vmax=8)
fig = plt.figure(figsize=(14, 4))
map_m_fig(fig, era5_Imerg["ETOPO_consmap_z"].mean(dim='time').values, era5_elev, era5_long, era5_lat, title_fig=f'Elevation', m_unit=era5_Imerg["ETOPO_consmap_z"].attrs.get('units', 'unitless'), chosen_map="terrain", vmin=0)
fig = plt.figure(figsize=(14, 4))
map_m_fig(fig, era5_Imerg["precipitationQualityIndex"].mean(dim='time').values, era5_elev, era5_long, era5_lat, title_fig=f'IMERG Quality Index Mean', m_unit=era5_Imerg["precipitationQualityIndex"].attrs.get('units', 'unitless'), chosen_map="viridis", vmin=0)
fig = plt.figure(figsize=(14, 4))
map_m_fig(fig, era5_Imerg["ppt_era5"].where(mask,drop=True).mean(dim='time').values, era5_elev, era5_long, era5_lat, title_fig=f'ERA5 Precipitation Mean', m_unit=era5_Imerg["ppt_era5"].attrs.get('units', 'unitless'), chosen_map="Blues", vmin=0, vmax=8)
fig = plt.figure(figsize=(14, 4))
map_m_fig(fig, era5_Imerg["camr"].where(mask,drop=True).mean(dim='time').values, era5_elev, era5_long, era5_lat, title_fig=f'Camr Mean', m_unit=era5_Imerg["camr"].attrs.get('units', 'unitless'), chosen_map="GnBu", vmin=0)
fig = plt.figure(figsize=(14, 4))
map_m_fig(fig, era5_Imerg["tciw"].where(mask,drop=True).mean(dim='time').values, era5_elev, era5_long, era5_lat, title_fig=f'Tciw Mean', m_unit=era5_Imerg["tciw"].attrs.get('units', 'unitless'), chosen_map="GnBu", vmin=0)
#########################################################################################################################
x = era5_Imerg['ETOPO_consmap_z'].values.flatten()  #
y = era5_Imerg['precipitationQualityIndex'].values.flatten()

# Plot hexbin + linear regression
plt.figure(figsize=(7, 5))
plt.hexbin(x, y, mincnt=1, gridsize=2000, bins='log', cmap='inferno')  # mincnt=1 to avoid sparse hexes
plt.colorbar(label='log(counts)')
plt.xlabel("Elevation (m)", fontsize=15)
plt.ylabel("IMERG Precipitation Quality Index", fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.title("Precipitation Quality Index vs Elevation")
plt.legend()
plt.tight_layout()
plt.show()
#########################################################################################################################
# Flatten the data
pq_index = era5_Imerg['precipitationQualityIndex'].values.flatten()

# Remove NaNs if present
pq_index = pq_index[~np.isnan(pq_index)]

# Count values below 0.6
below_threshold = pq_index < 0.6
count_below = np.sum(below_threshold)
total_count = len(pq_index)
percentage_below = (count_below / total_count) * 100

# Print results
print(f"Count below 0.6: {count_below}")
print(f"Percentage below 0.6: {percentage_below:.2f}%")
# Plot histogram
plt.hist(pq_index, bins=30)
plt.axvline(x=0.6, color='red', linestyle='--', label='Threshold = 0.6')
plt.legend()
plt.xlabel('IMERG Precipitation Quality Index',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.title('Histogram of Imerg Precipitation Quality Index')
plt.show()
```
---

## 4) Feature selection (SHAP)
### 4.1 SHAP: variable selection and feature stack
**What this does:**  
- Chooses the full predictor list for the SHAP/XGB run.  
- Builds `Y` (IMERG precipitation) and `X` (stacked predictors) arrays from `era5_Imerg`.

```python
'''
selected_variables=['tciw','u10', 'v10', 'd2m', 't2m', 'cape', 'ETOPO_consmap_z', 'kx', 'mslhf', 'msshf',
           'sp', 'tisr', 'tcwv', 'totalx', 'viewvf', 'vinwvf', 'WVT','ppt_era5',
           'swvl1', 't', 'u', 'v', 'vo', 'q2m', 'camr', 'bs','mvimd','w', 'tclw'] 

Y=era5_Imerg["Imerg_orgres"].values     #Y.shape=(.., 60, 150)
X = np.stack([era5_Imerg[var].values for var in selected_variables], axis=-1)   #X.shape=(.., 60, 150, 5)
'''
```

---
### 4.2 SHAP: Daily-wise split into Train/Valid/Test
**What this does:**  
- Randomly splits along the **time** dimension only (entire days).  
- Writes a `Train/Valid/Test indices` label cube into the dataset for later masking.  
- Extracts `X_train`, `X_valid`, `X_test` .

```python
#daily_wise data split
'''
# split into training, validation, and test sets (randomized split) based on the 'time' dimension
time_indices = np.arange(Y.shape[0])  # Array of time indices

# Randomly split the time indices into train, validation, and test sets
time_train, time_temp = train_test_split(time_indices, test_size=0.4, random_state=42)  # 40% for temp set (validation + test)
time_valid, time_test = train_test_split(time_temp, test_size=0.5, random_state=42)     # 50% validation, 50% test from temp set

# Create an array for holding the train/valid/test indices based on 'time'
train_test_indices = np.full((Y.shape[0], Y.shape[1], Y.shape[2]), 'unassigned', dtype=object)

# Mark entire days (time steps) as train, valid, or test
train_test_indices[time_train, :, :] = 'train'  # Mark all lat/lon points for the train time steps
train_test_indices[time_valid, :, :] = 'valid'  # Mark all lat/lon points for the valid time steps
train_test_indices[time_test, :, :] = 'test'    # Mark all lat/lon points for the test time steps

# Create a new DataArray for the train/valid/test indices with the original dimensions
era5_Imerg['Train/Valid/Test indices'] = xr.DataArray(
    train_test_indices,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': era5_Imerg.coords['time'],
        'lat': era5_Imerg.coords['lat'],
        'lon': era5_Imerg.coords['lon']
    }
)


# Extract the training data based on time_train indices
X_train = X[time_train, :, :, :]  
y_train = Y[time_train, :, :]

# Extract the validation data based on time_valid indices
X_valid = X[time_valid, :, :, :]  
y_valid = Y[time_valid, :, :]     

# Extract the test data based on time_test indices
X_test = X[time_test, :, :, :]   
y_test = Y[time_test, :, :] 
  
# Display the updated dataset
era5_Imerg
'''
```

---
### 4.3 Train/Valid/Test masking by IMERG QI and NaN removal
**What this does:**  
- Builds **train/test/valid** masks using `precipitationQualityIndex >= 0.6` **AND** the split labels.  
- Stacks masked predictors and target, then removes any NaNs across features/target.  
- Produces flattened arrays ready for XGBoost.

```python
#Train
'''
#Mask of QI
Mask_train=(era5_Imerg['precipitationQualityIndex'] >= 0.6) & (era5_Imerg['Train/Valid/Test indices'] == 'train')
#X
X_masked = [era5_Imerg[var].where(Mask_train, drop=False).values for var in selected_variables]
X_train_QI = np.stack(X_masked, axis=-1)
#y
y_train_QI = era5_Imerg['Imerg_orgres'].where(Mask_train, drop=False).values
print(X_train_QI.shape)
print(y_train_QI.shape)
#####eliminating NA####
mask_valid = ~np.isnan(y_train_QI) & ~np.isnan(X_train_QI).any(axis=-1)
X_train_QI = X_train_QI[mask_valid]
y_train_QI = y_train_QI[mask_valid]
print(X_train_QI.shape)
print(y_train_QI.shape)
#################################################
#Test
#Mask of QI
Mask_test=(era5_Imerg['precipitationQualityIndex'] >= 0.6) & (era5_Imerg['Train/Valid/Test indices'] == 'test')
#X
X_masked = [era5_Imerg[var].where(Mask_test, drop=False).values for var in selected_variables]
X_test_QI = np.stack(X_masked, axis=-1)
#y
y_test_QI = era5_Imerg['Imerg_orgres'].where(Mask_test, drop=False).values
print(X_test_QI.shape)
print(y_test_QI.shape)
#####eliminating NA####
mask_valid = ~np.isnan(y_test_QI) & ~np.isnan(X_test_QI).any(axis=-1)
X_test_QI = X_test_QI[mask_valid]
y_test_QI = y_test_QI[mask_valid]
print(X_test_QI.shape)
print(y_test_QI.shape)
#################################################
#valid
#Mask of QI
Mask_valid=(era5_Imerg['precipitationQualityIndex'] >= 0.6) & (era5_Imerg['Train/Valid/Test indices'] == 'valid')
#X
X_masked = [era5_Imerg[var].where(Mask_valid, drop=False).values for var in selected_variables]
X_valid_QI = np.stack(X_masked, axis=-1)
#y
y_valid_QI = era5_Imerg['Imerg_orgres'].where(Mask_valid, drop=False).values
print(X_valid_QI.shape)
print(y_valid_QI.shape)
#####eliminating NA####
mask_valid = ~np.isnan(y_valid_QI) & ~np.isnan(X_valid_QI).any(axis=-1)
X_valid_QI = X_valid_QI[mask_valid]
y_valid_QI = y_valid_QI[mask_valid]
print(X_valid_QI.shape)
print(y_valid_QI.shape)
'''
```

---
### 4.4 Fit XGBoost, evaluate, and write predictions back to xarray
**What this does:**  
- Trains an `XGBRegressor` on the QI-masked training set.  
- Predicts over the full `X` grid (to reconstruct a time–lat–lon cube) and on the test set.  
- Reports R² and RMSE, then stores the 3D prediction back into `era5_Imerg['XGB']`.

```python
# Define XGBoost regressor
'''
xgb = XGBRegressor(random_state=42, n_jobs=-1) #**best_params,
xgb.fit(X_train_QI, y_train_QI)

yxgb_flat = xgb.predict(X.reshape(-1, X.shape[-1]))
yxgb = yxgb_flat.reshape(-1,Y.shape[1],Y.shape[2])

yxgb_test_flat = xgb.predict(X_test_QI)
r2_xgb = r2_score(y_test_QI, yxgb_test_flat)
rmse_xgb = mean_squared_error(y_test_QI, yxgb_test_flat, squared=False)
print(f'R² XGBoost: {r2_xgb}')
print(f'rmse XGBoost: {rmse_xgb}')
print(X.shape)
print(X_test_QI.shape)
print(X_test.shape)
'''
'''
new_data_array = xr.DataArray(
    yxgb,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': era5_Imerg.coords['time'],
        'lat': era5_Imerg.coords['lat'],
        'lon': era5_Imerg.coords['lon']
    }
)
era5_Imerg['XGB']= new_data_array
'''
```

---
### 4.5 Feature importances (built-in) and SHAP setup
**What this does:**  
- (Optional) Plots model’s built-in feature importances.  
- Creates a SHAP `Explainer`, computes SHAP values on the validation set, and saves them via `joblib` for reuse.

```python
'''
plt.figure(figsize=(10, 6))
bars = plt.bar(selected_variables, xgb.feature_importances_, align='center') #xgb_class, xgb_reg

   
for bar, importance in zip(bars, xgb.feature_importances_):                  #xgb_class, xgb_reg
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X-coordinate
        bar.get_height() + 0.01,           # Y-coordinate
        f'{importance:.2f}',               # Format importance to one decimal place
        ha='center',                       # Align horizontally
        va='bottom',                       # Align vertically
        #rotation=90                        # Rotate text 90 degrees
    )
    
    
plt.xlabel('Feature Importance')
#plt.ylabel('Variables')
plt.xticks(rotation=90)
plt.title('Feature Importances from XGBoost')
#plt.savefig('/leonardo_work/mBI25_AmbCo/igoudarz/test20241226.png', bbox_inches='tight')
plt.show()
'''
'''
#1 step
import joblib
import shap

# Initialize the SHAP explainer with the XGBRegressor model
explainer = shap.Explainer(xgb, X_valid_QI)

# Compute SHAP values for the entire dataset
shap_values = explainer(X_valid_QI)

# Save SHAP values and related data
joblib.dump((shap_values, X_valid_QI, selected_variables), 'shap_data20250614.pkl')
'''
```

---
### 4.6 SHAP visualizations & ranked features

**What this does:**  
- Creates SHAP summary scatter and bar plots.  
- Ranks features by mean |SHAP| and performs iterative top-k selection to evaluate RMSE vs. number of variables.  
- Saves results and visualizes the RMSE curve.

```python
'''
# Set font sizes before the plot
plt.rc('axes', labelsize=18)   # Axis labels 
plt.rc('xtick', labelsize=16)  # X-axis tick labels 
plt.rc('ytick', labelsize=16)  # Y-axis tick labels 

# Plot the summary of SHAP values to show global feature importance
shap.summary_plot(shap_values, X_valid_QI, feature_names=selected_variables)

plt.show()
'''
'''
# Set font sizes before the plot
plt.rc('axes', labelsize=18)   # Axis labels 
plt.rc('xtick', labelsize=16)  # X-axis tick labels 
plt.rc('ytick', labelsize=16)  # Y-axis tick labels 

# Plot the bar chart for mean absolute SHAP values (global feature importance)
shap.summary_plot(shap_values, X_valid_QI, plot_type="bar", feature_names=selected_variables)

plt.show()
'''
'''
# Compute the mean absolute SHAP values for each feature
mean_abs_shap_values = pd.DataFrame({
    'Feature': selected_variables,
    'Mean Absolute SHAP Value': np.abs(shap_values.values).mean(axis=0)
})

# Sort features by mean absolute SHAP value in descending order
ranked_features = mean_abs_shap_values.sort_values(by="Mean Absolute SHAP Value", ascending=False)
ranked_features=ranked_features.iloc[:15,:]
print(ranked_features['Feature'])
#############################################
results = []

for i in tqdm(range(1, len(ranked_features) + 1)):  # Ensure at least one variable is selected
    vars = ranked_features.iloc[:i, 0].tolist()  # Get the top `i` variables as a list
    
    #train
    Mask_train=(era5_Imerg['precipitationQualityIndex'] >= 0.6) & (era5_Imerg['Train/Valid/Test indices'] == 'train')
    X_masked = [era5_Imerg[var].where(Mask_train, drop=False).values for var in vars]
    X_train_internal = np.stack(X_masked, axis=-1)
    y_train_internal = era5_Imerg['Imerg_orgres'].where(Mask_train, drop=False).values
    mask_valid = ~np.isnan(y_train_internal) & ~np.isnan(X_train_internal).any(axis=-1)
    X_train_internal = X_train_internal[mask_valid]
    y_train_internal = y_train_internal[mask_valid]
    #test
    Mask_test=(era5_Imerg['precipitationQualityIndex'] >= 0.6) & (era5_Imerg['Train/Valid/Test indices'] == 'test')
    X_masked = [era5_Imerg[var].where(Mask_test, drop=False).values for var in vars]
    X_test_internal = np.stack(X_masked, axis=-1)
    y_test_internal = era5_Imerg['Imerg_orgres'].where(Mask_test, drop=False).values
    mask_valid = ~np.isnan(y_test_internal) & ~np.isnan(X_test_internal).any(axis=-1)
    X_test_internal = X_test_internal[mask_valid]
    y_test_internal = y_test_internal[mask_valid]
     
    
    # Define XGBoost regressor
    xgb_internal = XGBRegressor(random_state=42, n_jobs=-1) #**best_params,
    xgb_internal.fit(X_train_internal, y_train_internal)
    yxgb_test_flat_internal = xgb_internal.predict(X_test_internal)
    
    #we decided to put 0 all below 1 for Imerg and XGB
    yxgb_test_flat_internal=np.where(yxgb_test_flat_internal<1,0,yxgb_test_flat_internal)
    y_test_internal=np.where(y_test_internal<1,0,y_test_internal)
    
    rmse_xgb_internal = mean_squared_error(y_test_internal, yxgb_test_flat_internal, squared=False)
    
    # Append the result as a tuple (variables, accuracy)
    results.append((vars, round(rmse_xgb_internal, 2)))
    
# Create a DataFrame from the results
df_1s = pd.DataFrame(results, columns=['Variables', 'RMSE_XGB_1s'])
###############################################
df_1s['Variables'] = df_1s['Variables'].apply(lambda x: ', '.join(x))
##############################################à
#save the df
df_1s.to_csv("RMSE_XGB_1s_results20250614.csv", index=False)
###############################################
# Bar plot for RMSE by Feature_Set_Season and Target_Season
plt.figure(figsize=(12, 6))
plt.plot(df_1s['RMSE_XGB_1s'],df_1s['Variables'], marker='o', linestyle='-', color='b')
plt.title("RMSE_XGB_1s")
plt.xlabel("RMSE(mm/d)")
plt.ylabel("Feature")
plt.show()
'''
```

---
### 4.7 RMSE graph from saved CSV + reloading SHAP data

**What this does:**  
- Loads the saved CSV of incremental top-k results and plots RMSE vs. features.  
- Shows how to reload the persisted SHAP objects and redraw summary plots with custom font sizes.

```python
#RMSE graph of SHAP ranking
'''
df = pd.read_csv("RMSE_XGB_1s_results20250614.csv")

# Count the number of variables used in each row
df["Num_Variables"] = df["Variables"].apply(lambda x: len(x.split(",")))

# Plot RMSE vs. number of variables
plt.figure(figsize=(10, 6))
plt.plot(df["RMSE_XGB_1s"],df["Variables"], marker='o')
plt.xlabel("RMSE (mm/d)")
plt.ylabel("Variables")
#plt.title("RMSE vs. Variables")
plt.grid(True)
#plt.xticks(df["Num_Variables"])  # Show all points on x-axis
#plt.gca().invert_yaxis()  # Lower RMSE is better
plt.tight_layout()
plt.show()
'''
'''
#load the stored SHAP data
shap_values, X_valid, selected_variables = joblib.load('C:\Iman\Maxm\shap_data20250614.pkl')
'''
'''
# Create the SHAP summary plot and get the figure
fig = plt.figure()
shap.summary_plot(shap_values, X_valid.reshape(-1, X_valid.shape[-1]), feature_names=selected_variables, show=False)

# Now modify font sizes
plt.xlabel(plt.gca().get_xlabel(), fontsize=18)
plt.ylabel(plt.gca().get_ylabel(), fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
'''
'''
fig = plt.figure()
shap.summary_plot(shap_values, X_valid.reshape(-1, X_valid.shape[-1]), plot_type="bar", feature_names=selected_variables, show=False)

# Now modify font sizes
plt.xlabel(plt.gca().get_xlabel(), fontsize=15)
plt.ylabel(plt.gca().get_ylabel(), fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.show()
'''
```

---
## 5) Machine Learning: Model training

> **Target skills:** RMSE ≈ 4.39 mm/d vs ERA5 5.13 mm/d; R² ≈ 0.59 on test (spatio‑temporal).

### 5.1 Input and Output section

**What this does:**  
- Runs a compact example with just three predictors.  
- Repeats the time-based split and stores split labels.  
- Extracts `X_train/X_valid/X_test` and targets.

```python
selected_variables= ['tciw','ppt_era5', 'camr']

Y=era5_Imerg["Imerg_orgres"].values    
X = np.stack([era5_Imerg[var].values for var in selected_variables], axis=-1)  
#daily_wise data split

# split into training, validation, and test sets (randomized split) based on the 'time' dimension
time_indices = np.arange(Y.shape[0])  # Array of time indices

# Randomly split the time indices into train, validation, and test sets
time_train, time_temp = train_test_split(time_indices, test_size=0.4, random_state=42)  # 40% for temp set (validation + test)
time_valid, time_test = train_test_split(time_temp, test_size=0.5, random_state=42)     # 50% validation, 50% test from temp set

# Create an array for holding the train/valid/test indices based on 'time'
train_test_indices = np.full((Y.shape[0], Y.shape[1], Y.shape[2]), 'unassigned', dtype=object)

# Mark entire days (time steps) as train, valid, or test
train_test_indices[time_train, :, :] = 'train'  # Mark all lat/lon points for the train time steps
train_test_indices[time_valid, :, :] = 'valid'  # Mark all lat/lon points for the valid time steps
train_test_indices[time_test, :, :] = 'test'    # Mark all lat/lon points for the test time steps

# Create a new DataArray for the train/valid/test indices with the original dimensions

era5_Imerg['Train/Valid/Test indices'] = xr.DataArray(
    train_test_indices,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': era5_Imerg.coords['time'],
        'lat': era5_Imerg.coords['lat'],
        'lon': era5_Imerg.coords['lon']})


# Extract the training data based on time_train indices
X_train = X[time_train, :, :, :]  
y_train = Y[time_train, :, :]

# Extract the validation data based on time_valid indices
X_valid = X[time_valid, :, :, :]  
y_valid = Y[time_valid, :, :]     

# Extract the test data based on time_test indices
X_test = X[time_test, :, :, :]   
y_test = Y[time_test, :, :] 
```

---
### 5.2 Modelling: mask, train, evaluate, write back, and inspect hyperparameters
**What this does:**  
- Applies the QI and split masks; removes NaNs.  
- Fits XGB, predicts a full-cube `yxgb`, prints metrics, stores as `era5_Imerg["XGB"]`.  
- Reads and prints trained booster hyperparameters.  
- Applies the trained model to the **back-in-time** dataset (`era5_backintime`) using the same three predictors.

```python
#Train
#Mask of QI
Mask_train=(era5_Imerg['precipitationQualityIndex'] >= 0.6) & (era5_Imerg['Train/Valid/Test indices'] == 'train')
#X
X_masked = [era5_Imerg[var].where(Mask_train, drop=False).values for var in selected_variables]
X_train_QI = np.stack(X_masked, axis=-1)
#y
y_train_QI = era5_Imerg['Imerg_orgres'].where(Mask_train, drop=False).values
print(X_train_QI.shape)
print(y_train_QI.shape)
#####eliminating NA####
mask_valid = ~np.isnan(y_train_QI) & ~np.isnan(X_train_QI).any(axis=-1)
X_train_QI = X_train_QI[mask_valid]
y_train_QI = y_train_QI[mask_valid]
print(X_train_QI.shape)
print(y_train_QI.shape)
#################################################
#Test
#Mask of QI
Mask_test=(era5_Imerg['precipitationQualityIndex'] >= 0.6) & (era5_Imerg['Train/Valid/Test indices'] == 'test')
#X
X_masked = [era5_Imerg[var].where(Mask_test, drop=False).values for var in selected_variables]
X_test_QI = np.stack(X_masked, axis=-1)
#y
y_test_QI = era5_Imerg['Imerg_orgres'].where(Mask_test, drop=False).values
print(X_test_QI.shape)
print(y_test_QI.shape)
#####eliminating NA####
mask_valid = ~np.isnan(y_test_QI) & ~np.isnan(X_test_QI).any(axis=-1)
X_test_QI = X_test_QI[mask_valid]
y_test_QI = y_test_QI[mask_valid]
print(X_test_QI.shape)
print(y_test_QI.shape)
#################################################
#valid
#Mask of QI
Mask_valid=(era5_Imerg['precipitationQualityIndex'] >= 0.6) & (era5_Imerg['Train/Valid/Test indices'] == 'valid')
#X
X_masked = [era5_Imerg[var].where(Mask_valid, drop=False).values for var in selected_variables]
X_valid_QI = np.stack(X_masked, axis=-1)
#y
y_valid_QI = era5_Imerg['Imerg_orgres'].where(Mask_valid, drop=False).values
print(X_valid_QI.shape)
print(y_valid_QI.shape)
#####eliminating NA####
mask_valid = ~np.isnan(y_valid_QI) & ~np.isnan(X_valid_QI).any(axis=-1)
X_valid_QI = X_valid_QI[mask_valid]
y_valid_QI = y_valid_QI[mask_valid]
print(X_valid_QI.shape)
print(y_valid_QI.shape)
######################################
######################################
######################################
######################################
######################################
######################################   #**best_params,
# Define XGBoost regressor
xgb = XGBRegressor(random_state=42, n_jobs=-1) 
xgb.fit(X_train_QI, y_train_QI)

yxgb_flat = xgb.predict(X.reshape(-1, X.shape[-1]))
yxgb = yxgb_flat.reshape(-1,Y.shape[1],Y.shape[2])

print(xgb.get_xgb_params())

yxgb_test_flat = xgb.predict(X_test_QI)
r2_xgb = r2_score(y_test_QI, yxgb_test_flat)
rmse_xgb = mean_squared_error(y_test_QI, yxgb_test_flat, squared=False)
print(f'R² XGBoost: {r2_xgb}')
print(f'rmse XGBoost: {rmse_xgb}')
print(X.shape)
print(X_test_QI.shape)
print(X_test.shape)
################
new_data_array = xr.DataArray(
    yxgb,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': era5_Imerg.coords['time'],
        'lat': era5_Imerg.coords['lat'],
       'lon': era5_Imerg.coords['lon']
    })
era5_Imerg['XGB']= new_data_array
################################################################################################################
# Get booster config
config_json = xgb.get_booster().save_config()
parsed = json.loads(config_json)

# Extract relevant sections
generic_params = parsed["learner"]["generic_param"]
tree_params = parsed["learner"]["gradient_booster"]["tree_train_param"]
model_params = parsed["learner"]["gradient_booster"]["gbtree_model_param"]

# Extract hyperparameters
n_estimators = int(model_params.get("num_trees")) 
learning_rate = float(tree_params.get("eta"))      
max_depth = int(tree_params.get("max_depth"))       
min_child_weight = float(tree_params.get("min_child_weight"))  
n_jobs = int(generic_params.get("n_jobs"))          
random_state = int(generic_params.get("random_state"))  
max_leaves = int(tree_params.get("max_leaves"))      

# Print the results
print(f"n_estimators: {n_estimators}")
print(f"learning_rate: {learning_rate}")
print(f"max_depth: {max_depth}")
print(f"min_child_weight: {min_child_weight}")
print(f"n_jobs: {n_jobs}")
print(f"random_state: {random_state}")
print(f"max_leaves: {max_leaves}")
################################################################################################################
selected_variables= ['tciw','ppt_era5', 'camr']
X_back = np.stack([era5_backintime[var].values for var in selected_variables], axis=-1) 
internal = xgb.predict(X_back.reshape(-1, X_back.shape[-1]))
internal = internal.reshape(-1,Y.shape[1],Y.shape[2])

new_data_array = xr.DataArray(
    internal,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': era5_backintime.coords['time'],
        'lat': era5_backintime.coords['lat'],
        'lon': era5_backintime.coords['lon']
    }
)
era5_backintime['XGB']= new_data_array
```

---

## 6) Error Analysis

> **Comment:** Compute RMSE between IMERG and ERA5 vs. IMERG and XGB per gridpoint and season (MAM, JJA, SON, DJF) and other analysis.

### 6.1 Quick RMSE between ERA5 and IMERG (all times with QI ≥ 0.6)

**What this does:**  
- Applies a basic quality mask (`precipitationQualityIndex >= 0.6`).  
- Computes a single **spatiotemporal RMSE** between IMERG (`Imerg_orgres`) and ERA5 precipitation (`ppt_era5`).  
- Also prepares a **per-gridcell RMSE map** averaged over time.

```python
#a quick analysis regarding era5 and IMERG
mask_q=  (era5_Imerg['precipitationQualityIndex'] >= 0.6)
A = era5_Imerg['Imerg_orgres'].where(mask_q, drop=True)
B = era5_Imerg['ppt_era5'].where(mask_q, drop=True)

squared_diff = (A - B) ** 2
mean_squared_diff = squared_diff.mean(dim='time')
rmse = np.sqrt(np.nanmean((A.values.flatten() - B.values.flatten()) ** 2))
print(rmse)
#RMSE MAP of ERA5 and IMERG
mask_t_q= (era5_Imerg['precipitationQualityIndex'] >= 0.6) #& (era5_Imerg['Train/Valid/Test indices'] == 'test')
A = era5_Imerg['Imerg_orgres'].where(mask_t_q, drop=True)
B = era5_Imerg['ppt_era5'].where(mask_t_q, drop=True)

squared_diff = (A - B) ** 2
mean_squared_diff = squared_diff.mean(dim='time')
rmse_map = np.sqrt(mean_squared_diff)
rmse = np.sqrt(np.nanmean((A.values.flatten() - B.values.flatten()) ** 2))
print(rmse)
# Plot RMSE map
fig1 = plt.figure(figsize=(14, 4), facecolor='none')
map_m_fig(fig1, rmse_map, era5_elev, era5_long, era5_lat,title_fig=f'', m_unit=r"RMSE mm/d", chosen_map="Reds", vmin=4, vmax=12)
```

---
### 6.2 RMSE map of **XGB vs IMERG**, plus mean IMERG precipitation map
**What this does:**  
- Computes RMSE between IMERG and **XGB** predictions (`era5_Imerg['XGB']`) using the same mask.  
- Plots the RMSE map and the mean IMERG precipitation map for context.

```python
#################################################################################################################################
A = era5_Imerg['Imerg_orgres'].where(mask_t_q, drop=True)
B = era5_Imerg['XGB'].where(mask_t_q, drop=True)

squared_diff = (A - B) ** 2
mean_squared_diff = squared_diff.mean(dim='time')
rmse_map = np.sqrt(mean_squared_diff)
rmse = np.sqrt(np.nanmean((A.values.flatten() - B.values.flatten()) ** 2))
print(rmse)
# Plot RMSE map
fig1 = plt.figure(figsize=(14, 4), facecolor='none')
map_m_fig(fig1, rmse_map, era5_elev, era5_long, era5_lat,title_fig=f'', m_unit=r"RMSE mm/d", chosen_map="Reds", vmin=4, vmax=12)
#################################################################################################################################
fig = plt.figure(figsize=(14, 4))
map_m_fig(fig, era5_Imerg["Imerg_orgres"].where((era5_Imerg['precipitationQualityIndex'] >= 0.6),drop=True).mean(dim='time').values, era5_elev, era5_long, era5_lat, title_fig=f'', m_unit=era5_Imerg["Imerg_orgres"].attrs.get('units', 'unitless'), chosen_map="Blues", vmin=0, vmax=6)
```

---
### 6.3 **Seasonal** RMSE maps (test subset, QI ≥ 0.6)
**What this does:**  
- Loops over seasons (`fall`, `winter`, `spring`, `summer`).  
- Within the **test** subset and QI mask, computes RMSE maps for **ERA5 vs IMERG** and **XGB vs IMERG**.  
- Also maps **mean IMERG precipitation** per season.  
- Collects and shows all figures.

```python
#seasonal ERA5 RMSE 

seasons = ['fall', 'winter', 'spring', 'summer']
figures = []  # Collect all figures here

for season in seasons:
    print(f"\nProcessing season: {season}")
    mask_s_t_q=  (era5_Imerg['Seasonal Mask'] == season) & (era5_Imerg['Train/Valid/Test indices'] == 'test') & (era5_Imerg['precipitationQualityIndex'] >= 0.6)
    A = era5_Imerg['Imerg_orgres'].where(mask_s_t_q,drop=True)
    B = era5_Imerg['ppt_era5'].where(mask_s_t_q,drop=True)
    C = era5_Imerg['XGB'].where(mask_s_t_q,drop=True)
    ################################################################################################################################
    ################################################################################################################################
    squared_diff = (A - B) ** 2
    mean_squared_diff = squared_diff.mean(dim='time')
    rmse_map = np.sqrt(mean_squared_diff)
    rmse = np.sqrt(np.nanmean((A.values.flatten() - B.values.flatten()) ** 2))

    # ########Plot RMSE map#################################### 
    fig1 = plt.figure(figsize=(14, 4), facecolor='none')
    map_m_fig(fig1, rmse_map, era5_elev, era5_long, era5_lat,title_fig=f'RMSE Map of ERA5 (test data) & ({season}), SpatioTemporal RMSE = {rmse:.2f} mm/d',
              m_unit=r"RMSE mm/d", chosen_map="Reds", vmin=4)
    figures.append(fig1)
    ######## Plot RMSE map without title########################
    fig2 = plt.figure(figsize=(14, 4), facecolor='none')
    map_m_fig(fig2, rmse_map, era5_elev, era5_long, era5_lat,title_fig=f'', m_unit=r"RMSE mm/d", chosen_map="Reds", vmin=4, vmax=12)
    figures.append(fig2)
    ######## Mean Precipitation####################################
    fig3 = plt.figure(figsize=(14, 4), facecolor='none')
    map_m_fig(fig3, era5_Imerg['Imerg_orgres'].where((era5_Imerg['Seasonal Mask'] == season)& (era5_Imerg['precipitationQualityIndex'] >= 0.6), drop=True).mean(dim='time'),
              era5_elev, era5_long, era5_lat,title_fig=f'', m_unit=r"mm/d", chosen_map="Blues", vmin=0 ,vmax=6)
    figures.append(fig3)
    ################################################################################################################################à
    ################################################################################################################################à
    squared_diff = (A - C) ** 2
    mean_squared_diff = squared_diff.mean(dim='time')
    rmse_map = np.sqrt(mean_squared_diff)
    rmse = np.sqrt(np.nanmean((A.values.flatten() - C.values.flatten()) ** 2))

    # ########Plot RMSE map#################################### 
    fig4 = plt.figure(figsize=(14, 4), facecolor='none')
    map_m_fig(fig4, rmse_map, era5_elev, era5_long, era5_lat,title_fig=f'RMSE Map of XGB (test data) & ({season}), SpatioTemporal RMSE = {rmse:.2f} mm/d',
              m_unit=r"RMSE mm/d", chosen_map="Reds", vmin=4)
    figures.append(fig4)
    ######## Plot RMSE map without title########################
    fig5 = plt.figure(figsize=(14, 4), facecolor='none')
    map_m_fig(fig5, rmse_map, era5_elev, era5_long, era5_lat,title_fig=f'', m_unit=r"RMSE mm/d", chosen_map="Reds", vmin=4, vmax=12)
    figures.append(fig5)


# After loop: show all collected figures
print("\nDisplaying all figures...")
for fig in figures:
    fig.show()
```

---
### 6.4 Error vs **Elevation percentile** (Z)
**What this does:**  
- Classifies grid cells into **100 percentiles** based on elevation (`ETOPO_consmap_z`).  
- Aggregates **RMSE(ERA5), RMSE(XGB), and mean IMERG** by percentile **bins**.  
- Plots RMSE vs. elevation percentile with a secondary x-axis showing the **actual elevation ranges**.

```python
percentiles = range(100, 0, -1)

# Flatten the DataArray to get all z values across time, long, lat
z_values_flattened = era5_Imerg['ETOPO_consmap_z'].values.flatten()

# Initialize an empty DataArray for storing the percentile class
percentile_class = xr.full_like(era5_Imerg['ETOPO_consmap_z'], np.nan)

# Loop through each percentile and assign classes
for i in percentiles:
    percentile_value = np.percentile(z_values_flattened, i)
    # Use xarray's where method to assign the class where the condition is met
    percentile_class = percentile_class.where(era5_Imerg['ETOPO_consmap_z'] >= percentile_value, i) #if false, i

# Add the new DataArray as a variable in the original dataset 
era5_Imerg['Z 100 percentile class'] = percentile_class
era5_Imerg
# Define percentile bins
percentile_bins = [(0, 13), (14, 20), (20, 30), (30, 40), (40, 50),
                   (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]

# Initialize list to collect results
results = []

for lower, upper in percentile_bins:
    mask = (
        (era5_Imerg['Z 100 percentile class'] > lower) &
        (era5_Imerg['Z 100 percentile class'] <= upper) &
        (era5_Imerg['precipitationQualityIndex'] >= 0.6) &
        (era5_Imerg['Train/Valid/Test indices'] == 'test')
    )

    A = era5_Imerg['Imerg_orgres'].where(mask).values.flatten()
    B = era5_Imerg['XGB'].where(mask).values.flatten()
    C = era5_Imerg['ppt_era5'].where(mask).values.flatten()

    rmse_xgb = np.sqrt(np.nanmean((A - B) ** 2))
    rmse_era5 = np.sqrt(np.nanmean((A - C) ** 2))
    mean_Imerg = np.nanmean(A)

    lower_elev = np.percentile(era5_Imerg['ETOPO_consmap_z'].values.flatten(), lower)
    upper_elev = np.percentile(era5_Imerg['ETOPO_consmap_z'].values.flatten(), upper)

    results.append({
        'lower_percentile': lower,
        'upper_percentile': upper,
        'lower_elev': lower_elev,
        'upper_elev': upper_elev,
        'rmse_xgb': rmse_xgb,
        'rmse_era5': rmse_era5,
        'mean_Imerg': mean_Imerg
    })

# Create DataFrame
df_zclass = pd.DataFrame(results)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot RMSE
ax1.errorbar(range(len(df_zclass)), df_zclass['rmse_era5'], fmt='o-', color='blue', label='ERA5')
ax1.errorbar(range(len(df_zclass)), df_zclass['rmse_xgb'], fmt='o-', color='purple', label='XGB')
ax1.set_ylabel('RMSE (mm/d)', color='black', fontsize=20)
ax1.set_xticks(range(len(df_zclass)))
xticklabels = [f"({l},{u}]" for l, u in zip(df_zclass['lower_percentile'], df_zclass['upper_percentile'])]
elev_classes = [f"({int(l)},{int(u)})" for l, u in zip(df_zclass['lower_elev'], df_zclass['upper_elev'])]
ax1.set_xticklabels(xticklabels, rotation=45, fontsize=20)
ax1.set_xlabel('Elevation percentile', fontsize=20)
ax1.grid(True, which='both', linestyle='--', alpha=0.6)

# Secondary axis for mean IMERG precipitation
ax2 = ax1.twinx()
ax2.plot(range(len(df_zclass)), df_zclass['mean_Imerg'], 'g--o', label='Mean IMERG Precipitation')
ax2.set_ylabel('Mean IMERG Precipitation (mm/d)', color='green', fontsize=20)

# Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=15)

# Elevation classes as secondary x-axis
ax3 = ax1.twiny()
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(range(len(df_zclass)))
ax3.set_xticklabels(elev_classes, rotation=45, fontsize=20)
ax3.set_xlabel('Elevation class (m)', fontsize=20)

plt.tight_layout()
plt.show()
```

---
### 6.5 Error vs **IMERG precipitation percentile** (P)
**What this does:**  
- Classifies grid cells (time–lat–lon) by **99–1 percentiles** of IMERG precipitation.  
- Aggregates **RMSE(ERA5), RMSE(XGB), and mean IMERG** by precipitation percentile **bins**.  
- Plots RMSE vs. percentile with a secondary x-axis showing **actual IMERG value ranges**.

```python
# 100 percent classification for Z

percentiles = range(100, 0, -1)

# Flatten the DataArray to get all z values across time, long, lat
Imerg_orgres_values_flattened = era5_Imerg['Imerg_orgres'].values.flatten()

# Initialize an empty DataArray for storing the percentile class
percentile_class = xr.full_like(era5_Imerg['Imerg_orgres'], np.nan)

# Loop through each percentile and assign classes
for i in percentiles:
    percentile_value = np.percentile(Imerg_orgres_values_flattened, i)
    # Use xarray's where method to assign the class where the condition is met
    percentile_class = percentile_class.where(era5_Imerg['Imerg_orgres'] >= percentile_value, i) 

# Add the new DataArray as a variable in the original dataset 
era5_Imerg['Imerg_orgres 100 percentile class'] = percentile_class
era5_Imerg

# Define percentile bins
percentile_bins = [(0, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 100)]

# Initialize list to collect results
results = []

for lower, upper in percentile_bins:
    mask = (
        (era5_Imerg['Imerg_orgres 100 percentile class'] > lower) &
        (era5_Imerg['Imerg_orgres 100 percentile class'] <= upper) &
        (era5_Imerg['precipitationQualityIndex'] >= 0.6) &
        (era5_Imerg['Train/Valid/Test indices'] == 'test')
    )

    A = era5_Imerg['Imerg_orgres'].where(mask).values.flatten()
    B = era5_Imerg['XGB'].where(mask).values.flatten()
    C = era5_Imerg['ppt_era5'].where(mask).values.flatten()

    rmse_xgb = np.sqrt(np.nanmean((A - B) ** 2))
    rmse_era5 = np.sqrt(np.nanmean((A - C) ** 2))
    mean_Imerg = np.nanmean(A)

    lower_elev = np.percentile(era5_Imerg['Imerg_orgres'].values.flatten(), lower)
    upper_elev = np.percentile(era5_Imerg['Imerg_orgres'].values.flatten(), upper)

    results.append({
        'lower_percentile': lower,
        'upper_percentile': upper,
        'lower_elev': lower_elev,
        'upper_elev': upper_elev,
        'rmse_xgb': rmse_xgb,
        'rmse_era5': rmse_era5,
        'mean_Imerg': mean_Imerg
    })

# Create DataFrame
df_pclass = pd.DataFrame(results)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot RMSE
ax1.errorbar(range(len(df_pclass)), df_pclass['rmse_era5'], fmt='o-', color='blue', label='ERA5')
ax1.errorbar(range(len(df_pclass)), df_pclass['rmse_xgb'], fmt='o-', color='purple', label='XGB')
ax1.set_ylabel('RMSE (mm/d)', color='black', fontsize=20)
ax1.set_xticks(range(len(df_pclass)))
xticklabels = [f"({l},{u}]" for l, u in zip(df_pclass['lower_percentile'], df_pclass['upper_percentile'])]
elev_classes = [f"({l:.1f},{u:.1f})" for l, u in zip(df_pclass['lower_elev'], df_pclass['upper_elev'])]
ax1.set_xticklabels(xticklabels, rotation=45, fontsize=20)
ax1.set_xlabel('IMERG Precipitation percentile', fontsize=20)
ax1.grid(True, which='both', linestyle='--', alpha=0.6)

# Secondary axis for mean IMERG precipitation
ax2 = ax1.twinx()
ax2.plot(range(len(df_pclass)), df_pclass['mean_Imerg'], 'g--o', label='Mean IMERG Precipitation')
ax2.set_ylabel('Mean IMERG Precipitation (mm/d)', color='green', fontsize=20)

# Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=15)

# Elevation classes as secondary x-axis
ax3 = ax1.twiny()
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(range(len(df_pclass)))
ax3.set_xticklabels(elev_classes, rotation=45, fontsize=20)
ax3.set_xlabel('IMERG Precipitation (mm/d)', fontsize=20)

plt.tight_layout()
plt.show()
```

## 7) Validation against HISTALP & CHAPTER (monthly)

- For each HISTALP station: use the nearest grid point from IMERG.  
- Build monthly totals and compare **R²** and **RMSE** vs. HISTALP.

> **Expected:** ML‑IMEX shows **R² ≈ 0.87** against HISTALP (1960–2000), outperforming ERA5 and CHAPTER.

---
### 7.1 Load HISTALP station CSVs, parse metadata, clean, and concatenate

**What this does:**  
- Iterates over a list of station IDs and reads each `HISTALP_{LOC}_R01_1760_2025.csv`.  
- Extracts metadata (lon/lat/height/station name) from commented lines.  
- Drops seasonal/annual summary columns if present; keeps monthly columns.  
- Filters the data to `1960–2020` and concatenates all stations into a single DataFrame.

```python
loc_list = [
    "AT_BMU", "AT_FRE", "AT_KOL", "AT_LAD", "AT_SPO", "AT_WIE",
    "BA_JAJ", "DE_KMF", "DE_LHU", "DE_ULM", "HR_POZ", "HR_ZAD", "HU_PAP",
    "IT_BAR", "IT_FOR", "IT_IMP", "IT_MIL", "IT_UDI", "IT_VAL", "SI_LJU", 
    "AT_ADM", "SI_CEL", "IT_ROV", "IT_BRX", "DE_OBS","IT_TOM",
    "IT_BOZ", "HU_SZO", "HR_KVA", "DE_STU", "DE_ROS","IT_PAR",
    "BA_BIH", "AT_WAI", "AT_TAM", "AT_SSB", "AT_KAL","IT_VEN",
    "AT_SAL", "AT_RAU", "AT_OFK", "AT_LAG", "AT_KRE", "AT_KOR", 
    "AT_BST", "AT_BRE", "AT_FLA", "AT_INN", "AT_VIL", "AT_RET",    
    ]
all_dfs = []

for loc in loc_list: 
    # Define the data directory
    data_dir = f"{path}/HISTALP_stations/HISTALP_{loc}_R01_1760_2025.csv"

    # Read raw data to extract metadata
    with open(data_dir, "r") as f:
        raw_lines = f.readlines()

    # Extract metadata from comment lines
    metadata = {}
    data_start_idx = 0  # Default index to locate actual data start

    for i, line in enumerate(raw_lines):
        if line.startswith("#"):
            key_value = line.replace("#", "").strip().split(": ", 1)  # Ensure only 1 split
            if len(key_value) == 2:
                metadata[key_value[0].strip()] = key_value[1].strip()
        else:
            data_start_idx = i  # Identify where actual data starts
            break

    # Read CSV with proper delimiter (;)
    df = pd.read_csv(data_dir, delimiter=";", skiprows=data_start_idx)

    # Rename the first column as 'Year'
    df.rename(columns={df.columns[0]: "Year"}, inplace=True)

    # Convert 'Year' to integer and replace missing values
    df["Year"] = df["Year"].astype(int)
    df.replace(999999, float("nan"), inplace=True)

    # Add longitude from metadata (handle missing key safely)
    df["longitude"] = float(metadata.get("longitude", "nan"))  # Default to NaN if missing
    df["latitude"] = float(metadata.get("latitude", "nan"))
    df["height"] = float(metadata.get("height", "nan"))
    df["station"] = str(metadata.get("station", "nan"))

    columns_to_remove = ["mar-may", "jun-aug", "sep-nov", "dec-feb", "apr-sep", "oct-mar", "jan-dec"]
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')  # 'errors="ignore"' prevents errors if columns don't exist

    # Filter rows where 'Year' is between 1960 and 2014
    df = df[(df["Year"] >= 1960) & (df["Year"] <= 2020)] 
    
    all_dfs.append(df)

# Concatenate all DataFrames into one
Histalp_stations_df = pd.concat(all_dfs, ignore_index=True)
#print(Histalp_stations_df.isnull().sum())
Histalp_stations_df
```

---
### 7.2 Find closest IMERG grid point to each station (geodesic) and merge coordinates
**What this does:**  
- Computes nearest IMERG grid cell for each unique station coordinate using great-circle distance.  
- Merges the matched IMERG coordinates into the station table.  
- Rounds IMERG-lat/lon to fixed precision for consistent selection.

```python
for Mean of 20 stations
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km
################################################################################
def find_closest_point(Imerg, station_latitude, station_longitude):
    closest_lat, closest_lon = None, None
    min_distance = float("inf")  # Start with a very large distance

    for lat in Imerg['lat']:  # Loop through all latitudes
        for lon in Imerg['lon']:  # Loop through all longitudes
            distance = calculate_distance(lat, lon, station_latitude, station_longitude)

            if distance < min_distance:
                min_distance = distance
                closest_lat, closest_lon = lat, lon

    return float(closest_lat), float(closest_lon)  # Return the closest point found
#closest Grid point of era5_Imerg based on stations' positions
df_lon_lat= Histalp_stations_df[['latitude', 'longitude']].drop_duplicates()

# Loop through each row 
closest_latitudes = []
closest_longitudes = []

for index, row in tqdm(df_lon_lat.iterrows()):
    closest_lat, closest_lon = find_closest_point(era5_Imerg, row['latitude'], row['longitude'])
    closest_latitudes.append(closest_lat)
    closest_longitudes.append(closest_lon)

# Add the new columns to the DataFrame
df_lon_lat['closest_Imerg_lat'] = closest_latitudes
df_lon_lat['closest_Imerg_lon'] = closest_longitudes
#####################################################################################################
# Merge Histalp_stations_df with df_lon_lat on latitude and longitude
Histalp_stations_df = Histalp_stations_df.merge(
    df_lon_lat[['latitude', 'longitude', 'closest_Imerg_lat', 'closest_Imerg_lon']],
    on=['latitude', 'longitude'], 
    how='left'  # Use 'left' to keep all Histalp_stations_df rows
)
#####################################################################################################
#round the long and lat to be sure
Histalp_stations_df['closest_Imerg_lat'] = Histalp_stations_df['closest_Imerg_lat'].round(6)
Histalp_stations_df['closest_Imerg_lon'] = Histalp_stations_df['closest_Imerg_lon'].round(6)

#Histalp_stations_df['closest_Imerg_lat'] = Histalp_stations_df['closest_Imerg_lat'].apply(lambda x: float(format(x, '.6f')))
#Histalp_stations_df['closest_Imerg_lon'] = Histalp_stations_df['closest_Imerg_lon'].apply(lambda x: float(format(x, '.6f')))

Histalp_stations_df
```

---
### 7.3 Build overall-station mean series, reshape to long time series

**What this does:**  
- Averages monthly values across all stations by year.  
- Converts from wide (12 monthly columns) to long format with a proper datetime index.

```python
Histalp_stations_df_overallstations=Histalp_stations_df.drop(columns= ['longitude', 'latitude', 'station','closest_Imerg_lat','closest_Imerg_lon','height']).groupby('Year').mean()
Histalp_stations_df_overallstations = Histalp_stations_df_overallstations.reset_index()
# Convert wide format (monthly columns) to long format
Histalp_df_long = Histalp_stations_df_overallstations.melt(id_vars=["Year"],var_name="Month", value_name="Value")
# Map month names to numeric values for proper datetime conversion
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

Histalp_df_long["Month"] = Histalp_df_long["Month"].map(month_map)

# Create a datetime column
Histalp_df_long["Date"] = pd.to_datetime(Histalp_df_long[["Year", "Month"]].assign(day=1))

# Sort the DataFrame for proper plotting
Histalp_df_long = Histalp_df_long.sort_values(["Date"])
Histalp_df_long
```

---
### 7.4 Monthly sums/means for ERA5/IMERG + select station points

**What this does:**  
- Rounds model grid coordinates and resamples to monthly.  
- Selects values at the exact IMERG-nearest station points (no nearest search at selection).

```python
era5_Imerg['lat'] = era5_Imerg['lat'].round(2).astype(float)
era5_Imerg['lon'] = era5_Imerg['lon'].round(2).astype(float)
era5_backintime['lat'] = era5_backintime['lat'].round(2).astype(float)
era5_backintime['lon'] = era5_backintime['lon'].round(2).astype(float)


era5_Imerg_filt_month= era5_Imerg.resample(time="1MS").sum()
era5_backintime_filt_month=era5_backintime.resample(time="1MS").sum()
era5_backintime_filt_month
lat_points = xr.DataArray(Histalp_stations_df[['closest_Imerg_lat', 'closest_Imerg_lon']].drop_duplicates()['closest_Imerg_lat'].values, dims="points")
lon_points = xr.DataArray(Histalp_stations_df[['closest_Imerg_lat', 'closest_Imerg_lon']].drop_duplicates()['closest_Imerg_lon'].values, dims="points")

# Select only matching lat/lon pairs
era5_backintime_filt_month_stations = era5_backintime_filt_month.sel(lat=lat_points, lon=lon_points, method=None) # No nearest
era5_Imerg_filt_month_stations = era5_Imerg_filt_month.sel(lat=lat_points, lon=lon_points, method=None)  # No nearest

era5_Imerg_filt_month_stations
```

---
### 7.5 RMSE/R² comparisons vs HISTALP by period & pairwise checks

**What this does:**  
- Computes RMSE and R² for **HISTALP vs ERA5/XGB (1960–2000)** and **HISTALP vs IMERG/ERA5/XGB (2001–2020)**.  
- Also adds pairwise checks like IMERG vs ERA5 and IMERG vs XGB for 2001–2020.  
- Builds bar charts for RMSE and R².

```python
#Select Time
time_start='1960-01-01'  
time_end = '2000-12-31'   
######################
A= Histalp_df_long["Value"][(Histalp_df_long["Date"] >= time_start ) & (Histalp_df_long["Date"] < time_end)].values
A_df = Histalp_df_long[(Histalp_df_long["Date"] >= time_start) & (Histalp_df_long["Date"] < time_end)][['Date', 'Value']].rename(columns={'Value': 'A'})
B_df = era5_backintime_filt_month_stations['ppt_era5'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='B').reset_index()
merged = pd.merge(A_df, B_df, left_on='Date', right_on='time', how='inner')
# Step 4: Compute RMSE
rmse_his_era5_1960_2000 = np.sqrt(np.nanmean((merged['A'] - merged['B']) ** 2))
r2_his_era5_1960_2000 = r2_score(merged['A'], merged['B'])
print('rmse_his_era5_1960_2000: ', rmse_his_era5_1960_2000)
print('r2_his_era5_1960_2000: ', r2_his_era5_1960_2000)
#####################################################################àà
A_df = Histalp_df_long[(Histalp_df_long["Date"] >= time_start) & (Histalp_df_long["Date"] < time_end)][['Date', 'Value']].rename(columns={'Value': 'A'})
B_df = era5_backintime_filt_month_stations['XGB'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='B').reset_index()
merged = pd.merge(A_df, B_df, left_on='Date', right_on='time', how='inner')
# Step 4: Compute RMSE
rmse_his_xgb_1960_2000 = np.sqrt(np.nanmean((merged['A'] - merged['B']) ** 2))
r2_his_xgb_1960_2000 = r2_score(merged['A'], merged['B'])
print('rmse_his_xgb_1960_2000: ', rmse_his_xgb_1960_2000)
print('r2_his_xgb_1960_2000: ', r2_his_xgb_1960_2000)
####################################################################################################################################
#Select Time
time_start='2001-01-01'  
time_end = '2020-12-31'  
######################
A= Histalp_df_long["Value"][(Histalp_df_long["Date"] >= time_start ) & (Histalp_df_long["Date"] < time_end)].values
A_df = Histalp_df_long[(Histalp_df_long["Date"] >= time_start) & (Histalp_df_long["Date"] < time_end)][['Date', 'Value']].rename(columns={'Value': 'A'})
B_df = era5_Imerg_filt_month_stations['Imerg_orgres'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='B').reset_index()
merged = pd.merge(A_df, B_df, left_on='Date', right_on='time', how='inner')
# Step 4: Compute RMSE
rmse_his_Imerg_2001_2020 = np.sqrt(np.nanmean((merged['A'] - merged['B']) ** 2))
r2_his_Imerg_2001_2020 = r2_score(merged['A'], merged['B'])
print('rmse_his_Imerg_2001_2020: ', rmse_his_Imerg_2001_2020)
print('r2_his_Imerg_2001_2020: ', r2_his_Imerg_2001_2020) 
##################################à
B_df = era5_Imerg_filt_month_stations['ppt_era5'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='B').reset_index()
merged = pd.merge(A_df, B_df, left_on='Date', right_on='time', how='inner')
# Step 4: Compute RMSE
rmse_his_era5_2001_2020 = np.sqrt(np.nanmean((merged['A'] - merged['B']) ** 2))
r2_his_era5_2001_2020 = r2_score(merged['A'], merged['B'])
print('rmse_his_era5_2001_2020: ', rmse_his_era5_2001_2020)
print('r2_his_era5_2001_2020: ', r2_his_era5_2001_2020) 

######################
B_df = era5_Imerg_filt_month_stations['XGB'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='B').reset_index()
merged = pd.merge(A_df, B_df, left_on='Date', right_on='time', how='inner')
# Step 4: Compute RMSE
rmse_his_xgb_2001_2020 = np.sqrt(np.nanmean((merged['A'] - merged['B']) ** 2))
r2_his_xgb_2001_2020 = r2_score(merged['A'], merged['B'])
print('rmse_his_xgb_2001_2020: ', rmse_his_xgb_2001_2020)
print('r2_his_xgb_2001_2020:', r2_his_xgb_2001_2020)


# Given RMSE values (replace with actual numbers)
rmse_values = [ rmse_his_era5_1960_2000,rmse_his_era5_2001_2020, rmse_his_xgb_1960_2000,rmse_his_xgb_2001_2020,rmse_his_Imerg_2001_2020,rmse_Imerg_era5_2001_2020,rmse_Imerg_xgb_2001_2020]
labels = [ 'Hist_Era5_1960_2000','Hist_Era5_2001_2020','Hist_XGB_1960_2000', 'Hist_XGB_2001_2020','Hist_Imerg_2001_2020','Imerg_Era5_2001_2020','Imerg_XGB_2001_2020' ]

# Create the plot
plt.figure(figsize=(5, 3))
plt.bar(labels, rmse_values, color=[ 'blue','blue', 'purple', 'purple','green','orange','grey'])
plt.ylabel('RMSE')
plt.title('RMSE Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for i, value in enumerate(rmse_values):
    plt.text(i, value + 0.02 * max(rmse_values), f'{value:.2f}', ha='center', fontsize=12)

plt.xticks(rotation=90)
# Show the plot
plt.show()
# Given RMSE values (replace with actual numbers)
r2_values = [ r2_his_era5_1960_2000,r2_his_era5_2001_2020, r2_his_xgb_1960_2000,r2_his_xgb_2001_2020,r2_his_Imerg_2001_2020,r2_Imerg_era5_2001_2020,r2_Imerg_xgb_2001_2020]
labels = [ 'Hist_Era5_1960_2000','Hist_Era5_2001_2020','Hist_XGB_1960_2000', 'Hist_XGB_2001_2020','Hist_Imerg_2001_2020','Imerg_Era5_2001_2020','Imerg_XGB_2001_2020' ]

# Create the plot
plt.figure(figsize=(5, 3))
plt.bar(labels, r2_values, color=[ 'blue','blue', 'purple', 'purple','green','orange','grey'])
plt.ylabel('R2')
plt.title('R2 Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for i, value in enumerate(r2_values):
    plt.text(i, value + 0.02 * max(r2_values), f'{value:.2f}', ha='center', fontsize=12)

plt.xticks(rotation=90)
# Show the plot
plt.show()
```

> **Note:** The code below also computes pairwise comparisons used above:  
> `rmse_Imerg_era5_2001_2020`, `r2_Imerg_era5_2001_2020`, `rmse_Imerg_xgb_2001_2020`, `r2_Imerg_xgb_2001_2020`.

```python
#Select Time
time_start='2001-01-01'  
time_end = '2020-12-31'   
######################
A= era5_Imerg_filt_month_stations['Imerg_orgres'].sel(time=slice(time_start , time_end)).mean(dim='points').values
B= era5_Imerg_filt_month_stations['ppt_era5'].sel(time=slice(time_start , time_end)).mean(dim='points').values
rmse_Imerg_era5_2001_2020 = np.sqrt(np.nanmean((A - B) ** 2))
r2_Imerg_era5_2001_2020 = r2_score(A, B)
print('rmse_Imerg_era5_2001_2020:', rmse_Imerg_era5_2001_2020)
print('r2_Imerg_era5_2001_2020: ', r2_Imerg_era5_2001_2020)
######################
C= era5_Imerg_filt_month_stations['XGB'].sel(time=slice(time_start , time_end)).mean(dim='points').values
rmse_Imerg_xgb_2001_2020 = np.sqrt(np.nanmean((A - C) ** 2))
r2_Imerg_xgb_2001_2020 = r2_score(A, C)
print('rmse_Imerg_xgb_2001_2020: ', rmse_Imerg_xgb_2001_2020)
print('r2_Imerg_xgb_2001_2020:', r2_Imerg_xgb_2001_2020)
```

---
### 7.6 Train on HISTALP monthly data (preparation)

**What this does:**  
- Merges back-in-time and present-day variables into a single monthly dataset.  
- Aggregates to monthly totals/means (ppt sum; camr max; tciw mean).  
- Selects the station points for later training.

```python
#for the HISTALP (instead of IMERG) training
era5_Imerg['lat'] = era5_Imerg['lat'].round(2).astype(float)
era5_Imerg['lon'] = era5_Imerg['lon'].round(2).astype(float)
era5_backintime['lat'] = era5_backintime['lat'].round(2).astype(float)
era5_backintime['lon'] = era5_backintime['lon'].round(2).astype(float)
########################################################
# Keep only specified variables
internal1 = era5_backintime[['camr', 'tciw', 'ppt_era5']]
internal2 = era5_Imerg[['camr', 'tciw', 'ppt_era5']]
# Merge the datasets
era5_Imerg_totime = xr.merge([internal1, internal2])
#######################################################
# Sum for ppt_era5 over each month
tp_monthly_sum = era5_Imerg_totime['ppt_era5'].resample(time='1MS').sum()
# Mean for camr and tciw over each month
camr_monthly_mean = era5_Imerg_totime[['camr']].resample(time='1MS').max()
tciw_monthly_mean = era5_Imerg_totime[['tciw']].resample(time='1MS').mean()
# Merge the monthly-aggregated datasets
era5_totime_monthly = xr.merge([tp_monthly_sum, camr_monthly_mean, tciw_monthly_mean])

era5_totime_monthly

lat_points = xr.DataArray(Histalp_stations_df[['closest_Imerg_lat', 'closest_Imerg_lon']].drop_duplicates()['closest_Imerg_lat'].values, dims="points")
lon_points = xr.DataArray(Histalp_stations_df[['closest_Imerg_lat', 'closest_Imerg_lon']].drop_duplicates()['closest_Imerg_lon'].values, dims="points")

# Select only matching lat/lon pairs
era5_totime_monthly_stations = era5_totime_monthly.sel(lat=lat_points, lon=lon_points, method=None) # No nearest
era5_totime_monthly_stations
```

---
### 7.7 Convert HISTALP to long format per station and attach as xarray variable

**What this does:**  
- Reshapes the per-station HISTALP series to align with ERA5 monthly times and station points.  
- Inserts the result into the monthly dataset as `histalp_precipitation_monthly`.

```python
# Step 1: Convert Histalp to long format
df = Histalp_stations_df.copy()

# Create a 'month' and 'value' column
df_long = df.melt(
    id_vars=['Year', 'closest_Imerg_lat', 'closest_Imerg_lon'],
    value_vars=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    var_name='month',
    value_name='precip'
)

df_long
# Month mapping to integer
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df_long['month'] = df_long['month'].map(month_map)
df_long
# Create datetime
df_long['time'] = pd.to_datetime(dict(year=df_long['Year'], month=df_long['month'], day=1))
df_long
df_long_NA=df_long[df_long['precip'].isna()]
df_long_NA
# Step 2: Group by station coordinate
grouped = df_long.groupby(['closest_Imerg_lat', 'closest_Imerg_lon'])

# Step 3: Create array to match shape of ERA5
era5_times = era5_totime_monthly_stations.time.values
points = len(era5_totime_monthly_stations.points)
histalp_values = np.full((len(era5_times), points), np.nan)

# Step 4: Match and fill values
for i in range(points):
    lat = float(era5_totime_monthly_stations.lat[i].values)
    lon = float(era5_totime_monthly_stations.lon[i].values)

    try:
        group = grouped.get_group((lat, lon))
        group = group.set_index('time').sort_index()
        # Align with ERA5 time
        group_aligned = group.reindex(era5_times)
        histalp_values[:, i] = group_aligned['precip'].values
    except KeyError:
        pass  # No matching station
# Step 5: Add to xarray as new variable
histalp_da = xr.DataArray(
    data=histalp_values,
    dims=['time', 'points'],
    coords={'time': era5_times, 'points': era5_totime_monthly_stations.points},
    name='histalp_precipitation_monthly')

era5_totime_monthly_stations['histalp_precipitation_monthly'] = histalp_da
era5_totime_monthly_stations
#Dealing with 6percent of NA values

era5_totime_monthly_stations_clean = era5_totime_monthly_stations#.dropna(dim="time", subset=["histalp_precipitation_monthly"])

era5_totime_monthly_stations_clean
```

---
### 7.8 CHAPTER dataset: open, monthly sum, nearest cells to stations, and map

**What this does:**  
- Loads CHAPTER files, drops bounds, removes Feb 29, and resamples to monthly sums.  
- Finds the nearest CHAPTER grid cell for each station.  
- Plots station locations over the study domain.

```python
file_list = [os.path.join(f"{path}/CHAPTER", f) for f in os.listdir(f"{path}/CHAPTER") if f.endswith('.nc')]
chapter = xr.open_mfdataset(file_list,combine='by_coords',chunks={'time': 1},parallel=True)
chapter=chapter.drop_vars('XTIME_bnds')

#removing 29th of Feb to have a same data for all years
chapter = chapter.sel(XTIME=~((chapter['XTIME'].dt.month == 2) & (chapter['XTIME'].dt.day == 29)))
chapter

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km
def find_closest_point_chapter(chapter, station_latitude, station_longitude):
    # Compute squared Euclidean distance (avoid sqrt for efficiency)
    dist = (chapter.XLAT.compute() - station_latitude) ** 2 + (chapter.XLONG.compute() - station_longitude) ** 2

    # Find the indices of the minimum distance
    min_index = np.unravel_index(dist.argmin(), dist.shape)
    i, j = min_index

    # Extract closest latitude and longitude
    closest_lat = chapter.XLAT.isel(south_north=i, west_east=j).compute().item()
    closest_lon = chapter.XLONG.isel(south_north=i, west_east=j).compute().item()

    return closest_lat, closest_lon  # Return float values
chapter_month= chapter.resample(XTIME="1MS").sum()
# chapter_month = chapter_month.compute()
chapter_month
#closest Grid point of era5_Imerg based on stations' positions
df_lon_lat= Histalp_stations_df[['latitude', 'longitude']].drop_duplicates()

# Loop through each row 
closest_latitudes = []
closest_longitudes = []

for index, row in tqdm(df_lon_lat.iterrows()):
    closest_lat, closest_lon = find_closest_point_chapter(chapter_month, row['latitude'], row['longitude'])
    closest_latitudes.append(closest_lat)
    closest_longitudes.append(closest_lon)

# Add the new columns to the DataFrame
df_lon_lat['closest_chapter_lat'] = closest_latitudes
df_lon_lat['closest_chapter_lon'] = closest_longitudes
#####################################################################################################
# Merge Histalp_stations_df with df_lon_lat on latitude and longitude
Histalp_stations_df = Histalp_stations_df.merge(
    df_lon_lat[['latitude', 'longitude', 'closest_chapter_lat', 'closest_chapter_lon']],
    on=['latitude', 'longitude'], 
    how='left'  # Use 'left' to keep all Histalp_stations_df rows
)

Histalp_stations_df
#Map of Closest points 

Histalp_stations_lon= Histalp_stations_df['longitude'].values
Histalp_stations_lat= Histalp_stations_df['latitude'].values

closest_Imerg_lon= Histalp_stations_df['closest_Imerg_lon'].values
closest_Imerg_lat= Histalp_stations_df['closest_Imerg_lat'].values

closest_chapter_lon= Histalp_stations_df['closest_chapter_lon'].values
closest_chapter_lat= Histalp_stations_df['closest_chapter_lat'].values

era5_Imerg_lon=era5_Imerg.lon.values
era5_Imerg_lat=era5_Imerg.lat.values
##############################
# Set up the map
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAKES, facecolor='lightblue')
#ax.add_feature(cfeature.RIVERS)
ax.set_extent([np.min(era5_Imerg_lon), np.max(era5_Imerg_lon), np.min(era5_Imerg_lat), np.max(era5_Imerg_lat)], crs=ccrs.PlateCarree())



# Gridlines with larger font size and same color as title
title_color = '#000000'  # Title color
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False
gl.xlabel_style = {'size': 15, 'color': title_color}
gl.ylabel_style = {'size': 15, 'color': title_color}
# label height
levels = np.arange(0, 3000, 500)
ax = fig.add_subplot(projection=crs)
contour = ax.contour(era5_Imerg['lon'], era5_Imerg['lat'], era5_elev, levels, colors='k', linewidths=0.5)
ax.add_feature(cfeature.COASTLINE)




# Plot each dataset
ax.scatter(Histalp_stations_lon, Histalp_stations_lat, color='red', marker='o',s=50, label='HISTALP Stations')
#ax.scatter(closest_Imerg_lon, closest_Imerg_lat, color='red', marker='x',s=50, label='Closest IMERG Points')
#ax.scatter(closest_chapter_lon, closest_chapter_lat, color='blue', marker='^',s=50, label='Closest CHAPTER Points')
#ax.scatter(era5_Imerg_lon, era5_Imerg_lat, color='green', marker='s', label='ERA5 IMERG Grid')

# Add a legend
plt.legend(loc='upper left')

# Add title
plt.title('', fontsize=20, fontweight='bold', color=title_color)


plt.show()
```

---
### 7.9 Extract CHAPTER station time series & plot HISTALP overview

**What this does:**  
- For each station point, extracts the nearest CHAPTER monthly precipitation time series.  
- Plots an overview line of HISTALP monthly precipitation across the requested period.

```python
# List to store the extracted precipitation values
all_chapter_month_stations = []

# Iterate over each unique closest_lat and closest_lon
for closest_lat, closest_lon in zip(lat_points, lon_points):
    # Compute squared distance
    dist = (chapter.XLAT.compute() - closest_lat) ** 2 + (chapter.XLONG.compute() - closest_lon) ** 2

    # Find the indices of the minimum distance
    i, j = np.unravel_index(dist.argmin(), dist.shape)

    # Extract the precipitation time series at the closest grid cell
    chapter_month_station = chapter_month.isel(south_north=i, west_east=j)

    # Store the extracted data along with coordinates
    all_chapter_month_stations.append(chapter_month_station)

# Combine results into a new xarray Dataset or DataArray if needed
chapter_month_stations = xr.concat(all_chapter_month_stations, dim="points")


#chapter_month_stations=chapter_month_stations.compute()


# Display the final dataset with all selected points
chapter_month_stations

#Select Time
time_start='1960-01-01'   #'1960-01-01'   '1995-01-01'
time_end = '2020-12-31'   #'2020-12-31'   '2004-12-31'
########################################
plt.figure(figsize=(15, 5))

# Histalp Data (1960-2021) - Orange Solid Line
plt.plot(Histalp_df_long["Date"][(Histalp_df_long["Date"] > time_start) & (Histalp_df_long["Date"] < time_end)], Histalp_df_long["Value"][(Histalp_df_long["Date"] > time_start) & (Histalp_df_long["Date"] < time_end)], color='orange',
         linestyle='-', label='HISTALP')

# Formatting
plt.xlabel("Year", fontsize=20)
plt.ylabel("Monthly precipitation (mm)", fontsize=20)
plt.title("")

#plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=15)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()
```

---
### 7.10 Build filled ERA5 series (back-in-time + present), align XGB & CHAPTER; compute errors and KDE

**What this does:**  
- Combines ERA5 present and back-in-time (outer join, then fills present with back-in-time where missing).  
- Does the same for XGB.  
- Merges HISTALP, ERA5, XGB, CHAPTER to compute ME/MAE and plot differences and KDEs.

```python
era5_Imerg_filt_month_stations
Era5_ts = era5_Imerg_filt_month_stations['ppt_era5'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='ERA5').reset_index()
Era5_ts
# Create DataFrames from the xarray DataArrays
Era5_ts1 = era5_Imerg_filt_month_stations['ppt_era5'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='ERA5').reset_index()
Era5_ts2 = era5_backintime_filt_month_stations['ppt_era5'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='ERA5').reset_index()
merged = Era5_ts1.merge(Era5_ts2, on='time', how='outer', suffixes=('_present', '_backintime'))
merged['ERA5_filled'] = merged['ERA5_present'].fillna(merged['ERA5_backintime'])
Era5_ts = merged[['time', 'ERA5_filled']].rename(columns={'ERA5_filled': 'ERA5'})
Era5_ts

#Select Time
time_start='1960-01-01'  
time_end = '2020-12-31'   
######################
HISTALP_ts = Histalp_df_long[(Histalp_df_long["Date"] >= time_start) & (Histalp_df_long["Date"] < time_end)][['Date', 'Value']].rename(columns={'Value': 'HISTALP'})
XGB_ts = era5_backintime_filt_month_stations['XGB'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='XGB').reset_index()
CHAPTER_ts = chapter_month_stations['PREC_ACC_NC'].sel(XTIME=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='CHAPTER').reset_index()
#######
Era5_ts1 = era5_Imerg_filt_month_stations['ppt_era5'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='ERA5').reset_index()
Era5_ts2 = era5_backintime_filt_month_stations['ppt_era5'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='ERA5').reset_index()
merged = Era5_ts1.merge(Era5_ts2, on='time', how='outer', suffixes=('_present', '_backintime'))
merged['ERA5_filled'] = merged['ERA5_present'].fillna(merged['ERA5_backintime'])
Era5_ts = merged[['time', 'ERA5_filled']].rename(columns={'ERA5_filled': 'ERA5'})
##############################
XGB_ts1 = era5_Imerg_filt_month_stations['XGB'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='XGB').reset_index()
XGB_ts2 = era5_backintime_filt_month_stations['XGB'].sel(time=slice(time_start , time_end)).mean(dim="points").to_dataframe(name='XGB').reset_index()
merged = XGB_ts1.merge(XGB_ts2, on='time', how='outer', suffixes=('_present', '_backintime'))
merged['XGB_filled'] = merged['XGB_present'].fillna(merged['XGB_backintime'])
XGB_ts = merged[['time', 'XGB_filled']].rename(columns={'XGB_filled': 'XGB'})
##############################
merged = pd.merge(HISTALP_ts, Era5_ts, left_on='Date', right_on='time', how='left')
merged = pd.merge(merged, XGB_ts, left_on='Date', right_on='time', how='left')
#merged = pd.merge(merged, CHAPTER_ts, left_on='Date', right_on='XTIME', how='inner')
merged = pd.merge(merged, CHAPTER_ts, left_on='Date', right_on='XTIME', how='left')


merged['Era5-HIS']=merged['ERA5']-merged['HISTALP']
merged['XGB-HIS']=merged['XGB']-merged['HISTALP']
merged['CHAPTER-HIS']=merged['CHAPTER']-merged['HISTALP']
merged
#Errors
MAE_Era5 = np.mean(np.abs(merged['Era5-HIS']))
ME_Era5 = np.mean(merged['Era5-HIS']) 
print(MAE_Era5)
print(ME_Era5)
MAE_XGB = np.mean(np.abs(merged['XGB-HIS']))
ME_XGB = np.mean(merged['XGB-HIS']) 
print(MAE_XGB)
print(ME_XGB)
MAE_CHAPTER = np.mean(np.abs(merged['CHAPTER-HIS']))
ME_CHAPTER = np.mean(merged['CHAPTER-HIS']) 
print(MAE_CHAPTER)
print(ME_CHAPTER)
# Plot with specific colors
plt.figure(figsize=(15, 5))
plt.plot(merged['Date'], merged['Era5-HIS'], label=f'ERA5-HISTALP (ME={ME_Era5:.1f} mm)', color='blue')
plt.plot(merged['Date'], merged['XGB-HIS'], label=f'XGB-HISTALP (ME={ME_XGB:.1f} mm)', color='purple') 
plt.plot(merged['Date'], merged['CHAPTER-HIS'], label=f'CHAPTER-HISTALP (ME={ME_CHAPTER:.1f} mm)', color='black') 

# Labels and formatting
plt.xlabel('Year',fontsize=20)
plt.ylabel('Difference in Precipitation',fontsize=20)
plt.title('')
plt.legend(fontsize=15)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(7, 5))

sns.kdeplot(merged['ERA5'], color='blue', label='ERA5', linewidth=1)
sns.kdeplot(merged['HISTALP'], color='orange', label='HISTALP', linewidth=2)
sns.kdeplot(merged['XGB'], color='purple', label='XGB', linewidth=1)
sns.kdeplot(merged['CHAPTER'], color='black', label='CHAPTER', linewidth=1)

plt.xlabel('Monthly Precipitation (mm)', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=16)
plt.show()
```

---
### 7.11 HISTALP vs CHAPTER (1983–2000) and comparative bars

**What this does:**  
- Compares HISTALP and CHAPTER over 1983–2000 across the station points.  
- Plots bar charts comparing RMSE and R² for **ERA5**, **CHAPTER**, and **XGB**.

```python
#Select Time
time_start='1983-01-01'  
time_end = '2000-12-31' #'2000-12-31'   
# Step 1: Convert xarray to pandas DataFrame
B_df = chapter_month_stations['PREC_ACC_NC'].sel(XTIME=slice(time_start , time_end))     .mean(dim="points")     .to_dataframe(name='B')     .reset_index()

# Step 2: Prepare Histalp data
A_df = Histalp_df_long[(Histalp_df_long["Date"] >= time_start) & 
                       (Histalp_df_long["Date"] < time_end)][['Date', 'Value']].rename(columns={'Value': 'A'})

# Step 3: Merge on Date
merged_pd = pd.merge(A_df, B_df, left_on='Date', right_on='XTIME', how='inner')

# Step 4: Compute RMSE
rmse_his_chapter_1983_2000 = np.sqrt(np.nanmean((merged_pd['A'] - merged_pd['B']) ** 2))
r2_his_chapter_1983_2000 = r2_score(merged_pd['A'], merged_pd['B'])
print('rmse_his_chapter_1983_2000: ', rmse_his_chapter_1983_2000)
print('r2_his_chapter_1983_2000: ', r2_his_chapter_1983_2000)

# Given RMSE values (replace with actual numbers)
rmse_values = [ rmse_his_era5_1960_2000, rmse_his_chapter_1983_2000, rmse_his_xgb_1960_2000]
labels = ['ERA5','CHAPTER', 'XGB' ]

# Create the plot
plt.figure(figsize=(2.5, 5))
plt.bar(labels, rmse_values, color=[ 'blue', 'black', 'purple'])  
plt.ylabel('RMSE mm/month', fontsize=20)
plt.title('')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for i, value in enumerate(rmse_values):
    plt.text(i, value + 0.02 * max(rmse_values), f'{value:.2f}', ha='center', fontsize=15)

plt.xticks(rotation=90, fontsize=15)
plt.yticks(rotation=90, fontsize=15)
# Show the plot
plt.show()
# Given RMSE values (replace with actual numbers)
r2_values = [ r2_his_era5_1960_2000, r2_his_chapter_1983_2000, r2_his_xgb_1960_2000]
labels = ['ERA5','CHAPTER', 'XGB' ]

# Create the plot
plt.figure(figsize=(2.5, 5))
plt.bar(labels, r2_values, color=[ 'blue', 'black', 'purple'])  
plt.ylabel('R2', fontsize=20)
plt.title('')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for i, value in enumerate(r2_values):
    plt.text(i, value + 0.02 * max(r2_values), f'{value:.2f}', ha='center', fontsize=15)

plt.xticks(rotation=90, fontsize=15)
plt.yticks(rotation=90, fontsize=15)
# Show the plot
plt.show()
```
