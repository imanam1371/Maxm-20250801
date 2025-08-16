# CDO Processing Documentation

This document describes the complete set of Climate Data Operators (CDO) operations performed across the provided Bash scripts.  
It summarizes datasets, variables, remapping methods, processing steps, and derived calculations.

---

## 1. Overview

The workflow involves:
- **Subsetting and preprocessing** precipitation and other atmospheric variables from IMERG and ERA5 datasets.
- **Temporal aggregation** (daily sums, daily means, daily maxima, daily minima).
- **Spatial remapping** to a target grid (either *conservative remapping* or *nearest neighbor* mapping).
- **Derived variable computation** (e.g., CAMR, Q2m, upslope wind, shear).

Data sources:
- **IMERG** (precipitation data)
- **ERA5** (2001 to 2021)(hourly/single level/pressure level data)
- **ERA5 (1960 to 2000)**

Target grids:
- `Imerg_grid.txt`

Remapping methods:
- **Conservative remapping (`remapcon`)**: Preserves integral quantities like precipitation totals when changing grids. Used mainly for precipitation.
- **Nearest neighbor (`remapnn`)**: Preserves exact original values without interpolation. Used for variables where conservation is less critical.
---

## 2. Script-by-Script Breakdown

### 2.1 `bsh_Imerg_long_lat.sh`
**Purpose:** Extract precipitation from IMERG files over a lat/lon box (4°E–19°E, 43°N–49°N).

**Steps:**
1. Loop daily.
2. For each day:
   - Read IMERG `.nc4` file.
   - Select variable `precipitation`.
   - Spatial subset to bounding box.
   - Save to output directory.


### bsh_Imerg_long_lat.sh
```bash
#!/bin/bash

# Path to your data directory (converted Windows path)
path=''

# Loop over dates (seq is not ideal for dates, so using date command in loop)
start_date="20150101"
end_date="20190101"

current_date=$start_date
while [[ "$current_date" -le "$end_date" ]]; do
    echo $current_date
    
    # Set input and output file paths
    input_file="$path/3B-DAY.MS.MRG.3IMERG.$current_date-S000000-E235959.V07B.nc4"
    output_file="$path/CDO_Precipitation_3B-DAY.MS.MRG.3IMERG.$current_date-S000000-E235959.V07B.nc4"

    # Check if input file exists before processing
    if [[ -f $input_file ]]; then
        # Extract precipitation and subset latitude and longitude
        cdo -b F32 -selname,precipitation \
            -sellonlatbox,4,19,43,49 \
            $input_file $output_file
    else
        echo "File $input_file does not exist."
    fi

    # Increment date by 1 day (using date command for proper date increment)
    current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
done
```
---

### 2.2 `bsh_conservative mapping.sh`
**Purpose:** Remap daily IMERG precipitation to a target grid using *conservative remapping*.

**Steps:**
1. Loop daily.
2. For each day:
   - Read processed IMERG precipitation file.
   - Apply `cdo remapcon` with `Imerg_grid.txt`.


### bsh_conservative mapping.sh
```bash
#!/bin/bash

# Path to your data directory (converted Windows path)
path=''

# Loop over dates (using date command to handle date ranges)
start_date=""
end_date=""

current_date=$start_date
while [[ "$current_date" -le "$end_date" ]]; do
    echo "Processing date: $current_date"
    
    # Set input and output file paths
    input_file="$path/CDO_Precipitation_3B-DAY.MS.MRG.3IMERG.$current_date-S000000-E235959.V07B.nc4"
    output_file="$path/CDO_Precipitation_consmap_3B-DAY.MS.MRG.3IMERG.$current_date-S000000-E235959.V07B.nc4"

    # Check if input file exists before processing
    if [[ -f $input_file ]]; then
        # Upscaling with conservative remapping
        cdo remapcon,//Imerg_grid.txt $input_file $output_file

        echo "Processed $input_file -> $output_file"
    else
        echo "File $input_file does not exist. Skipping."
    fi

    # Increment date by 1 day
    current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
done

```


---

### 2.3 `bsh_backintime_2.sh`
**Purpose:** Process ERA5 data.

**Variables processed:**
- `tp` (total precipitation) — daily sum → conservative remap.
- `tciw` (total column cloud ice water) — daily mean → nearest neighbor remap.
- `sp` (surface pressure) — daily mean → nearest neighbor remap.
- `tcwv` (total column water vapour) — daily mean → nearest neighbor remap.
- `CAMR` (tcwv / (sp / 9.81)) — daily mean → nearest neighbor remap.

**Notes:** Uses both `remapcon` and `remapnn` depending on variable.

### bsh_backintime_2.sh
```bash
#!/bin/bash

path=''

for y in $(seq 1960 1984); do
    echo $y

    # Daily total precipitation (tp)
    cdo -s -b F32 -daysum -selname,tp "$path/ERA5_hourly_backintime/total_precipitation_${y}.nc" "$path/ERA5_hourly_backintime/total_precipitation_${y}_daysum.nc"
    # Remapping (Consmap)
    cdo remapcon,"/mnt/c/Iman/Maxm/Imerg_grid.txt" "$path/ERA5_hourly_backintime/total_precipitation_${y}_daysum.nc" "$path/ERA5_hourly_backintime/total_precipitation_${y}_daysum_consmap.nc"
######################################################################################################################
    # tciw
    cdo -b F32 -daymean -selname,tciw "$path/ERA5_hourly_backintime/total_column_cloud_ice_water_${y}.nc" "$path/ERA5_hourly_backintime/total_column_cloud_ice_water_${y}_daymean.nc"
    # Remapping (NN)
    cdo remapnn,"/mnt/c/Iman/Maxm/Imerg_grid.txt" "$path/ERA5_hourly_backintime/total_column_cloud_ice_water_${y}_daymean.nc" "$path/ERA5_hourly_backintime/total_column_cloud_ice_water_${y}_daymean_NN.nc"
######################################################################################################################
    # sp
    cdo -b F32 -daymean -selname,sp "$path/ERA5_hourly_backintime/surface_pressure_${y}.nc" "$path/ERA5_hourly_backintime/surface_pressure_${y}_daymean.nc"
    # Remapping (NN)
    cdo remapnn,"/mnt/c/Iman/Maxm/Imerg_grid.txt" "$path/ERA5_hourly_backintime/surface_pressure_${y}_daymean.nc" "$path/ERA5_hourly_backintime/surface_pressure_${y}_daymean_NN.nc"
######################################################################################################################
    # tcwv
    cdo -b F32 -daymean -selname,tcwv "$path/ERA5_hourly_backintime/total_column_water_vapour_${y}.nc" "$path/ERA5_hourly_backintime/total_column_water_vapour_${y}_daymean.nc"
    # Remapping (NN)
    cdo remapnn,"/mnt/c/Iman/Maxm/Imerg_grid.txt" "$path/ERA5_hourly_backintime/total_column_water_vapour_${y}_daymean.nc" "$path/ERA5_hourly_backintime/total_column_water_vapour_${y}_daymean_NN.nc"
######################################################################################################################
    # camr
    # Merge
    cdo merge "$path/ERA5_hourly_backintime/total_column_water_vapour_${y}.nc" "$path/ERA5_hourly_backintime/surface_pressure_${y}.nc" "$path/Temporary/merged_tcwv_sp_${y}.nc"
    # Calculation
    cdo -b F32 -daymean -expr,'camr=tcwv/(sp/9.81)' "$path/Temporary/merged_tcwv_sp_${y}.nc" "$path/ERA5_hourly_backintime/camr_${y}_daymean.nc"
    # Remapping (NN)
    cdo remapnn,"/mnt/c/Iman/Maxm/Imerg_grid.txt" "$path/ERA5_hourly_backintime/camr_${y}_daymean.nc" "$path/ERA5_hourly_backintime/camr_${y}_daymean_NN.nc"

done

```
---

### 2.4 `bsh_ERA5_consmap_tp.sh`
**Purpose:** Conservative remapping of ERA5 total precipitation.

**Steps:**
- Variable: `total_precipitation_daycum`
- Apply `remapcon` with `Imerg_grid.txt`.


### bsh_ERA5_consmap_tp.sh
```bash
#!/bin/bash

# Path to your data directory (converted Windows path)
path=''

# Define years array
iyear=(1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021)

# Define variable names
ivar=('total_precipitation_daycum')

# Loop through variables and years
for var in "${ivar[@]}"; do
    for year in "${iyear[@]}"; do  
        # Set input and output file paths
        input_file="$path/ERA5_before_NearestN/${var}_${year}.nc"
        output_file="$path/ERA5_consmap/${var}_${year}_consmap.nc"

        # Perform the remapping with CDO
        cdo remapcon,/mnt/c/Iman/Maxm/Imerg_grid.txt "$input_file" "$output_file"
    done
done

```
---

### 2.5 `bsh_NearestN mapping (1).sh`
**Purpose:** Apply *nearest neighbor* remapping to many daily aggregated ERA5 variables.

**Steps:**
- Loop over years and variables.
- Remap using `remapnn` to `Imerg_grid.txt`.

### bsh_NearestN mapping (1).sh
```bash
#!/bin/bash

# Path to your data directory (converted Windows path)
path=''

# Define years array
iyear=(1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021)

# Define variable names
ivar=(
    '10m_u_component_of_wind_daymean' 
    '10m_v_component_of_wind_daymean'
    '2m_dewpoint_temperature_daymean'
    '2m_temperature_daymax'
    'convective_available_potential_energy_daymax'
    'convective_inhibition_daymin'
    'geopotential_daymean'
    'k_index_daymean'
    'mean_surface_latent_heat_flux_daymean'
    'mean_surface_sensible_heat_flux_daymean'
    'mean_vertically_integrated_moisture_divergence_daymean'
    'sea_surface_temperature_daymean'
    'surface_pressure_daymean'
    'toa_incident_solar_radiation_daymean'
    'total_column_cloud_liquid_water_daymean'
    'total_column_cloud_ice_water_daymean'   
    'total_column_water_vapour_daymean'
    'total_totals_index_daymean'
    'vertical_integral_of_eastward_water_vapour_flux_daymean'
    'vertical_integral_of_northward_water_vapour_flux_daymean'   
    'volumetric_soil_water_layer_1_daymean'
    'total_precipitation_daycum'
    'temperature_daymean' 
    'u_component_of_wind_daymean'
    'v_component_of_wind_daymean'
    'vertical_velocity_daymean'
    'vorticity_daymean'
    'q2m_daymean'
    'camr_daymean'
    'bs_daymax'
    'WVT_daymean'
)

# Loop through variables and years
for var in "${ivar[@]}"; do
    for year in "${iyear[@]}"; do  
        # Set input and output file paths
        input_file="$path/ERA5_before_NearestN/${var}_${year}.nc"
        output_file="$path/ERA5_NearestN/${var}_${year}_NearestN.nc"

        # Perform the remapping with CDO
        cdo remapnn,/mnt/c/Iman/Maxm/Imerg_grid.txt "$input_file" "$output_file"
    done
done

```


---

### 2.6 `bsh_NearestN mapping (2).sh`
**Purpose:** Daily aggregation and derived variable computation from raw ERA5 single/pressure level data.

**Processing:**
- **Daily means**: wind, geopotential, pressure, SST, TCWV, dewpoint, soil water, cloud water/ice, moisture divergence, radiation, TT index, etc.
- **Derived variables:**
  - Q2m: specific humidity formula from dewpoint & surface pressure.
  - CAMR: column water vapour ratio.
  - **Upslope wind**: computed from wind × geopotential gradient.
  - **Shear (bs)**: difference between 925 hPa and 500 hPa winds.

Results are stored in dedicated `*_CDO` folders.
### bsh_NearestN mapping (2).sh
```bash

path=''

for y in $(seq 1983 1993); do
	echo $y
	#single level
	cdo -b F32 -daymean -selname,u10 $path/'ERA5_single_level/single_level_10m_u_component_of_wind_'$y'.nc' $path/'ERA5_single_level_CDO/10m_u_component_of_wind_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,v10 $path/'ERA5_single_level/single_level_10m_v_component_of_wind_'$y'.nc' $path/'ERA5_single_level_CDO/10m_v_component_of_wind_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,z $path/'ERA5_single_level/single_level_geopotential_'$y'.nc' $path/'ERA5_single_level_CDO/geopotential_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,sp $path/'ERA5_single_level/single_level_surface_pressure_'$y'.nc' $path/'ERA5_single_level_CDO/surface_pressure_daymean_'$y'.nc'
	cdo -b F32 -daymax -selname,cape $path/'ERA5_single_level/single_level_convective_available_potential_energy_'$y'.nc' $path/'ERA5_single_level_CDO/convective_available_potential_energy_daymax_'$y'.nc'
	cdo -b F32 -daymean -selname,t2m $path/'ERA5_single_level/single_level_2m_temperature_'$y'.nc' $path/'ERA5_single_level_CDO/2m_temperature_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,sst $path/'ERA5_single_level/single_level_sea_surface_temperature_'$y'.nc' $path/'ERA5_single_level_CDO/sea_surface_temperature_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,tcwv $path/'ERA5_single_level/single_level_total_column_water_vapour_'$y'.nc' $path/'ERA5_single_level_CDO/total_column_water_vapour_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,d2m $path/'ERA5_single_level/single_level_2m_dewpoint_temperature_'$y'.nc' $path/'ERA5_single_level_CDO/2m_dewpoint_temperature_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,swvl1 $path/'ERA5_single_level/single_level_volumetric_soil_water_layer_1_'$y'.nc' $path/'ERA5_single_level_CDO/volumetric_soil_water_layer_1_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,tclw $path/'ERA5_single_level/single_level_total_column_cloud_liquid_water_'$y'.nc' $path/'ERA5_single_level_CDO/total_column_cloud_liquid_water_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,tciw $path/'ERA5_single_level/single_level_total_column_cloud_ice_water_'$y'.nc' $path/'ERA5_single_level_CDO/total_column_cloud_ice_water_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,p71.162 $path/'ERA5_single_level/single_level_vertical_integral_of_eastward_water_vapour_flux_'$y'.nc' $path/'ERA5_single_level_CDO/vertical_integral_of_eastward_water_vapour_flux_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,p72.162 $path/'ERA5_single_level/single_level_vertical_integral_of_northward_water_vapour_flux_'$y'.nc' $path/'ERA5_single_level_CDO/vertical_integral_of_northward_water_vapour_flux_daymean_'$y'.nc'
	#Unknowns
	cdo -b F32 -daymin -selname,cin $path/'ERA5_single_level/single_level_convective_inhibition_'$y'.nc' $path/'ERA5_single_level_CDO/convective_inhibition_daymin_'$y'.nc'
	cdo -b F32 -daymean -selname,kx $path/'ERA5_single_level/single_level_k_index_'$y'.nc' $path/'ERA5_single_level_CDO/k_index_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,mslhf $path/'ERA5_single_level/single_level_mean_surface_latent_heat_flux_'$y'.nc' $path/'ERA5_single_level_CDO/mean_surface_latent_heat_flux_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,msshf $path/'ERA5_single_level/single_level_mean_surface_sensible_heat_flux_'$y'.nc' $path/'ERA5_single_level_CDO/mean_surface_sensible_heat_flux_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,mvimd $path/'ERA5_single_level/single_level_mean_vertically_integrated_moisture_divergence_'$y'.nc' $path/'ERA5_single_level_CDO/mean_vertically_integrated_moisture_divergence_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,tisr $path/'ERA5_single_level/single_level_toa_incident_solar_radiation_'$y'.nc' $path/'ERA5_single_level_CDO/toa_incident_solar_radiation_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,totalx $path/'ERA5_single_level/single_level_total_totals_index_'$y'.nc' $path/'ERA5_single_level_CDO/total_totals_index_daymean_'$y'.nc'
	#pressure level
	cdo -b F32 -daymean -selname,t $path/'ERA5_500hpa/500hpa_level_temperature_'$y'.nc' $path/'ERA5_500hpa_CDO/temperature_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,u $path/'ERA5_500hpa/500hpa_level_u_component_of_wind_'$y'.nc' $path/'ERA5_500hpa_CDO/u_component_of_wind_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,v $path/'ERA5_500hpa/500hpa_level_v_component_of_wind_'$y'.nc' $path/'ERA5_500hpa_CDO/v_component_of_wind_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,w $path/'ERA5_500hpa/500hpa_level_vertical_velocity_'$y'.nc' $path/'ERA5_500hpa_CDO/vertical_velocity_daymean_'$y'.nc'
	cdo -b F32 -daymean -selname,vo $path/'ERA5_500hpa/500hpa_level_vorticity_'$y'.nc' $path/'ERA5_500hpa_CDO/vorticity_daymean_'$y'.nc'

	#OTHERS
	#1_ Specific humidity
	cdo merge $path/'ERA5_single_level/single_level_2m_dewpoint_temperature_'$y'.nc' $path/'ERA5_single_level/single_level_surface_pressure_'$y'.nc' $path/'Temporary/merged_d2m_sp_'$y'.nc'
	cdo -b F32 -daymean -expr,'q2m=(0.622*(611.12*exp(5420*(1/273.15-1/d2m))))/(sp-0.378*(611.12*exp(5420*(1/273.15-1/d2m))))' $path/'Temporary/merged_d2m_sp_'$y'.nc' $path/'ERA5_other_var_CDO/q2m_daymean_'$y'.nc'

	#2_ camr
	cdo merge $path/'ERA5_single_level/single_level_total_column_water_vapour_'$y'.nc' $path/'ERA5_single_level/single_level_surface_pressure_'$y'.nc' $path/'Temporary/merged_tcwv_sp_'$y'.nc'
	cdo -b F32 -daymean -expr,'camr = tcwv/(sp/9.81)' $path/'Temporary/merged_tcwv_sp_'$y'.nc' $path/'ERA5_other_var_CDO/camr_daymean_'$y'.nc'
	
	#3_ upslope wind
		#derivative
	cdo dx $path/ERA5_single_level/single_level_geopotential_$y.nc $path/'Temporary/dz_dx_'$y'.nc'
	cdo dy $path/ERA5_single_level/single_level_geopotential_$y.nc $path/'Temporary/dz_dy_'$y'.nc'
		#multiply
	cdo mul $path/ERA5_single_level/single_level_10m_u_component_of_wind_$y.nc $path/'Temporary/dz_dx_'$y'.nc' $path/'Temporary/u10_dz_dx_'$y'.nc'
	cdo mul $path/ERA5_single_level/single_level_10m_v_component_of_wind_$y.nc $path/'Temporary/dz_dy_'$y'.nc' $path/'Temporary/v10_dz_dy_'$y'.nc'
		#add
	cdo add $path/'Temporary/u10_dz_dx_'$y'.nc' $path/'Temporary/v10_dz_dy_'$y'.nc' $path/'Temporary/upslope_'$y'.nc' 
		#daily
	cdo -b F32 -daymean -selname,upslope $path/'Temporary/upslope_'$y'.nc' $path/'ERA5_other_var_CDO/upslope_daymean_'$y'.nc'


	#4_ Shear
	cdo chname,u,u925 $path/'ERA5_other_var/925hpa_level_u_component_of_wind_'$y'.nc' $path/'Temporary/renamed_925hpa_u_component_of_wind_'$y'.nc'
	cdo chname,v,v925 $path/'ERA5_other_var/925hpa_level_v_component_of_wind_'$y'.nc' $path/'Temporary/renamed_925hpa_v_component_of_wind_'$y'.nc'
	cdo chname,u,u500 $path/'ERA5_500hpa/500hpa_level_u_component_of_wind_'$y'.nc' $path/'Temporary/renamed_500hpa_u_component_of_wind_'$y'.nc'
	cdo chname,v,v500 $path/'ERA5_500hpa/500hpa_level_v_component_of_wind_'$y'.nc' $path/'Temporary/renamed_500hpa_v_component_of_wind_'$y'.nc'
	cdo merge $path/'Temporary/renamed_925hpa_u_component_of_wind_'$y'.nc' $path/'Temporary/renamed_925hpa_v_component_of_wind_'$y'.nc' $path/'Temporary/renamed_500hpa_u_component_of_wind_'$y'.nc' $path/'Temporary/renamed_500hpa_v_component_of_wind_'$y'.nc' $path/'Temporary/merged_u_v_500_925_'$y'.nc'
	cdo -b F32 -daymax -expr,'bs= sqrt((u925-u500)^2+(v925-v500)^2)' $path/'Temporary/merged_u_v_500_925_'$y'.nc' $path/'ERA5_other_var_CDO/bs_daymax_'$y'.nc'

```


## 3. Notes

- All scripts are designed to skip missing input files.
- All operations assume NetCDF inputs (`.nc` or `.nc4`).
- Precision often set to `-b F32`.


