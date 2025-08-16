# ML-IMEX: A Machine Learning-Based Backward Extension of IMERG Precipitation Over the Greater Alpine Region

## Overview

Accurate and high-resolution precipitation estimates are vital for climate research and hydrological modeling, especially in mountainous regions like the Greater Alpine Region (GAR), where observational networks are sparse and traditional models often underperform. This project presents **ML-IMEX**, a novel machine learning-based reconstruction of daily precipitation at 0.1° resolution for the pre-satellite era (1960–2000), leveraging ERA5 atmospheric reanalysis data and IMERG satellite observations.

---

## Objectives

- Downscale and bias-correct ERA5 precipitation fields using IMERG as a reference.
- Reconstruct IMERG-like daily precipitation for 1960–2000 using machine learning.
- Identify key atmospheric predictors for precipitation over the GAR.
- Validate the resulting ML-IMEX dataset using independent observational data (HISTALP, CHAPTER).

---

## Data Sources

- **ERA5**: Atmospheric reanalysis dataset (ECMWF) providing over 100 variables at 30 km resolution from 1940–present.
- **IMERG Final V07**: Satellite-based precipitation estimates at 0.1° resolution from 2001 onward.
- **ETOPO**: Topography data regridded to 0.1° for orographic predictors.
- **HISTALP**: Historical monthly precipitation records from 50 gauge stations across the GAR.
- **CHAPTER**: High-resolution (3x3 km) downscaled climate simulations used for additional validation.

---

## Methodology

### Preprocessing

- ERA5 was regridded to match IMERG’s spatial resolution (0.1°).
- February 29th was removed for temporal consistency across years.
- Days with IMERG Quality Index (QI) below 0.6 were excluded.
- Conservative remapping was applied to preserve physical consistency in precipitation fields.

### Machine Learning Model

- **Model**: Extreme Gradient Boosting (XGBoost)
- **Input Features**: 28 ERA5-derived predictors including thermodynamics, wind, moisture, and orography.
- **Feature Selection**: SHAP (SHapley Additive exPlanations) was used to identify the three most impactful variables:
  - Total Column Ice Water (TCIW)
  - ERA5 Precipitation
  - Column-Averaged Mixing Ratio (CAMR)
- **Training Setup**:
  - 60% training / 20% validation / 20% test
  - Daily granularity with spatial independence
  - Trained on all seasons (not season-specific models)

---

## Performance

| Metric        | ERA5 vs IMERG | XGB vs IMERG |
|---------------|---------------|---------------|
| **RMSE**      | 5.13 mm/day   | 4.39 mm/day   |
| **R²**        | 0.75          | 0.90          |

- XGB model reduces RMSE by ~14% compared to ERA5.
- Performs better across most terrain elevations and precipitation intensities.
- Shows robust generalization and spatial consistency.

---

## Validation

- **HISTALP**: XGB-generated data achieved **R² = 0.87**, outperforming both ERA5 (0.43) and CHAPTER (0.52).
- Demonstrated high agreement in monthly climatology over historical periods (1960–2000).

---

## Applications

- Climate change studies in data-sparse regions.
- Hydrological modeling and water resource planning.
- Historical weather event analysis in complex terrains.

---

## Repository Structure

```
├── ML-IMEX/
│   ├── data/              # ERA5, IMERG, and derived datasets
│   ├── code/              # Jupyter Notebooks for modeling
│   ├── figure/            # Core ML and preprocessing scripts
│   └── README.md          # This file
```

---

## Citation

If you use this dataset or code, please cite:

Goudarzi, I., Fazzini, D., Pasquero, C., Meroni, A. N., Borgnino, M. (2025).  
**A Machine Learning-Based Backward Extension of IMERG Daily Precipitation over the Greater Alpine Region**.  
[DOI:10.5281/zenodo.16631516](https://doi.org/10.5281/zenodo.16631516)

---

## License

This project is open source under the MIT License.

---

## Acknowledgments

This work was supported by the Department of Earth and Environmental Sciences, University of Milano-Bicocca.

