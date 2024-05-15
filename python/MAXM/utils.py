import os
import numpy as np
import xarray as xr
from math import radians, sin, cos, sqrt, atan2

def create_path(inpaths):
    if not isinstance(inpaths, list):
        inpaths = [inpaths]

    for path in inpaths:
        if not os.path.exists(path):
            print(f"Creating dir {path}")
            is_global = str(path).startswith("/")
            list_subdirs = path.split("/")
            tmp_path = "" if not is_global else "/"
            for subdir in list_subdirs:
                tmp_path += f"{subdir}/"
                if not subdir or subdir in [".", ".."]:
                    continue
                elif not os.path.exists(tmp_path):
                    os.system(f"mkdir {tmp_path}")


def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the Earth in kilometers (mean value)
    radius = 6371.0

    # Calculate the distance
    distance = radius * c

    return distance


def check_value(val, require_positive = False, no_zero = False):
    flag = True
    if np.isinf(val) or np.isnan(val):
        flag = False
    if require_positive and val < 0:
        flag = False
    if no_zero and val == 0:
        flag = False

    return flag


def fix_unsafe_map_vals(map2d, window_size = 1, default_value = 0., require_positive = False, no_zero = False):

    dims = list(map2d.shape)

    if len(dims) == 2:
        add_new_dim = True
        dims = [1] + dims
        map2d_copy = [map2d]
    else:
        add_new_dim = False
        map2d_copy = map2d


    for i in range(dims[0]):
        for x in range(dims[1]):
            for y in range(dims[2]):
                if not check_value(map2d_copy[i][x][y], require_positive, no_zero):
                    val = 0.
                    nvals = 0
                    for ix in range(x-window_size, x+window_size+1):
                        for iy in range(y-window_size, y+window_size+1):
                            if ix < 0 or ix >= dims[1] or iy < 0 or iy >= dims[2]:
                                continue
                            if check_value(map2d_copy[i][ix][iy], require_positive, no_zero):
                                val += map2d_copy[i][ix][iy]
                                nvals += 1
                    map2d_copy[i][x][y] = val/nvals if nvals else default_value

    return map2d_copy[0] if add_new_dim else map2d_copy


def transform_dataset(map2d, scale=1., use_log=False, fix_values=False, window_size=1):
    if fix_values:
        map2d = fix_unsafe_map_vals(map2d, window_size)

    if scale != 1:
        map2d = map2d * scale

    if use_log:
        # Replace negative or zero values
        map2d = fix_unsafe_map_vals(map2d, window_size, 1, True, True)
        map2d = np.log(map2d)

    return map2d


def extract_features(ds, variables):
    features = []
    
    # Iterate over variables and compute features
    for var in variables:
        # Extract variable data
        var_data = ds[var].values
        
        # Compute mean and standard deviation along spatial dimensions
        mean_feature = np.mean(var_data, axis=(1, 2))  # Compute mean along latitude and longitude
        std_feature = np.std(var_data, axis=(1, 2))    # Compute standard deviation along latitude and longitude
        
        # Concatenate features
        var_features = np.column_stack((mean_feature, std_feature))
        
        # Append variable features to list
        features.append(var_features)
    
    # Combine features from all variables
    all_features = np.concatenate(features, axis=1)
    
    return all_features


def check_map_values(map2d):
    if isinstance(map2d, list) or isinstance(map2d, np.ndarray):
        map_array = map2d
    elif isinstance(map2d, xr.Dataset):
        map_array = map2d.to_array().values
    else:
        map_array = map2d.read()
  
    has_nan = np.isnan(map_array).any()
    has_inf = np.isinf(map_array).any()
    unique_values = np.unique(map_array)

    if has_nan or has_inf or len(unique_values) == 1:
        return False
    else:
        return True


def flatten_map(map2d, first_col=0, last_col=None):
    dims = map2d.shape
    last_col = last_col if last_col else len(dims)
    num_samples = np.prod(dims[first_col:last_col])
    new_dims = list(dims[0:first_col]) + [num_samples] + list(dims[last_col:len(dims)])
    map2d_flat = map2d.reshape(new_dims)
    return map2d_flat