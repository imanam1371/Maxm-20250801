import rasterio, glob, os
import numpy as np
import xarray as xr
import rioxarray as rxr
import cartopy.crs as ccrs
#import netCDF4 as nc
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from rioxarray.merge import merge_datasets
from rasterio.merge import merge
#from rasterio.plot import show
from MAXM.plot_utils import plot_map, draw_map_from_rea, draw_map_from_sat
from MAXM.utils import (create_path, check_value, fix_unsafe_map_vals, flatten_map,
                        transform_dataset, extract_features, check_map_values)


def main(iyear, fyear, draw_variables, run_training):
    transform = ccrs.PlateCarree() 
    era5_path = "../data/ERA5_GAR/monthly"
    era5_data1 = xr.open_dataset(f"{era5_path}/var_singlelevel_{iyear}-{fyear}.nc")
    era5_data2 = xr.open_dataset(f"{era5_path}/var_500hPa_{iyear}-{fyear}.nc")
    era5_data = xr.merge([era5_data1, era5_data2], compat='override')

    era5_long = era5_data.variables["longitude"][:]
    era5_lat  = era5_data.variables["latitude"][:]
    era5_variables = [var for var in era5_data.variables.keys() if var not in ["time", "longitude", "latitude"]]

    persian_path = "../data/PERSIANN-CDR"
    list_all_files = glob.glob(f"{persian_path}/CDR_*.tif")
    list_files = [f for f in list_all_files for y in range(iyear, fyear) if str(y) in f]
    list_files.sort()

    sat_data = []
    for fp in list_files:
        if not os.path.isfile(fp):
            raise FileNotFoundError("File not found: {}".format(fp))
        else:
            data = rasterio.open(fp)
            sat_data.append(data)

    use_tp_log = True
    tp_units = "mm/day"
    tp_label = f"log(Total precipitation) [{tp_units}]" if use_tp_log else f"Total precipitation [{tp_units}]" 
    sat_tp_scale = 1./30
    era5_tp_scale = 1000

    output_pred = "./predictions/"

    if draw_variables:
        rea_path = "plots/ERA5-GAR"
        sat_path = "plots/PERSIANN-CDR"
        create_path([rea_path, sat_path])

        bug_maps_file = "./list_bugged_maps.txt"
        if os.path.exists(bug_maps_file):
            os.system(f"rm {bug_maps_file}")

        time = 0 
        with open(bug_maps_file, "w") as f:
            for y in range(iyear, fyear+1):
                for m in range(12):
                    suffix = f"{y}_{m}"

                    for var in era5_variables:
                        create_path(f"{rea_path}/{var}")
                        units = era5_data.variables[var].attrs["units"]
                        name = era5_data.variables[var].attrs["long_name"]
                        use_log = use_tp_log if var == "tp" else False
                        label =  tp_label if var == "tp" else f"{var} [{units}]"
                        figname = f"{rea_path}/{var}/eda_era5_{var}_{suffix}.png"
                        scale = era5_tp_scale if var == "tp" else 1

                        if check_map_values(era5_data):
                            draw_map_from_rea(dataset = era5_data, transform = transform, time = time, var = var, 
                                            label=label, title = "Data from ERA5-GAR", figname = figname, scale = scale, use_log = use_log)
                        else:
                            f.write(f"ERA5 {var} {y} {m} \n")

                    figname = f"{sat_path}/eda_persian_{suffix}.png"
                    if check_map_values(sat_data[time]):
                        draw_map_from_sat(dataset=sat_data[time], transform = transform, label = tp_label, xlongs=era5_long,
                                        xlats=era5_lat, title = "Data from Persian-CDR", figname = figname, scale = sat_tp_scale, use_log = use_tp_log)
                    else:
                        f.write(f"PERSIAN {var} {y} {m} \n")
                    time += 1
            f.close() 

    if run_training:
        height = int((era5_lat.max()-era5_lat.min())/0.25)+1
        width = int((era5_long.max()-era5_long.min())/0.25)+1
        channels = len(era5_variables)-1
        target_is_sat = False

        Ysat = []
        for data in sat_data:
            xmin, ymin = data.transform * (0, 0)  # Upper-left corner
            xmax, ymax = data.transform * (data.width, data.height)  # Lower-right corner

            # Create a meshgrid of the x and y coordinates
            x = np.linspace(xmin, xmax, data.width)
            y = np.linspace(ymax, ymin, data.height)

            # Determine row and column indices corresponding to the longitude and latitude bounds
            xlat_start = np.searchsorted(y, era5_lat[-1], side='right')
            xlat_end = np.searchsorted(y, era5_lat[0], side='right') + 1
            xlong_start = np.searchsorted(x, era5_long[0])
            xlong_end = np.searchsorted(x, era5_long[-1]) + 1

            data = data.read()
            # Extract the subset of the raster data within the specified bounds
            subdata = data[:, -xlat_end:-xlat_start, xlong_start:xlong_end]
            Ysat = subdata if not len(Ysat) else np.append(Ysat, subdata, axis=0)


        Ysat = transform_dataset(Ysat, sat_tp_scale, use_tp_log, True)

        Yera5 = era5_data["tp"].values
        Yera5 = transform_dataset(Yera5, era5_tp_scale, use_tp_log, True)

        Y = Ysat if target_is_sat else Yera5
        X = np.stack([era5_data[var].values for var in era5_variables if var != "tp"], axis=-1)

        # Load and preprocess data (assuming 'X' contains input maps and 'y' contains target variables)
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, range(len(X)), test_size=0.2, random_state=42)

        X_train_flat = flatten_map(X_train, last_col=-1)
        X_test_flat = flatten_map(X_test, last_col=-1)
        y_train_flat = flatten_map(y_train)
        y_test_flat = flatten_map(y_test)

        # Define RandomForest regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_flat, y_train_flat)

        yrf_test_flat = rf.predict(X_test_flat)
        yrf_test = np.reshape(yrf_test_flat, y_test.shape) 
        mse = mean_squared_error(y_test_flat, yrf_test_flat)
        print(f"Random Forest Mean Squared Error: {mse}")

        # Define XGBoost regressor
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        xgb.fit(X_train_flat, y_train_flat)
        yxgb_test_flat = xgb.predict(X_test_flat)
        yxgb_test = np.reshape(yxgb_test_flat, y_test.shape) 
        mse = mean_squared_error(y_test_flat, yxgb_test_flat)
        print(f"XGBoost Mean Squared Error: {mse}")
        
        # Define MLP regressor
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)
        mlp.fit(X_train_flat, y_train_flat)
        ymlp_test_flat = mlp.predict(X_test_flat)
        ymlp_test = np.reshape(ymlp_test_flat, y_test.shape) 
        mse = mean_squared_error(y_test_flat, ymlp_test_flat)
        print(f"MLP Mean Squared Error: {mse}")

        # Define CNN architecture
        activation = "relu"
        loss = "mse"
        metrics = ["accuracy"]
        epochs = 50
        batch_size = 32
        loss_weights = [0.9, 0.1]
        cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (1, 1), activation=activation, input_shape=(height, width, channels)),
            #tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation=activation), 
            #tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            #tf.keras.layers.Conv2DTranspose(64, (1, 1), activation=activation, strides=(2, 2), padding='same'),
            tf.keras.layers.Dense(64, activation=activation),
            tf.keras.layers.Dense(1, activation="linear")  # Output layer for regression
        ])        
        optimizer = tf.keras.optimizers.Adam(3e-3, clipnorm=1.)
        cnn.compile(optimizer=optimizer, loss=loss, metrics=metrics)#, loss_weights=[0.9, 0.3])
        cnn.summary()

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-5)
        history = cnn.fit(X_train, y_train, epochs=epochs, callbacks=[reduce_lr], batch_size=batch_size, validation_data=(X_test, y_test))

        ycnn_test = cnn.predict(X_test)
        ycnn_test_vals = flatten_map(ycnn_test, first_col=-2)

        fig, ax = plt.subplots(figsize=(14, 4))
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        fig.savefig("accuracy.png", bbox_inches='tight')
        plt.close()

        test_loss, test_acc = cnn.evaluate(X_test, y_test, verbose=2)

        output_era5 = f"{output_pred}/ERA5"
        output_rf = f"{output_pred}/RF"
        output_xgb = f"{output_pred}/XGB"
        output_mlp = f"{output_pred}/MLP"
        output_cnn = f"{output_pred}/CNN"
        for path in [output_era5, output_rf, output_xgb, output_mlp, output_cnn]:
            create_path(path)
        for i in range(len(y_test)):
            #plot_map(era5_long, era5_lat, y_test[i], None, tp_label, "ERA5 data", f"{output_era5}/era5_data_{i}.png")
            #plot_map(era5_long, era5_lat, yrf_test[i], None, tp_label, "Random Forest prediction", f"{output_rf}/rf_prediction_{i}.png")
            #plot_map(era5_long, era5_lat, yrf_test[i], None, tp_label, "XGBoost prediction", f"{output_xgb}/xgb_prediction_{i}.png")
            plot_map(era5_long, era5_lat, ymlp_test_vals[i], None, tp_label, "MLP prediction", f"{output_mlp}/mlp_prediction_{i}.png")
            plot_map(era5_long, era5_lat, ycnn_test_vals[i], None, tp_label, "CNN prediction", f"{output_cnn}/cnn_prediction_{i}.png")



if __name__ == "__main__":
    iyear=1983 
    fyear=2021 
    draw_variables = False
    run_training = True

    main(iyear, fyear, draw_variables, run_training)