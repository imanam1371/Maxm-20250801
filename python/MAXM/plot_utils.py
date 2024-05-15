import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from rasterio.plot import show
from MAXM.utils import transform_dataset


def plot_map(xlong, xlat, map2d, transform, label="", title="", namefig="", draw_grid=False, projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(14, 4))
    extent = [xlong.min(), xlong.max(), xlat.min(), xlat.max()]
    ax.set_extent(extent)

    if transform is None:
        c = ax.pcolormesh(xlong, xlat, map2d, cmap='viridis')
        #c = ax.imshow(map2d, cmap='viridis')
        #image = show(map2d, ax=ax, cmap='viridis)
    elif isinstance(transform, rasterio.Affine):
        c = ax.imshow(map2d, cmap='viridis', extent=extent)
        image = show(map2d, transform=transform, ax=ax, cmap='viridis', extent=extent)
    else:
        c = ax.pcolormesh(xlong, xlat, map2d, cmap='viridis', transform=transform)

    ax.coastlines()
    cbar = fig.colorbar(c, ax=ax, label=label)#, shrink=1.0)

    if title:
        ax.set_title(title)

    if draw_grid:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        gl = ax.gridlines(crs=transform, linewidth=2, color='white', alpha=0.3, linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right=False
        gl.xlines = True
        """
        import matplotlib.ticker as mticker
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        gl.xlocator = mticker.FixedLocator([120, 140, 160, 180, -160, -140, -120])
        gl.ylocator = mticker.FixedLocator([0, 20, 40, 60])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
        """

    if namefig:
        fig.savefig(namefig, bbox_inches='tight')

    plt.close()


def draw_map_from_sat(dataset, transform, time=0, title="", figname="", label="", scale = 1., use_log=False, xlongs=[], xlats=[]):
    """
    Shape of the raster data: (1, 480, 1440)
    Metadata of the raster dataset: {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -99.0, 'width': 1440, 
    'height': 480, 'count': 1, 'crs': None, 'transform': Affine(0.25, 0.0, -180.0, 0.0, -0.25, 60.0)}
    """
    # Read the raster data as a numpy array
    raster_data = dataset.read()

    # Get information about the raster data
    #print("Shape of the raster data:", raster_data.shape)
    #print("Metadata of the raster dataset:", dataset.meta)
    #real_world_coords = transform * (i, j)

    xmin, ymin = dataset.transform * (0, 0)  # Upper-left corner
    xmax, ymax = dataset.transform * (dataset.width, dataset.height)  # Lower-right corner

    # Create a meshgrid of the x and y coordinates
    x = np.linspace(xmin, xmax, dataset.width)
    y = np.linspace(ymax, ymin, dataset.height)

    if len(xlongs) and len(xlats): 
        # Determine row and column indices corresponding to the longitude and latitude bounds
        xlat_start = np.searchsorted(y, xlats[-1], side='right')
        xlat_end = np.searchsorted(y, xlats[0], side='right')+1
        xlong_start = np.searchsorted(x, xlongs[0])
        xlong_end = np.searchsorted(x, xlongs[-1])+1

        # Extract the subset of the raster data within the specified bounds
        raster_data = raster_data[:, -xlat_end:-xlat_start, xlong_start:xlong_end]
        # Create a meshgrid of the longitude and latitude values for the subset region
        xlong, xlat = np.meshgrid(x[xlong_start:xlong_end], y[xlat_start:xlat_end])
    else:
        xlong, xlat = np.meshgrid(x, y)
 
    map2d = transform_dataset(raster_data[time], scale, use_log, True)
    plot_map(xlong, xlat, map2d, dataset.transform, label, title, figname)


def draw_map_from_rea(dataset, transform, var, time = 0, title="", figname="", label="", scale=1., use_log=False, xlongs=[], xlats=[]):
    """
    dimensions(sizes): longitude(61), latitude(25), time(468)
    variables(dimensions): float32 longitude(longitude), float32 latitude(latitude), int32 time(time), 
                           int16 t2m(time, latitude, longitude), int16 z(time, latitude, longitude), 
                           int16 msl(time, latitude, longitude), int16 tp(time, latitude, longitude), int16 p84.162(time, latitude, longitude)
    """
    rea_map = dataset.variables[var][time, :, :]
    xlong = xlongs if len(xlongs) else dataset.variables["longitude"][:]
    xlat  = xlats if len(xlats) else dataset.variables["latitude"][:]
    
    map2d = transform_dataset(rea_map, scale, use_log, True)
    plot_map(xlong, xlat, map2d, transform, label, title, figname)