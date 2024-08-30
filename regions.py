"""Region definitions.

Functions
---------

"""

import xarray as xr
import numpy as np
import pandas as pd
import regionmask
import matplotlib.pyplot as plt

shape_directory = "shapefiles/"


# GET SHAPEFILE COUNTRIES
# [m1] conda install -c conda-forge geopandas pygeos regionmask
# import geopandas as gpd
# import regionmask
#
# SHAPE_DIRECTORY = 'shapefiles/'
# regs_shp = gpd.read_file(SHAPE_DIRECTORY + 'ne_10m_admin_0_countries.shp')
#
# mask_country = regionmask.mask_geopandas(regs_shp, np.arange(0,360,.1), np.arange(-90,90,.1))
# mask_country.to_netcdf(SHAPE_DIRECTORY + 'countries_10m_tenthDegreeGrid.nc')
#
# plt.imshow(mask_country,cmap="Accent")
# plt.colorbar()
# plt.show()
# print(np.unique(mask_country),len(np.unique(mask_country)))
#
# mask_country = regionmask.mask_geopandas(regs_shp, da_pop_regrid.lon, da_pop_regrid.lat)
# mask_country.to_netcdf(SHAPE_DIRECTORY + 'countries_10m_2.5x2.5.nc')
#
# plt.imshow(mask_country,cmap="Accent")
# plt.colorbar()
# plt.show()
# print(np.unique(mask_country),len(np.unique(mask_country)))


def extract_region(settings, data, lat=None, lon=None):

    if settings["target_region"] is None:
        return data, None, None

    elif settings["target_region"][:4] == "reg_":

        regions_range_dict = get_region_dict(settings["target_region"])

        min_lon, max_lon = regions_range_dict["lon_range"]
        min_lat, max_lat = regions_range_dict["lat_range"]

        mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
        mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
        return data.where(mask_lon & mask_lat, drop=True), None, None

    elif settings["target_region"][:5] == "ipcc_":
        ar6_land = regionmask.defined_regions.ar6.land
        mask = regionmask.defined_regions.ar6.land.mask(data)

        i = ar6_land.abbrevs.index(settings["target_region"][5:])
        mask_subset = np.where(mask == i, 1., np.nan)

        if settings["land_only"]:
            country_mask = xr.load_dataarray(shape_directory + 'countries_10m_2.5x2.5.nc')
            mask_subset = np.where(~np.isnan(country_mask), mask_subset, np.nan)

        # print(data.shape)
        # print(ar6_land.abbrevs[i])
        # plt.pcolor(data[0, :, :] * mask_subset)
        # plt.show()

        return data * mask_subset, None, None

    else:
        regs_shp = pd.read_csv(shape_directory + 'ne_10m_admin_0_countries_CSV.csv')
        country_mask = xr.load_dataarray(shape_directory + 'countries_10m_2.5x2.5.nc')

        region_indices = regs_shp[regs_shp["ADM0_A3"].isin(settings["target_region"])].index.values
        country_mask_subset = country_mask.where(country_mask.isin(region_indices), np.nan) * 0.0 + 1.

        return data * country_mask_subset, None, None


def get_region_dict(region=None):
    regions_dict = {

        "reg_nh": {
            "lat_range": [0, 90],
            "lon_range": [0, 360],
        },

        "reg_sh": {
            "lat_range": [-90, 0],
            "lon_range": [0, 360],
        },

        "reg_north_atlantic": {
            "lat_range": [40, 60],
            "lon_range": [360 - 70, 360 - 10],
        },

        "reg_eastern_europe": {
            "lat_range": [40, 60],
            "lon_range": [0, 30],
        },

        "reg_western_us": {
            "lat_range": [30, 49],
            "lon_range": [360 - 125, 360 - 110],
        },

    }

    if region is None:
        return regions_dict
    else:
        return regions_dict[region]
