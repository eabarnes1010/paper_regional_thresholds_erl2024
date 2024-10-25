"""Functions for working with generic files.

Functions
---------
get_model_name(settings)
get_netcdf_da(filename)
save_pred_obs(pred_vector, filename)
save_tf_model(model, model_name, directory, settings)
get_cmip_filenames(settings, verbose=0)
"""

import xarray as xr
import json
import pickle
import tensorflow as tf
import custom_metrics

__author__ = "Elizabeth A. Barnes and Noah Diffenbaugh"
__version__ = "07 September 2023"


def get_model_name(settings):
    model_name = settings["exp_name"] + "_rng_seed" + str(settings["rng_seed"])

    return model_name


def get_netcdf_da(filename):
    da = xr.open_dataarray(filename)
    return da


def save_predictions(pred_vector, filename):
    with open(filename + ".pickle", "wb") as f:
        pickle.dump(pred_vector, f)


def load_tf_model(model_name, directory):
    # loading a tf model
    model = tf.keras.models.load_model(
        directory + model_name + "_model",
        compile=False,
        custom_objects={
            "InterquartileCapture": custom_metrics.InterquartileCapture(),
            "SignTest": custom_metrics.SignTest(),
            "CustomMAE": custom_metrics.CustomMAE(),
        },
    )
    return model


def convert_to_cftime(da, orig_time):
    da = da.rename({orig_time: "time"})
    dates = xr.cftime_range(start="1850", periods=da.shape[0], freq="YS", calendar="noleap")
    da = da.assign_coords({"time": ("time", dates, {"units": "years since 1850-01-01"})})
    return da


def save_tf_model(model, model_name, directory, settings):
    # save the tf model
    tf.keras.models.save_model(model, directory + model_name + "_model", overwrite=True)

    # save the meta data
    with open(directory + model_name + "_metadata.json", "w") as json_file:
        json_file.write(json.dumps(settings))


def get_cmip_filenames(settings, verbose=0):
    if settings["ssp"] == "370" and settings["gcmsub"] == "ALL":
        filenames = (
            "tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_GISS-E2-1-G_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_IPSL-CM6A-LR_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_CESM2-LE2-smbb_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["ssp"] == "370" and settings["gcmsub"] == "NOHOT":
        # screening models with TCR > 2.2C
        # https://www.science.org/doi/full/10.1126/sciadv.aba1981
        # https://www.nature.com/articles/d41586-022-01192-2

        filenames = (
            # "tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            # "tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_GISS-E2-1-G_r1-10_ncecat_ann_mean_2pt5degree.nc",
            # "tas_Amon_historical_ssp370_IPSL-CM6A-LR_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_CESM2-LE2-smbb_r1-10_ncecat_ann_mean_2pt5degree.nc",  # CESM2 TCR = 2.0C
        )
    elif settings["ssp"] == "245" and settings["gcmsub"] == "ALL":
        filenames = (
            "tas_Amon_historical_ssp245_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp245_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp245_CNRM-ESM2-1_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp245_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp245_GISS-E2-1-G_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp245_IPSL-CM6A-LR_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["ssp"] == "370" and settings["gcmsub"] == "UNIFORM":
        filenames = (
            "tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            # 'tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
        )
    elif (settings["ssp"] == "126" and settings["gcmsub"] == "ALL") and (settings["target_temp"] == 2.0):
        filenames = (
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )

    elif settings["ssp"] == "126" and settings["gcmsub"] == "ALL7":
        filenames = (
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["ssp"] == "126" and settings["gcmsub"] == "ALL10":
        filenames = (
            "tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["ssp"] == "126" and settings["gcmsub"] == "noHIGH10":
        filenames = (
            "tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["ssp"] == "126" and settings["gcmsub"] == "noHIGH7":
        filenames = (
            "tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc",
            #  'tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            #  'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
            "tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc",
            #  'tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc',
        )
    elif settings["ssp"] == "126" and settings["gcmsub"] == "noHIGH5":
        filenames = (
            "tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc",
            #  'tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            "tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc",
            #  'tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc',
        )

    elif settings["ssp"] == "126" and settings["gcmsub"] == "ALL":
        filenames = (
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["ssp"] == "126" and settings["gcmsub"] == "noM6":
        filenames = (
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            # 'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["ssp"] == "126" and settings["gcmsub"] == "FORCE":
        filenames = (
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["ssp"] == "126" and settings["gcmsub"] == "EXTEND":
        filenames = (
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )

    elif settings["ssp"] == "126" and settings["gcmsub"] == "UNIFORM":
        filenames = (
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )
    elif settings["gcmsub"] == "OOS":
        filenames = (
            "tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc",
        )

    # elif settings["ssp"] == '126' and settings["gcmsub"] == 'MIROC':
    #     filenames = (
    #                  'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
    #                  'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
    #                 )
    # elif settings["ssp"] == '370' and settings["gcmsub"] == 'MIROC':
    #     filenames = (
    #                  # 'tas_Amon_historical_ssp370_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
    #                  'tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
    #                 )
    elif settings["ssp"] == "126" and settings["gcmsub"] == "MAX":
        filenames = (
            "tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc",
            "tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc",
        )

    else:
        raise NotImplementedError("no such SSP")

    if verbose != 0:
        print(filenames)

    return filenames
