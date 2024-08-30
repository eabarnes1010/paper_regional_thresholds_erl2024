"""Build the split and scaled training, validation and testing data.

Functions
---------
get_members(settings)
get_observations(directory, settings)
get_cmip_data(directory, rng, settings)
get_labels(da, settings, plot=False)
preprocess_data(da, members, settings)
make_data_split(da, data, f_labels, f_years, labels, years, members, settings)
"""
import numpy as np
import pandas as pd
import regions
import file_methods
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import tensorflow as tf
import xarray as xr
from scipy.optimize import curve_fit
import custom_metrics
import plots

__author__ = "Elizabeth A. Barnes and Noah Diffenbaugh"
__version__ = "18 December 2023"

FIGURE_DIRECTORY = "figures/"


def get_gcm_model_names(directory):
    import glob

    modelnames = []
    filenames = glob.glob(directory + "/*")
    for f in filenames:
        k = f.find("historical_")
        m = f.find("_ann_mean")
        if k >= 0:
            modelnames.append(f[k + len("historical_") : m])
    return modelnames


def get_members(settings):
    n_train = settings["n_train_val_test"][0]
    n_val = settings["n_train_val_test"][1]
    n_test = settings["n_train_val_test"][2]
    all_members = np.arange(0, n_train + n_val + n_test)

    return n_train, n_val, n_test, all_members


def get_hot_models():
    hot_models = (
        "CNRM-CM6-1-HR",
        "CanESM5",
        "E3SM-1-0",
        "EC-Earth3-Veg",
        "HadGEM3-GC31-LL",
        "HadGEM3-GC31-MM",
        "IPSL-CM6A-LR",
        "NESM3",
        "SAM0-UNICON",
        "UKESM1-0-LL",
    )

    return hot_models


def get_one_model_one_vote_data(directory, settings):
    data_all = None
    cmpi6_model_names = get_gcm_model_names(directory)
    cmpi6_model_names = np.unique(cmpi6_model_names)
    print(cmpi6_model_names)

    hot_models = get_hot_models()

    for model_name in cmpi6_model_names:
        if settings["gcmsub"] == "NOHOT":
            if model_name[: model_name.find("_")] in hot_models:
                print("skipping hot model: " + model_name)
                continue

        # combine historical and ssp
        nc_filename = "tas_Amon_historical_" + model_name + "_ann_mean_2pt5degree.nc"
        da_hist = file_methods.get_netcdf_da(directory + nc_filename)
        nc_filename = "tas_Amon_ssp370_" + model_name + "_ann_mean_2pt5degree.nc"
        da_ssp = file_methods.get_netcdf_da(directory + nc_filename)
        da = xr.concat([da_hist, da_ssp], dim="year")

        da = file_methods.convert_to_cftime(da, orig_time="year")

        x_gcm, __, __ = regions.extract_region(settings, da)
        x_gcm = preprocess_data(x_gcm, members=None, settings=settings)

        if data_all is None:
            data_all = x_gcm
        else:
            data_all = xr.concat([data_all, x_gcm], dim="gcm")

    return data_all


def get_observations(directory, settings, nc_filename_obs=None, verbose=True):
    if nc_filename_obs is None:
        if settings["obsdata"] == "BEST":
            if settings["final_year_of_obs"] == 2021:
                nc_filename_obs = "_Land_and_Ocean_LatLong1_185001_202112_ann_mean_2pt5degree.nc"
            elif settings["final_year_of_obs"] == 2022:
                nc_filename_obs = "_Land_and_Ocean_LatLong1_185001_202212_ann_mean_2pt5degree.nc"
            elif settings["final_year_of_obs"] == 2023:
                nc_filename_obs = "_Land_and_Ocean_LatLong1_185001_202312_ann_mean_2pt5degree.nc"
        else:
            raise NotImplementedError("no such obs data")

    da_obs = file_methods.get_netcdf_da(directory + nc_filename_obs)
    global_mean_obs, __, __ = regions.extract_region(settings, da_obs)
    global_mean_obs = compute_global_mean(global_mean_obs)

    data_obs = preprocess_data(da_obs, members=None, settings=settings)
    x_obs = data_obs.values.reshape(
        (data_obs.shape[0], data_obs.shape[1], data_obs.shape[2])
    )

    if settings["obs_training_only"]:
        iy = np.where(data_obs["time.year"] >= settings["training_yr_bounds"][0])[0]
        data_obs = data_obs[iy, :, :]
        x_obs = x_obs[iy, :]
        global_mean_obs = global_mean_obs[iy]

    if settings["cumulative_history"]:
        if settings["cumulative_sum"]:
            n_size = np.ones((data_obs.shape[0]))
        else:
            n_size = np.arange(0, data_obs.shape[0]) + 1
        d_obs = np.cumsum(data_obs.values, axis=0) / n_size[:, np.newaxis, np.newaxis]
        d_obs = d_obs.reshape((d_obs.shape[0], d_obs.shape[1], d_obs.shape[2]))
        if settings["cumulative_history_only"]:
            # add channel dimension
            x_obs = d_obs[:, :, :, np.newaxis]
        else:
            # concatenate channels across a new channel dimension
            x_obs = np.stack((x_obs, d_obs), axis=-1)
    else:
        # add channel dimension
        x_obs = x_obs[:, :, :, np.newaxis]

    if settings["anomalies"]:
        if verbose:
            print("observations: filling NaNs with zeros")
        x_obs = np.nan_to_num(x_obs, 0.0)

    if verbose:
        print("np.shape(x_obs) = " + str(np.shape(x_obs)))
        print("np.shape(data_obs) = " + str(np.shape(data_obs)))

    return data_obs, x_obs, global_mean_obs


def compute_global_mean(da):
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    temp_weighted = da.weighted(weights)
    global_mean = temp_weighted.mean(("lon", "lat"), skipna=True)

    return global_mean


def create_data(directory, settings, verbose=1):
    x_train = None
    for t_treshold in settings["target_temps"]:
        settings["target_temp"] = t_treshold
        (
            t_x_train,
            t_x_val,
            t_x_test,
            t_y_train,
            t_y_val,
            t_y_test,
            t_onehot_train,
            t_onehot_val,
            t_onehot_test,
            t_y_yrs_train,
            t_y_yrs_val,
            t_y_yrs_test,
            t_target_years,
            map_shape,
            settings,
        ) = get_cmip_data(directory, settings, verbose)

        if np.isnan(np.sum(t_target_years)):
            print("...throwing out this threshold due to models not reaching it...")
            continue

        if x_train is None:
            x_train = t_x_train
            x_val = t_x_val
            x_test = t_x_test
            y_train = t_y_train
            y_val = t_y_val
            y_test = t_y_test
            onehot_train = t_onehot_train
            onehot_val = t_onehot_val
            onehot_test = t_onehot_test
            y_yrs_train = t_y_yrs_train
            y_yrs_val = t_y_yrs_val
            y_yrs_test = t_y_yrs_test
            target_temp_train = np.ones(t_y_train.shape[0]) * t_treshold
            target_temp_val = np.ones(t_y_val.shape[0]) * t_treshold
            target_temp_test = np.ones(t_y_test.shape[0]) * t_treshold
            target_years = t_target_years
        else:
            x_train = np.concatenate((x_train, t_x_train), axis=0)
            x_val = np.concatenate((x_val, t_x_val), axis=0)
            x_test = np.concatenate((x_test, t_x_test), axis=0)
            y_train = np.concatenate((y_train, t_y_train), axis=0)
            y_val = np.concatenate((y_val, t_y_val), axis=0)
            y_test = np.concatenate((y_test, t_y_test), axis=0)
            onehot_train = np.concatenate((onehot_train, t_onehot_train), axis=0)
            onehot_val = np.concatenate((onehot_val, t_onehot_val), axis=0)
            onehot_test = np.concatenate((onehot_test, t_onehot_test), axis=0)
            y_yrs_train = np.concatenate((y_yrs_train, t_y_yrs_train), axis=0)
            y_yrs_val = np.concatenate((y_yrs_val, t_y_yrs_val), axis=0)
            y_yrs_test = np.concatenate((y_yrs_test, t_y_yrs_test), axis=0)
            target_temp_train = np.concatenate(
                (target_temp_train, np.ones(t_y_train.shape[0]) * t_treshold), axis=0
            )
            target_temp_val = np.concatenate(
                (target_temp_val, np.ones(t_y_val.shape[0]) * t_treshold), axis=0
            )
            target_temp_test = np.concatenate(
                (target_temp_test, np.ones(t_y_test.shape[0]) * t_treshold), axis=0
            )
            target_years = np.concatenate((target_years, t_target_years), axis=0)

    settings["target_temp"] = None
    if verbose == 1:
        print("...")
        print(
            f"{x_train.shape=}, {target_temp_train.shape=}, {y_train.shape=}, {y_yrs_train.shape=}"
        )
        print(
            f"{x_val.shape=}, {target_temp_val.shape=}, {y_val.shape=}, {y_yrs_val.shape=}"
        )
        print(
            f"{x_test.shape=}, {target_temp_test.shape=}, {y_test.shape=}, {y_yrs_test.shape=}"
        )
        print("\n")

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        onehot_train,
        onehot_val,
        onehot_test,
        y_yrs_train,
        y_yrs_val,
        y_yrs_test,
        target_temp_train,
        target_temp_val,
        target_temp_test,
        target_years,
        map_shape,
        settings,
    )


def get_cmip_data(directory, settings, verbose=1):
    data_train, data_val, data_test = None, None, None
    labels_train, labels_val, labels_test = None, None, None
    years_train, years_val, years_test = None, None, None
    target_years = []

    N_TRAIN, N_VAL, N_TEST, ALL_members = get_members(settings)

    rng_cmip = np.random.default_rng(settings["rng_seed"])
    train_members = rng_cmip.choice(ALL_members, size=N_TRAIN, replace=False)
    val_members = rng_cmip.choice(
        np.setdiff1d(ALL_members, train_members), size=N_VAL, replace=False
    )
    test_members = rng_cmip.choice(
        np.setdiff1d(ALL_members, np.append(train_members[:], val_members)),
        size=N_TEST,
        replace=False,
    )
    if verbose == 1:
        print(train_members, val_members, test_members)

    # save the meta data
    settings["train_members"] = train_members.tolist()
    settings["val_members"] = val_members.tolist()
    settings["test_members"] = test_members.tolist()

    # loop through and get the data
    filenames = file_methods.get_cmip_filenames(settings, verbose=0)
    for f in filenames:
        if verbose == 1:
            print(f)
        da = file_methods.get_netcdf_da(directory + f)
        f_labels, f_years, __, f_target_year = get_labels(
            da, settings, verbose=verbose, plot=False
        )
        if np.isnan(f_target_year):
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                [np.nan, np.nan],
                np.nan,
                settings,
            )

        # create sets of train / validaton / test
        target_years = np.append(target_years, f_target_year)
        data_train, labels_train, years_train = make_data_split(
            da,
            data_train,
            f_labels,
            f_years,
            labels_train,
            years_train,
            train_members,
            settings,
        )
        data_val, labels_val, years_val = make_data_split(
            da,
            data_val,
            f_labels,
            f_years,
            labels_val,
            years_val,
            val_members,
            settings,
        )
        data_test, labels_test, years_test = make_data_split(
            da,
            data_test,
            f_labels,
            f_years,
            labels_test,
            years_test,
            test_members,
            settings,
        )

    if verbose == 1:
        print("---------------------------")
        print("train.shape = " + str(np.shape(data_train)))
        print("val.shape = " + str(np.shape(data_val)))
        print("test.shape = " + str(np.shape(data_test)))

    # Reshape input data, although this data may be replaced by the cumulative history in the next code block
    x_train = data_train.reshape(
        (
            data_train.shape[0] * data_train.shape[1],
            data_train.shape[2],
            data_train.shape[3],
        )
    )
    x_val = data_val.reshape(
        (data_val.shape[0] * data_val.shape[1], data_val.shape[2], data_val.shape[3])
    )
    x_test = data_test.reshape(
        (
            data_test.shape[0] * data_test.shape[1],
            data_test.shape[2],
            data_test.shape[3],
        )
    )

    # Add cumulative history or not, and make sure matrices align
    if settings["cumulative_history"]:
        if settings["cumulative_sum"]:
            n_size = np.ones((data_train.shape[1]))
        else:
            n_size = np.arange(0, data_train.shape[1]) + 1

        d_train = (
            np.cumsum(data_train, axis=1)
            / n_size[np.newaxis, :, np.newaxis, np.newaxis]
        )
        d_train = d_train.reshape(
            (d_train.shape[0] * d_train.shape[1], d_train.shape[2], d_train.shape[3])
        )

        d_val = (
            np.cumsum(data_val, axis=1) / n_size[np.newaxis, :, np.newaxis, np.newaxis]
        )
        d_val = d_val.reshape(
            (d_val.shape[0] * d_val.shape[1], d_val.shape[2], d_val.shape[3])
        )

        d_test = (
            np.cumsum(data_test, axis=1) / n_size[np.newaxis, :, np.newaxis, np.newaxis]
        )
        d_test = d_test.reshape(
            (d_test.shape[0] * d_test.shape[1], d_test.shape[2], d_test.shape[3])
        )

        if settings["cumulative_history_only"]:
            # add channel dimension
            x_train = d_train[:, :, :, np.newaxis]
            x_val = d_val[:, :, :, np.newaxis]
            x_test = d_test[:, :, :, np.newaxis]
        else:
            # concatentate channels across a new channel dimension
            x_train = np.stack((x_train, d_train), axis=-1)
            x_val = np.stack((x_val, d_val), axis=-1)
            x_test = np.stack((x_test, d_test), axis=-1)
    else:
        # add channel dimension
        x_train = x_train[:, :, :, np.newaxis]
        x_val = x_val[:, :, :, np.newaxis]
        x_test = x_test[:, :, :, np.newaxis]

    # Create the labels
    y_train = labels_train.reshape((data_train.shape[0] * data_train.shape[1],))
    y_val = labels_val.reshape((data_val.shape[0] * data_val.shape[1],))
    y_test = labels_test.reshape((data_test.shape[0] * data_test.shape[1],))

    y_yrs_train = years_train.reshape((data_train.shape[0] * data_train.shape[1],))
    y_yrs_val = years_val.reshape((data_val.shape[0] * data_val.shape[1],))
    y_yrs_test = years_test.reshape((data_test.shape[0] * data_test.shape[1],))

    # make onehot vectors for training
    if settings["network_type"] == "shash2":
        onehot_train = np.zeros((x_train.shape[0], 2))
        onehot_train[:, 0] = y_train.astype("float32")
        onehot_val = np.zeros((x_val.shape[0], 2))
        onehot_val[:, 0] = y_val.astype("float32")
        onehot_test = np.zeros((x_test.shape[0], 2))
        onehot_test[:, 0] = y_test.astype("float32")
    else:
        onehot_train = np.copy(y_train)
        onehot_val = np.copy(y_val)
        onehot_test = np.copy(y_test)

    map_shape = np.shape(data_train)[2:]

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        onehot_train,
        onehot_val,
        onehot_test,
        y_yrs_train,
        y_yrs_val,
        y_yrs_test,
        target_years,
        map_shape,
        settings,
    )


# def preprocess_data_year(da, members, settings):
#     if members is None:
#         new_data = da
#     else:
#         new_data = da[members, :, :, :]

#     if settings["anomalies"] is True:
#         new_data = new_data - new_data.sel(
#             year=slice(
#                 (settings["anomaly_yr_bounds"][0]), (settings["anomaly_yr_bounds"][1])
#             )
#         ).mean("year")
#     if settings["anomalies"] == "Baseline":
#         new_data = new_data - new_data.sel(
#             year=slice(
#                 (settings["baseline_yr_bounds"][0]), (settings["baseline_yr_bounds"][1])
#             )
#         ).mean("year")
#         new_data = new_data - new_data.sel(
#             year=slice(
#                 (settings["anomaly_yr_bounds"][0]), (settings["anomaly_yr_bounds"][1])
#             )
#         ).mean("year")

#     if settings["remove_map_mean"] == "raw":
#         new_data = new_data - new_data.mean(("lon", "lat"))
#     elif settings["remove_map_mean"] == "weighted":
#         weights = np.cos(np.deg2rad(new_data.lat))
#         weights.name = "weights"
#         new_data_weighted = new_data.weighted(weights)
#         new_data = new_data - new_data_weighted.mean(("lon", "lat"))

#     if settings["remove_sh"]:
#         # print('removing SH')
#         i = np.where(new_data["lat"] <= -50)[0]
#         if len(new_data.shape) == 3:
#             new_data[:, i, :] = 0.0
#         else:
#             new_data[:, :, i, :] = 0.0

#     if settings["remove_poles"]:
#         ilat = np.where(np.abs(new_data["lat"]) > 70.0)[0]
#         if len(new_data.shape) == 3:
#             new_data[:, ilat, :] = 0.0
#         else:
#             new_data[:, :, ilat, :] = 0.0

#     return new_data


def preprocess_data(da, members, settings):
    if members is None:
        new_data = da
    else:
        new_data = da[members, :, :, :]

    if settings["anomalies"] is True:
        new_data = new_data - new_data.sel(
            time=slice(
                str(settings["anomaly_yr_bounds"][0]),
                str(settings["anomaly_yr_bounds"][1]),
            )
        ).mean("time")
    if settings["anomalies"] == "Baseline":
        new_data = new_data - new_data.sel(
            time=slice(
                str(settings["baseline_yr_bounds"][0]),
                str(settings["baseline_yr_bounds"][1]),
            )
        ).mean("time")
        new_data = new_data - new_data.sel(
            time=slice(
                str(settings["anomaly_yr_bounds"][0]),
                str(settings["anomaly_yr_bounds"][1]),
            )
        ).mean("time")

    if settings["remove_map_mean"] == "raw":
        new_data = new_data - new_data.mean(("lon", "lat"))
    elif settings["remove_map_mean"] == "weighted":
        weights = np.cos(np.deg2rad(new_data.lat))
        weights.name = "weights"
        new_data_weighted = new_data.weighted(weights)
        new_data = new_data - new_data_weighted.mean(("lon", "lat"))

    if settings["remove_sh"]:
        # print('removing SH')
        i = np.where(new_data["lat"] <= -50)[0]
        if len(new_data.shape) == 3:
            new_data[:, i, :] = 0.0
        else:
            new_data[:, :, i, :] = 0.0

    if settings["remove_poles"]:
        ilat = np.where(np.abs(new_data["lat"]) > 66.0)[0]
        if len(new_data.shape) == 3:
            new_data[:, ilat, :] = 0.0
        else:
            new_data[:, :, ilat, :] = 0.0

    return new_data


def make_data_split(da, data, f_labels, f_years, labels, years, members, settings):
    # process the data, i.e. compute anomalies, subtract the mean, etc.
    new_data = preprocess_data(da, members, settings)

    # only train on certain samples
    iyears = np.where(
        (f_years >= settings["training_yr_bounds"][0])
        & (f_years <= settings["training_yr_bounds"][1])
    )[0]
    f_years = f_years[iyears]
    f_labels = f_labels[iyears]
    new_data = new_data[:, iyears, :, :]

    if data is None:
        data = new_data.values
        labels = np.tile(f_labels, (len(members), 1))
        years = np.tile(f_years, (len(members), 1))
    else:
        data = np.concatenate((data, new_data.values), axis=0)
        labels = np.concatenate((labels, np.tile(f_labels, (len(members), 1))), axis=0)
        years = np.concatenate((years, np.tile(f_years, (len(members), 1))), axis=0)

    return data, labels, years


def get_labels(
    da,
    settings,
    plot=False,
    verbose=1,
    compute_global_mean_bool=True,
    quad_interp=False,
):

    # plot = True
    # compute the ensemble mean, global mean temperature
    # these computations should be based on the training set only
    if compute_global_mean_bool:
        data_output, __, __ = regions.extract_region(settings, da)
        global_mean = compute_global_mean(data_output.mean(axis=0))
        global_mean_ens = compute_global_mean(data_output)
    else:
        global_mean = da
        global_mean_ens = da

    # compute the target year
    baseline_mean = global_mean.sel(
        time=slice(
            str(settings["baseline_yr_bounds"][0]),
            str(settings["baseline_yr_bounds"][1]),
        )
    ).mean("time")

    if quad_interp:
        years = global_mean["time.year"].values
        iyrs = np.where(years >= settings["fit_start_year"])[0]

        fit, __ = curve_fit(
            custom_metrics.quadraticFunc, years[iyrs], global_mean.values[iyrs]
        )
        # y_values = custom_metrics.cubicFunc(years, *fit)
        y_values = custom_metrics.quadraticFunc(years[iyrs], *fit)
        # plt.plot(years, y_values)
        # plt.show()

    else:
        years = global_mean["time.year"].values
        iyrs = np.where(years >= settings["fit_start_year"])[0]
        y_values = global_mean.values[iyrs]

    if not settings["smooth"]:
        iwarmer = (y_values - baseline_mean.values) - settings["target_temp"]
    else:
        raise NotImplementedError()

    i_neg = np.where(iwarmer < 0)[0]
    if len(i_neg) < 1:
        target_year = np.nan

        # print(i_neg)
        # plt.figure()
        # plt.plot(years, global_mean.values)
        # plt.plot(years[iyrs], y_values, '-r')
        # plt.title(settings["exp_name"])
        # plt.show()
    else:
        i_threshold = i_neg[-1] + 1
        if i_threshold == np.shape(y_values)[0]:
            target_year = np.nan
        else:
            target_year = global_mean["time.year"].values[iyrs[i_threshold]]

    # plot the calculation to make sure things make sense
    if plot:
        plt.figure(figsize=(8, 4), dpi=300)

        plt.plot(
            global_mean["time.year"].values,
            global_mean.values - baseline_mean.values,
            linewidth=1.5,
            label="Berkeley Observations (5-yr running mean)",
            color="k",
        )

        try:
            for ens in np.arange(0, global_mean_ens.shape[0]):
                if quad_interp:
                    plt.plot(
                        # years[iyrs],
                        # y_values[iyrs] - baseline_mean.values,
                        years[iyrs],
                        y_values - baseline_mean.values,
                        "-",
                        color="deeppink",
                        linewidth=2.5,
                        label="Fit (1970-" + str(settings["final_year_of_obs"]) + ")",
                    )

                plt.plot(
                    global_mean_ens["time.year"].values,
                    global_mean_ens[ens, :].values - baseline_mean.values,
                    linewidth=1,
                    color="gray",
                    alpha=0.5,
                )
        except:
            pass

        for thresh in settings["transfer_temp_vec"]:
            plt.axhline(
                y=thresh,
                color="tab:gray",
                linewidth=1.0,
                linestyle="--",
            )
        plt.xlabel("year")
        plt.ylabel("temp anomaly (C)")
        plt.xlim(1850, 2025)
        plt.legend(fontsize=10)

    # define the labels
    if verbose == 1:
        print(
            "TARGET_YEAR = "
            + str(target_year)
            + ", TARGET_TEMP = "
            + str(settings["target_temp"])
        )
    labels = target_year - da["time.year"].values

    return (
        labels,
        global_mean["time.year"].values,
        global_mean.values - baseline_mean.values,
        target_year,
    )


def make_transfer_data(
    settings, transfer_temps, da_obs, x_obs, plot=False, quad_interp=False
):
    onehot_vec = None
    input_target_threshold = None
    x_obs_concat = None
    obs_yearsvals_dict = dict()

    for ttemp in transfer_temps:
        settings["target_temp"] = ttemp

        obs_labels, obs_years, obs_timeseries, obs_target_year = get_single_labels(
            da_obs,
            settings,
            plot=False,
            verbose=0,
            compute_global_mean_bool=False,
            quad_interp=True,
        )

        # check that treshold year is less than the max allowed
        if obs_target_year > settings["transfer_max_year"] or np.isnan(obs_target_year):
            continue
            # obs_target_year = np.nan
        obs_yearsvals_dict[ttemp] = obs_target_year
        print(f"{ttemp=}, {obs_target_year=}")

        # make one hot vector
        onehots = np.asarray(
            obs_target_year
            - np.arange(
                settings["training_yr_bounds"][0], settings["final_year_of_obs"] + 1
            ),
            dtype=float,
        )
        if onehot_vec is None:
            onehot_vec = onehots
            input_target_threshold = np.ones(onehots.shape) * settings["target_temp"]
            x_obs_concat = x_obs
        else:
            onehot_vec = np.concatenate((onehot_vec, onehots), axis=0)
            input_target_threshold = np.concatenate(
                (
                    input_target_threshold,
                    np.ones(onehots.shape) * settings["target_temp"],
                ),
                axis=0,
            )
            x_obs_concat = np.concatenate((x_obs_concat, x_obs), axis=0)

    if settings["network_type"] == "reg":
        onehot_vec = onehot_vec[:, np.newaxis]
    else:
        if onehot_vec is None:
            onehot_vec = np.ones((np.shape(x_obs)[0], 1)) * np.nan
            input_target_threshold = (
                np.ones(onehot_vec.shape) * settings["target_temp"] * np.nan
            )
            x_obs_concat = x_obs * np.nan
        else:
            onehot_vec = np.concatenate(
                (onehot_vec[:, np.newaxis], np.zeros((onehot_vec.shape[0], 1))), -1
            )

    # make schematic
    if plot:
        settings_plot = settings.copy()
        settings_plot["target_temp"] = 1.0
        obs_labels, obs_years, obs_timeseries, obs_target_year = get_single_labels(
            da_obs,
            settings,
            plot=True,
            verbose=0,
            compute_global_mean_bool=False,
            quad_interp=True,
        )
        y, x = zip(*obs_yearsvals_dict.items())
        plt.plot(
            x,
            y,
            "o",
            markersize=6,
            color="None",
            markeredgecolor="k",
            markeredgewidth=0.5,
        )
        plots.savefig(
            FIGURE_DIRECTORY + settings["exp_name"] + "_define_obs_tresholds", dpi=300
        )
        plt.show()

    if len(obs_yearsvals_dict) >= 2:
        if len(obs_yearsvals_dict) > 3:
            eligible_keys = np.sort(list(obs_yearsvals_dict.keys()))[-3:]
            i = np.where(np.isin(input_target_threshold, eligible_keys))[0]
            return (
                input_target_threshold[i],
                x_obs_concat[i],
                onehot_vec[i],
                obs_timeseries,
                obs_years,
                {k: obs_yearsvals_dict.get(k, None) for k in eligible_keys},
            )

        else:
            return (
                input_target_threshold,
                x_obs_concat,
                onehot_vec,
                obs_timeseries,
                obs_years,
                obs_yearsvals_dict,
            )
    else:
        return (
            input_target_threshold * np.nan,
            x_obs_concat * np.nan,
            onehot_vec * np.nan,
            obs_timeseries,
            obs_years,
            obs_yearsvals_dict,
        )


def get_single_labels(
    x,
    settings,
    plot=False,
    verbose=0,
    compute_global_mean_bool=False,
    quad_interp=False,
):
    x = compute_global_mean(x)
    x = x.rolling(time=settings["rolling_mean_len"], center=False).mean().dropna("time")

    labels, years, timeseries, warming_year = get_labels(
        x,
        settings,
        plot=plot,
        verbose=verbose,
        compute_global_mean_bool=compute_global_mean_bool,
        quad_interp=quad_interp,
    )

    return labels, years, timeseries, warming_year


def get_warming_year_single(
    settings,
    data,
    mask,
    IPCC_REGION_LIST,
    kind,
    n=10,
    loops=1,
    seed=44,
    show_plot=False,
):
    gcm_warming = np.zeros((data.shape[0], loops, len(IPCC_REGION_LIST)))
    gcm_warming_year = np.zeros(
        (
            data.shape[0],
            loops,
            len(
                IPCC_REGION_LIST,
            ),
        )
    )
    for ireg, ipcc_region in enumerate(IPCC_REGION_LIST):
        data_reg = xr.where(mask == ireg, data, np.nan).transpose(
            "gcm", "time", "lat", "lon"
        )

        for loop in np.arange(0, loops):
            for imodel in np.arange(0, data.shape[0]):
                if kind == "rolling":
                    is_cubic = False
                elif kind == "quad_interp":
                    is_cubic = True

                # get warming year
                labels, years, timeseries, warming_year = get_single_labels(
                    data_reg[imodel, :, :, :],
                    settings,
                    plot=False,
                    verbose=0,
                    compute_global_mean_bool=False,
                    quad_interp=is_cubic,
                )

                gcm_warming[imodel, loop, ireg] = timeseries[-1]
                gcm_warming_year[imodel, loop, ireg] = warming_year

                if ipcc_region in ("WCE") and show_plot:
                    # if show_plot:
                    plt.plot(years, timeseries, ".-")
                    plt.axvline(
                        x=gcm_warming_year[imodel, loop, ireg], color="r", linewidth=1.0
                    )
                    plt.axhline(
                        y=settings["target_temp"],
                        color="k",
                        linestyle="--",
                        linewidth=1.0,
                    )
                    plt.title(
                        str(imodel)
                        + ", "
                        + ipcc_region
                        + " : "
                        + str(gcm_warming_year[imodel, loop, ireg])
                    )
                    plt.show()

                    # raise ValueError()

    return gcm_warming_year, gcm_warming
