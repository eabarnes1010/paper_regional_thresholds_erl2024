"""Metrics for generic plotting.

Functions
---------
plot_metrics(history,metric)
plot_metrics_panels(history, settings)
plot_map(x, clim=None, title=None, text=None, cmap='RdGy')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as ct
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import custom_metrics
import cmaps as cmaps_ncl
import regionmask
import gc
from scipy.optimize import curve_fit
import data_processing
from matplotlib import colors


mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150

FS = 10
plt.rc("text", usetex=False)
plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
plt.rc("savefig", facecolor="white")
plt.rc("axes", facecolor="white")
plt.rc("axes", labelcolor="dimgrey")
plt.rc("axes", labelcolor="dimgrey")
plt.rc("xtick", color="dimgrey")
plt.rc("ytick", color="dimgrey")


def savefig(filename, dpi=300):
    for fig_format in (".png", ".pdf"):
        plt.savefig(filename + fig_format, bbox_inches="tight", dpi=dpi)


def get_discrete_colornorm(cb_bounds, cmap):
    cb_n = int((cb_bounds[1] - cb_bounds[0]) / cb_bounds[-1])
    # cbar_n = (cb_bounds[1] - cb_bounds[-1]) - (cb_bounds[0] - cb_bounds[-1])
    clr_norm = colors.BoundaryNorm(
        np.linspace(
            cb_bounds[0] - cb_bounds[-1] / 2, cb_bounds[1] + cb_bounds[-1] / 2, cb_n + 2
        ),
        cmap.N,
    )

    return clr_norm


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))
        else:
            spine.set_color("none")
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])


def format_spines(ax):
    adjust_spines(ax, ["left", "bottom"])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("dimgrey")
    ax.spines["bottom"].set_color("dimgrey")
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params("both", length=4, width=2, which="major", color="dimgrey")


def plot_metrics(history, metric):
    imin = np.argmin(history.history["val_loss"])

    try:
        plt.plot(history.history[metric], label="training")
        plt.plot(history.history["val_" + metric], label="validation")
        plt.title(metric)
        plt.axvline(x=imin, linewidth=0.5, color="gray", alpha=0.5)
        plt.legend()
    except:
        pass


def plot_one_to_one_diagnostic(
    settings,
    model,
    predict_train,
    predict_val,
    predict_test,
    y_train,
    y_val,
    y_test,
    target_years,
    y_yrs_train,
    N_GCMS,
    N_VAL,
    N_TARGETS,
):
    if settings["network_type"] == "shash2":
        top_pred_idx = 0
    else:
        top_pred_idx = None

    YEARS_UNIQUE = np.unique(y_yrs_train)
    predict_train = predict_train[:, top_pred_idx].flatten()
    predict_val = predict_val[:, top_pred_idx].flatten()
    predict_test = predict_test[:, top_pred_idx].flatten()
    mae = np.mean(np.abs(predict_test - y_test[:]))

    # --------------------------------
    clr = (
        "tab:purple",
        "tab:orange",
        "tab:blue",
        "tab:green",
        "gold",
        "brown",
        "black",
        "darkorange",
        "fuchsia",
        "cornflowerblue",
        "lime",
    )
    plt.subplots(1, 2, figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(y_train, predict_train, ".", color="gray", alpha=0.25, label="training")
    plt.plot(
        y_val,
        predict_val,
        ".",
        label="validation",
        color="gray",
        alpha=0.75,
    )
    plt.plot(y_test, predict_test, ".", label="testing")
    plt.plot(y_train, y_train, "--", color="fuchsia")
    plt.axvline(x=0, color="gray", linewidth=1)
    plt.axhline(y=0, color="gray", linewidth=1)
    plt.title("Testing MAE = " + str(mae.round(2)) + " years")
    plt.xlabel("true number of years until target is reached")
    plt.ylabel("predicted number of years until target is reached")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_yrs_train, predict_train, ".", color="gray", alpha=0.5, label="training")
    plt.title(
        "Time to Target Year for "
        + str(settings["target_temp"])
        + "C using ssp"
        + str(settings["ssp"])
    )
    plt.xlabel("year of map")
    plt.ylabel("predicted number of years until target is reached")
    plt.axhline(y=0, color="gray", linewidth=1)

    predict_val_mat = predict_val.reshape(N_GCMS, N_VAL, len(YEARS_UNIQUE), N_TARGETS)
    for t in range(N_TARGETS):
        for i in np.arange(0, predict_val_mat.shape[0]):
            plt.plot(
                YEARS_UNIQUE,
                predict_val_mat[i, :, :, t].swapaxes(1, 0),
                ".",
                label="validation",
                color=clr[i],
            )
            plt.axvline(x=target_years[i], linestyle="--", color=clr[i])


def plot_metrics_panels(history, settings):
    if settings["network_type"] == "reg":
        error_name = "mse"
    elif settings["network_type"] == "shash2":
        error_name = "custom_mae"
    else:
        raise NotImplementedError("no such network_type")

    # imin = len(history.history[error_name])
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plot_metrics(history, "loss")
    plt.ylim(0, 10.0)

    plt.subplot(1, 4, 2)
    plot_metrics(history, error_name)
    plt.ylim(0, 10)

    try:
        plt.subplot(1, 4, 3)
        plot_metrics(history, "interquartile_capture")

        plt.subplot(1, 4, 4)
        plot_metrics(history, "sign_test")
    except:
        pass
    plt.tight_layout()


def plot_map(x, clim=None, title=None, text=None, cmap="RdGy"):
    plt.pcolor(
        x,
        cmap=cmap,
    )
    plt.clim(clim)
    plt.colorbar()
    plt.title(title, fontsize=15, loc="right")
    plt.yticks([])
    plt.xticks([])

    plt.text(
        0.01,
        1.0,
        text,
        fontfamily="monospace",
        fontsize="small",
        va="bottom",
        transform=plt.gca().transAxes,
    )


def drawOnGlobe(
    ax,
    map_proj,
    data,
    lats,
    lons,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    inc=None,
    cbarBool=True,
    contourMap=[],
    contourVals=[],
    fastBool=False,
    extent="both",
):
    data_crs = ct.crs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(
        data, coord=lons
    )  # fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
    data_cyc = data
    lons_cyc = lons

    #     ax.set_global()
    #     ax.coastlines(linewidth = 1.2, color='black')
    #     ax.add_feature(cartopy.feature.LAND, zorder=0, scale = '50m', edgecolor='black', facecolor='black')
    land_feature = cfeature.NaturalEarthFeature(
        category="physical",
        name="land",
        scale="50m",
        facecolor="None",
        edgecolor="k",
        linewidth=0.5,
    )
    ax.add_feature(land_feature)
    #     ax.GeoAxes.patch.set_facecolor('black')

    if fastBool:
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap)
    #         image = ax.contourf(lons_cyc, lats, data_cyc, np.linspace(0,vmax,20),transform=data_crs, cmap=cmap)
    else:
        image = ax.pcolor(
            lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, shading="auto"
        )

    if np.size(contourMap) != 0:
        contourMap_cyc, __ = add_cyclic_point(
            contourMap, coord=lons
        )  # fixes white line by adding point
        ax.contour(
            lons_cyc,
            lats,
            contourMap_cyc,
            contourVals,
            transform=data_crs,
            colors="fuchsia",
        )

    if cbarBool:
        cb = plt.colorbar(
            image, shrink=0.45, orientation="horizontal", pad=0.02, extend=extent
        )
        cb.ax.tick_params(labelsize=6)
    else:
        cb = None

    image.set_clim(vmin, vmax)

    return cb, image


def add_cyclic_point(data, coord=None, axis=-1):
    # had issues with cartopy finding utils so copied for myself

    if coord is not None:
        if coord.ndim != 1:
            raise ValueError("The coordinate must be 1-dimensional.")
        if len(coord) != data.shape[axis]:
            raise ValueError(
                "The length of the coordinate does not match "
                "the size of the corresponding dimension of "
                "the data array: len(coord) = {}, "
                "data.shape[{}] = {}.".format(len(coord), axis, data.shape[axis])
            )
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError("The coordinate must be equally spaced.")
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError(
            "The specified axis does not correspond to an array dimension."
        )
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value


def plot_pits(ax, x_val, onehot_val, model_shash):
    plt.sca(ax)
    clr_shash = "tab:blue"

    # shash pit
    bins, hist_shash, D_shash, EDp_shash = custom_metrics.compute_pit(
        onehot_val, x_data=x_val, model_shash=model_shash
    )
    bins_inc = bins[1] - bins[0]

    bin_add = bins_inc / 2
    bin_width = bins_inc * 0.98
    ax.bar(
        hist_shash[1][:-1] + bin_add,
        hist_shash[0],
        width=bin_width,
        color=clr_shash,
        label="SHASH",
    )

    # make the figure pretty
    ax.axhline(
        y=0.1,
        linestyle="--",
        color="k",
        linewidth=2.0,
    )
    # ax = plt.gca()
    yticks = np.around(np.arange(0, 0.55, 0.05), 2)
    plt.yticks(yticks, yticks)
    ax.set_ylim(0, 0.25)
    ax.set_xticks(bins, np.around(bins, 1))

    plt.text(
        0.0,
        np.max(ax.get_ylim()) * 0.99,
        "SHASH D: "
        + str(np.round(D_shash, 4))
        + " ("
        + str(np.round(EDp_shash, 3))
        + ")",
        color=clr_shash,
        verticalalignment="top",
        fontsize=12,
    )

    ax.set_xlabel("probability integral transform")
    ax.set_ylabel("probability")
    # plt.legend(loc=1)
    # plt.title('PIT histogram comparison', fontsize=FS, color='k')


def plot_xai_heatmaps(
    xplot,
    xplot_transfer,
    xplot_cmip,
    lat,
    lon,
    ipcc_region,
    subplots=3,
    scaling=1,
    diff_scaling=1.0,
    title=None,
    colorbar=True,
):
    c = cmaps_ncl.BlueDarkRed18_r.colors
    c = np.insert(c, 9, [1, 1, 1], axis=0)
    cmap = mpl.colors.ListedColormap(c)

    transform = ct.crs.PlateCarree()
    projection = ct.crs.EqualEarth(central_longitude=0.0)

    xplot = xplot * scaling
    xplot_transfer = xplot_transfer * scaling
    if subplots == 3:
        xplot_cmip = xplot_cmip * scaling

    fig = plt.figure(figsize=(1.5 * 5.25 * 2, 1.5 * 3.25 * 1), dpi=200)

    if subplots == 3:
        a1 = fig.add_subplot(1, 3, 1, projection=projection)
        c1 = a1.pcolormesh(
            lon,
            lat,
            xplot_cmip,
            cmap=cmap,
            transform=transform,
        )
        a1.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "land",
                "110m",
                edgecolor="k",
                linewidth=0.5,
                facecolor="None",
            )
        )
        regionmask.defined_regions.ar6.land[(ipcc_region,)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(
                linewidth=1.0,
            ),
        )
        c1.set_clim(-1, 1)
        if colorbar:
            fig.colorbar(
                c1,
                orientation="horizontal",
                shrink=0.35,
                extend="both",
                pad=0.02,
            )
        if title is not None:
            plt.title("(a) CMIP6 SHAP " + title)

    a1 = fig.add_subplot(1, 3, 2, projection=projection)
    c1 = a1.pcolormesh(
        lon,
        lat,
        xplot,
        cmap=cmap,
        transform=transform,
    )
    a1.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "110m",
            edgecolor="k",
            linewidth=0.5,
            facecolor="None",
        )
    )
    regionmask.defined_regions.ar6.land[(ipcc_region,)].plot(
        add_label=False,
        label_multipolygon="all",
        add_ocean=False,
        ocean_kws=dict(color="lightblue", alpha=0.25),
        line_kws=dict(
            linewidth=1.0,
        ),
    )
    c1.set_clim(-1, 1)
    if colorbar:
        fig.colorbar(
            c1,
            orientation="horizontal",
            shrink=0.35,
            extend="both",
            pad=0.02,
        )
    if title is not None:
        plt.title("(b) Observations SHAP " + title)

    a1 = fig.add_subplot(1, 3, 3, projection=projection)
    c1 = a1.pcolormesh(
        lon,
        lat,
        xplot_transfer - xplot,
        cmap=cmap,
        transform=transform,
    )
    a1.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "110m",
            edgecolor="k",
            linewidth=0.5,
            facecolor="None",
        )
    )
    regionmask.defined_regions.ar6.land[(ipcc_region,)].plot(
        add_label=False,
        label_multipolygon="all",
        add_ocean=False,
        ocean_kws=dict(color="lightblue", alpha=0.25),
        line_kws=dict(
            linewidth=1.0,
        ),
    )
    c1.set_clim(-1.0 * diff_scaling, 1.0 * diff_scaling)
    if colorbar:
        fig.colorbar(
            c1,
            orientation="horizontal",
            shrink=0.35,
            extend="both",
            pad=0.02,
        )
    if title is not None:
        plt.title("(c) Transfer minus Original SHAP " + title)


def plot_transferlearning_timeseries(
    model,
    transfer_model,
    x_obs,
    obs_timeseries,
    obs_years,
    obs_yearsvals_dict,
    cmip_masked_region,
    settings,
    title=None,
):
    target_vec = np.arange(0.5, 5.1, 0.1).round(3)
    obs_pred = np.zeros((target_vec.shape[0], 2)) * np.nan
    obs_transfer = np.zeros((target_vec.shape[0], 2)) * np.nan
    for i, target in enumerate(target_vec):
        target_temp_obs = np.ones((x_obs.shape[0],)) * target
        obs_pred[i, :] = model.predict(
            (x_obs, target_temp_obs),
            verbose=None,
        )[-1, :]
        obs_transfer[i, :] = transfer_model.predict(
            (x_obs, target_temp_obs),
            verbose=None,
        )[-1, :]
        _ = gc.collect()

    # plot the results
    plt.figure(figsize=(6, 4))
    plt.plot(
        obs_years,
        obs_timeseries,
        "-",
        color="k",
        label="Smoothed Berkeley Observations",
    )

    plt.plot(
        obs_pred[:, 0] + settings["final_year_of_obs"],
        target_vec,
        color="teal",
        linewidth=1.5,
        alpha=0.75,
        label="Base-CNN Initialized with " + str(settings["final_year_of_obs"]) + " Observations",
    )
    plt.fill_betweenx(
        target_vec,
        obs_pred[:, 0] + settings["final_year_of_obs"] - obs_pred[:, 1],
        obs_pred[:, 0] + settings["final_year_of_obs"] + obs_pred[:, 1],
        color="teal",
        alpha=0.2,
    )

    plt.plot(
        obs_transfer[:, 0] + settings["final_year_of_obs"],
        target_vec,
        color="darkorange",
        linewidth=1.5,
        alpha=0.75,
        label=f"Transfer-CNN Initialized with {settings['final_year_of_obs']} Observations",
    )
    plt.fill_betweenx(
        target_vec,
        obs_transfer[:, 0] + settings["final_year_of_obs"] - obs_transfer[:, 1],
        obs_transfer[:, 0] + settings["final_year_of_obs"] + obs_transfer[:, 1],
        color="darkorange",
        alpha=0.2,
    )

    # plot CMIP6 projections
    for imodel in np.arange(0, cmip_masked_region.shape[0]):
        x = data_processing.compute_global_mean(cmip_masked_region[imodel, :, :, :])
        x = (
            x.rolling(time=settings["rolling_mean_len"], center=False)
            .mean()
            .dropna("time")
        )
        plt.plot(x["time.year"], x, linewidth=0.5, color="gray", alpha=0.25, zorder=1)
    plt.plot(
        0,
        0,
        linewidth=0.75,
        color="gray",
        alpha=0.25,
        zorder=1,
        label="CMIP6 Projections",
    )

    # plot transfer circles
    if len(obs_yearsvals_dict) >= 2:
        y, x = zip(*obs_yearsvals_dict.items())
        plt.plot(
            x,
            y,
            "o",
            markersize=6,
            color="None",
            markeredgecolor="k",
            markeredgewidth=0.5,
            label="Transfer-Learning Thresholds",
        )

    plt.axhline(y=1.5, linewidth=1.0, color="gray", linestyle=":")
    plt.axhline(y=2.0, linewidth=1.0, color="gray", linestyle=":")
    plt.axhline(y=3.0, linewidth=1.0, color="gray", linestyle=":")

    plt.ylabel("temperature anomaly (deg. C)")
    plt.legend(fontsize=6, loc="upper left")
    if title is not None:
        plt.title(title)
    plt.xlim(1950, 2100)
    plt.ylim(-0.5, 4.5)
