import sys
import os
import importlib as imp
import gc

import scipy.stats as stats
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import random
import experiment_settings
import file_methods
import plots
import custom_metrics
import network
import data_processing
import initial_setup
import regionmask
from scipy.optimize import curve_fit

import matplotlib as mpl

mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150
plt.style.use("seaborn-v0_8-notebook")
savefig_dpi = 300

# np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on


print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"tensorflow version = {tf.__version__}")
print(f"tensorflow-probability version = {tfp.__version__}")

# ------------------------------------------------------------------------------------------
PARENT_EXP_NAME = "exp068"
OVERWRITE_MODEL = False
RERUN_METRICS = False

MODEL_DIRECTORY = "saved_models/"
PREDICTIONS_DIRECTORY = "saved_predictions/"
DATA_DIRECTORY = "../../../2022/target_temp_detection/data/"
OBS_DIRECTORY = "../data/"
DIAGNOSTICS_DIRECTORY = "model_diagnostics/"
FIGURE_DIRECTORY = "figures/"

# initial_setup.setup_directories()

# ------------------------------------------------------------------------------------------
ar6_land = regionmask.defined_regions.ar6.land
IPCC_REGION_LIST = ar6_land.abbrevs
ran_region = False

for ipcc_region in IPCC_REGION_LIST:

    # if ipcc_region not in ("WCE", "SAH", "CNA", "CAU", "NSA", "ESB"):
    #     continue
    if ipcc_region not in ("WCE",):
        continue

    settings = experiment_settings.get_settings(PARENT_EXP_NAME)
    settings["exp_name"] = PARENT_EXP_NAME + "_" + ipcc_region
    settings["target_region"] = "ipcc_" + ipcc_region
    settings["obs_training_only"] = True

    for rng_seed in settings["rng_seed_list"]:
        # define random number generator
        settings["rng_seed"] = rng_seed
        np.random.seed(settings["rng_seed"])
        random.seed(settings["rng_seed"])
        tf.random.set_seed(settings["rng_seed"])

        # get model name
        model_name = file_methods.get_model_name(settings)

        # skip if already exists and overwrite is false
        if (
            os.path.exists(MODEL_DIRECTORY + model_name + "_model")
            and OVERWRITE_MODEL is False
        ):
            model_exists = True
            if RERUN_METRICS:
                pass
            else:
                continue
        else:
            model_exists = False
            ran_region = True
            print("--------------------------------------------")
            print(settings["exp_name"])
            print(model_name)
            print("--------------------------------------------")

        # get the data
        N_TRAIN, N_VAL, N_TEST, ALL_MEMBERS = data_processing.get_members(settings)
        (
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
            target_temps_train,
            target_temps_val,
            target_temps_test,
            target_years,
            map_shape,
            settings,
        ) = data_processing.create_data(DATA_DIRECTORY, settings)

        # determine how many GCMs are being used for later re-shaping
        N_GCMS = len(file_methods.get_cmip_filenames(settings, verbose=0))
        N_TARGETS = len(np.unique(target_temps_train))

        # ----------------------------------------
        tf.keras.backend.clear_session()

        # define early stopping callback (cannot be done elsewhere)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=settings["patience"],
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )

        if model_exists:
            print(model_name + "exists. Loading the model...\n")
            model = file_methods.load_tf_model(model_name, MODEL_DIRECTORY)
            history = None
        else:
            # create and compile the model
            model = network.compile_model(
                (x_train, target_temps_train), y_train, settings
            )

            history = model.fit(
                (x_train, target_temps_train),
                onehot_train,
                epochs=settings["n_epochs"],
                batch_size=settings["batch_size"],
                shuffle=True,
                validation_data=[(x_val, target_temps_val), onehot_val],
                callbacks=[
                    early_stopping,
                ],
                verbose=2,
            )

            # ----------------------------------------
            # save the tensorflow model
            file_methods.save_tf_model(model, model_name, MODEL_DIRECTORY, settings)

        # make and save cmip predictions for train/val/test
        pred_train = model.predict((x_train, target_temps_train), verbose=None)
        pred_val = model.predict((x_val, target_temps_val), verbose=None)
        pred_test = model.predict((x_test, target_temps_test), verbose=None)
        _ = gc.collect()

        # file_methods.save_predictions(
        #     pred_train,
        #     PREDICTIONS_DIRECTORY + model_name + "_predictions_train",
        # )
        # file_methods.save_predictions(
        #     pred_val,
        #     PREDICTIONS_DIRECTORY + model_name + "_predictions_val",
        # )
        # file_methods.save_predictions(
        #     pred_test,
        #     PREDICTIONS_DIRECTORY + model_name + "_predictions_test",
        # )

        # ----------------------------------------
        # compute cmip metrics to compare
        error_val = np.mean(np.abs(pred_val[:, 0] - onehot_val[:, 0]))
        error_test = np.mean(np.abs(pred_test[:, 0] - onehot_test[:, 0]))
        __, __, d_val, __ = custom_metrics.compute_pit(onehot_val, pred_val)
        __, __, d_test, __ = custom_metrics.compute_pit(onehot_test, pred_test)
        __, __, d_valtest, __ = custom_metrics.compute_pit(
            np.append(onehot_val, onehot_test, axis=0),
            model.predict(
                (
                    np.append(x_val, x_test, axis=0),
                    np.append(target_temps_val, target_temps_test, axis=0),
                ),
                verbose=None,
            ),
        )
        loss_val = network.RegressLossExpSigma(onehot_val, pred_val).numpy()
        loss_test = network.RegressLossExpSigma(onehot_test, pred_test).numpy()

        # ----------------------------------------
        # create and save diagnostics plots
        if history is not None:
            plots.plot_metrics_panels(history, settings)
            plt.savefig(
                DIAGNOSTICS_DIRECTORY + model_name + "_metrics_diagnostic" + ".png",
                dpi=savefig_dpi,
            )
            plt.close()
            # plt.show()

        plots.plot_one_to_one_diagnostic(
            settings,
            model,
            pred_train,
            pred_val,
            pred_test,
            y_train,
            y_val,
            y_test,
            target_years,
            y_yrs_train,
            N_GCMS,
            N_VAL,
            N_TARGETS,
        )
        plt.savefig(
            DIAGNOSTICS_DIRECTORY + model_name + "_one_to_one_diagnostic" + ".png",
            dpi=savefig_dpi,
        )
        plt.close()

        # ----------------------------------------
        # fill and save the metrics dictionary
        d = {}
        d["exp_name"] = settings["exp_name"]
        d["rng_seed"] = settings["rng_seed"]
        d["hiddens"] = str(settings["hiddens"])
        d["ridge_param"] = settings["ridge_param"][0]
        d["error_val"] = error_val
        d["error_test"] = error_test
        d["loss_val"] = loss_val
        d["loss_test"] = loss_test
        d["d_val"] = d_val
        d["d_test"] = d_test
        d["d_valtest"] = d_valtest

        df = pd.DataFrame(d, index=[0])
        df.to_pickle(PREDICTIONS_DIRECTORY + model_name + "_metrics.pickle")

    if ran_region:
        break

if ipcc_region != IPCC_REGION_LIST[-1]:
    os.execv(sys.executable, ["python"] + sys.argv)
