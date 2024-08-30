import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import custom_metrics
import file_methods
from scipy.optimize import curve_fit
import xarray as xr
import pandas as pd

import data_processing
import network
import plots


MODEL_DIRECTORY = "saved_models/"
FIGURE_DIRECTORY = "figures/"
PREDICTIONS_DIRECTORY = "saved_predictions/"


def perform_transfer_learning(transfer_model, da_obs, x_obs, settings, plot=False, suffix=""):

    # make the transfer data
    transfer_temp_vec = settings["transfer_temp_vec"]
    (
        input_target_threshold,
        x_obs_concat,
        onehot_vec,
        obs_timeseries,
        obs_years,
        obs_yearsvals_dict,
    ) = data_processing.make_transfer_data(
        settings,
        transfer_temp_vec,
        da_obs,
        x_obs,
        plot=plot,
        quad_interp=True,
    )

    # create the transfer model and compile
    transfer_model, loss = network.make_transfer_model(
        settings,
        transfer_model,
        trainable_id="mu",
        verbose=False,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="custom_mae",
        patience=settings["transfer_patience"],
        min_delta=settings["transfer_min_delta"],
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    transfer_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=settings["learning_rate"] / 5.0
        ),
        loss=loss,
        metrics=[
            custom_metrics.CustomMAE(name="custom_mae"),
        ],
    )

    # train the transfer model
    history_obs = transfer_model.fit(
        (x_obs_concat, input_target_threshold),
        onehot_vec,
        batch_size=input_target_threshold.shape[0],
        epochs=10_000,
        verbose=0,
        callbacks=[
            early_stopping,
        ],
    )

    # save the transfer model
    transfer_model_name = (
        file_methods.get_model_name(settings) + "_transfer" + suffix
    )
    file_methods.save_tf_model(
        transfer_model, transfer_model_name, MODEL_DIRECTORY, settings
    )

    # # plot the loss for the transferring
    if plot:
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.plot(history_obs.history["loss"])
        plt.title("loss")
        plt.subplot(1, 2, 2)
        plt.plot(history_obs.history["custom_mae"])
        plt.title("custom_mae")
        plt.show()
        # plt.close()

    return transfer_model, obs_timeseries, obs_years, obs_yearsvals_dict


def compute_threshold_predictions(obs_transfer_dict, ireg, transfer_model, x_obs):

    # evaluate the model and save the results
    for temp_thresh in (1.5, 2.0, 3.0):
        target_temp_obs = np.ones((x_obs.shape[0],)) * temp_thresh
        obs_transfer_dict[temp_thresh][ireg, :] = transfer_model.predict(
            (x_obs, target_temp_obs),
            verbose=None,
        )[-1, :]
    _ = gc.collect()

    return obs_transfer_dict
