"""Network functions.

Classes
---------
Exponentiate(keras.layers.Layer)


Functions
---------
RegressLossExpSigma(y_true, y_pred)
compile_model(x_train, y_train, settings)


"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow import keras
import numpy as np

import custom_metrics


def add_cyclic_longitudes(inputs, nlon_wrap=10):
    # inputs: [sample, lat, lon, channel]
    # adding the last 10 lons to the front and first 10 lons to the end
    padded_inputs = tf.concat([inputs, inputs[:, :, :nlon_wrap]], axis=2)
    # padded_inputs = tf.concat([inputs[:, :, -1 * nlon_wrap :], padded_inputs], axis=2)

    return padded_inputs


class Exponentiate(keras.layers.Layer):
    """Custom layer to exp the sigma and tau estimates inline."""

    def __init__(self, **kwargs):
        super(Exponentiate, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)


def RegressLossExpSigma(y_true, y_pred):
    # network predictions
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]

    # normal distribution defined by N(mu,sigma)
    norm_dist = tfp.distributions.Normal(mu, sigma)

    # compute the log as the -log(p)
    loss = -norm_dist.log_prob(y_true[:, 0])

    return tf.reduce_mean(loss, axis=-1)


def compile_model(input_data, y_train, settings):
    x_train, target_temps = input_data

    # create input for target temp
    input_target_temp = Input(shape=(1,), name="input_target_temp")

    # create input for maps
    input_maps = Input(shape=x_train.shape[1:], name="input_maps")

    # normalization
    if settings["normalizer_index"] is None:
        print("no normalization layer.\n")
        layers = input_maps
    else:
        raise Warning("Are you sure you want this?")

        normalizer = tf.keras.layers.Normalization(axis=settings["normalizer_index"])
        normalizer.adapt(x_train)
        layers = normalizer(input_maps)

    # FEED FORWARD NETWORK
    if settings["architecture"] == "ffn":
        assert len(settings["hiddens"]) == len(
            settings["ridge_param"]
        ), "hiddens and ridge_param should be the same length."

        # dropout and split if necessary
        if settings["cumulative_history"]:
            layers, layers1 = tf.split(layers, num_or_size_splits=2, axis=-1)

            layers = tf.keras.layers.Flatten()(layers)
            layers = Dropout(rate=settings["dropout_rate"], seed=settings["rng_seed"])(
                layers
            )
            layers = tf.keras.layers.Concatenate(axis=-1)([layers, input_target_temp])

            layers1 = tf.keras.layers.Flatten()(layers1)
            layers1 = Dropout(rate=settings["dropout_rate"], seed=settings["rng_seed"])(
                layers1
            )
            layers1 = tf.keras.layers.Concatenate(axis=-1)([layers1, input_target_temp])

        else:
            layers = tf.keras.layers.Flatten()(layers)
            layers = Dropout(rate=settings["dropout_rate"], seed=settings["rng_seed"])(
                layers
            )
            layers = tf.keras.layers.Concatenate(axis=-1)([layers, input_target_temp])

        for hidden, ridge in zip(settings["hiddens"], settings["ridge_param"]):
            layers = Dense(
                hidden,
                activation=settings["activation"][0],
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                bias_initializer=tf.keras.initializers.RandomNormal(
                    seed=settings["rng_seed"]
                ),
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    seed=settings["rng_seed"]
                ),
            )(layers)

            if settings["cumulative_history"]:
                layers1 = Dense(
                    hidden,
                    activation=settings["activation"][0],
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        seed=settings["rng_seed"]
                    ),
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        seed=settings["rng_seed"]
                    ),
                )(layers1)

        if settings["cumulative_history"]:
            layers = tf.keras.layers.Concatenate(axis=-1)([layers, layers1])

    # CONVOLUTIONAL NEURAL NETWORK
    elif settings["architecture"] == "cnn":
        # convolutional layers
        layers = tf.keras.layers.Lambda(add_cyclic_longitudes, name="padded_inputs")(
            layers
        )

        for layer, kernel in enumerate(settings["filters"]):
            if settings["depthwise"]:
                layers = tf.keras.layers.DepthwiseConv2D(
                    depth_multiplier=kernel,
                    kernel_size=settings["kernel_size"][layer],
                    use_bias=True,
                    activation=settings["activation"][0],
                    padding="same",
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        seed=settings["rng_seed"]
                    ),
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        seed=settings["rng_seed"]
                    ),
                    name="conv_" + str(layer),
                )(layers)
            else:
                layers = tf.keras.layers.Conv2D(
                    filters=kernel,
                    kernel_size=settings["kernel_size"][layer],
                    use_bias=True,
                    activation=settings["activation"][0],
                    padding="same",
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        seed=settings["rng_seed"]
                    ),
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        seed=settings["rng_seed"]
                    ),
                    name="conv_" + str(layer),
                )(layers)
            layers = tf.keras.layers.MaxPooling2D(
                (2, 2), padding="same", name="maxpool_" + str(layer)
            )(layers)
        layers = tf.keras.layers.Flatten(name="flatten_0")(layers)

        # MU LAYERS --------------- --------------- --------------- ---------------
        layers_mu = tf.keras.layers.Concatenate(axis=-1, name="mu_concat_0")(
            [layers, input_target_temp]
        )

        # dense layers
        assert len(settings["hiddens"]) == len(
            settings["dropout_rate"]
        ), "hiddens and dropout_rate should be the same length."

        for layer, nodes in enumerate(settings["hiddens"]):
            if layer == 0:
                ridge_initial = settings["ridge_param"][0]
            else:
                ridge_initial = 0.0
            layers_mu = tf.keras.layers.Dense(
                nodes,
                activation=settings["activation"][1],
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    seed=settings["rng_seed"]
                ),
                bias_initializer=tf.keras.initializers.RandomNormal(
                    seed=settings["rng_seed"] + 1
                ),
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=0.0, l2=ridge_initial
                ),
                name="mu_dense_" + str(layer),
            )(layers_mu)

            layers_mu = tf.keras.layers.Dropout(
                settings["dropout_rate"][layer], name="mu_dropout_" + str(layer), seed=settings["rng_seed"]
            )(layers_mu)

        # SIGMA LAYERS --------------- --------------- --------------- ---------------
        layers_sigma = tf.keras.layers.Concatenate(axis=-1, name="sigma_concat_0")(
            [layers, input_target_temp]
        )

        # dense layers
        assert len(settings["hiddens"]) == len(
            settings["dropout_rate"]
        ), "hiddens and dropout_rate should be the same length."

        for layer, nodes in enumerate(settings["hiddens"]):
            if layer == 0:
                ridge_initial = settings["ridge_param"][0]
            else:
                ridge_initial = 0.0
            layers_sigma = tf.keras.layers.Dense(
                nodes,
                activation=settings["activation"][1],
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    seed=settings["rng_seed"]
                ),
                bias_initializer=tf.keras.initializers.RandomNormal(
                    seed=settings["rng_seed"] + 1
                ),
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=0.0, l2=ridge_initial
                ),
                name="sigma_dense_" + str(layer),
            )(layers_sigma)

            layers_sigma = tf.keras.layers.Dropout(
                settings["dropout_rate"][layer], name="sigma_dropout_" + str(layer), seed=settings["rng_seed"]
            )(layers_sigma)
    else:
        raise NotImplementedError()

    # final layer until output, concatenate target_temp again
    layers_mu = tf.keras.layers.Concatenate(axis=-1, name="mu_concat_1")(
        [layers_mu, input_target_temp]
    )
    layers_mu = Dense(
        settings["penultimate_hiddens"],
        activation=settings["activation"][-1],
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.0),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            seed=settings["rng_seed"]
        ),
        name="mu_finaldense",
    )(layers_mu)

    layers_sigma = tf.keras.layers.Concatenate(axis=-1, name="sigma_concat_1")(
        [layers_sigma, input_target_temp]
    )
    layers_sigma = Dense(
        settings["penultimate_hiddens"],
        activation=settings["activation"][-1],
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.0),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            seed=settings["rng_seed"]
        ),
        name="sigma_finaldense",
    )(layers_sigma)

    # Setup the final output
    if settings["network_type"] == "reg":
        LOSS = "mae"
        metrics = [
            "mse",
        ]

        output_layer = Dense(
            1,
            activation="linear",
            bias_initializer=tf.keras.initializers.RandomNormal(
                seed=settings["rng_seed"]
            ),
            kernel_initializer=tf.keras.initializers.RandomNormal(
                seed=settings["rng_seed"]
            ),
            name="output_layer",
        )(layers)

    elif settings["network_type"] == "shash2":
        LOSS = RegressLossExpSigma
        metrics = [
            custom_metrics.CustomMAE(name="custom_mae"),
            custom_metrics.InterquartileCapture(name="interquartile_capture"),
            custom_metrics.SignTest(name="sign_test"),
        ]

        y_avg = np.mean(y_train)
        y_std = np.std(y_train)

        mu_z_unit = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            use_bias=True,
            # bias_initializer=tf.keras.initializers.Zeros(),
            # kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.RandomNormal(
                seed=settings["rng_seed"] + 100
            ),
            kernel_initializer=tf.keras.initializers.RandomNormal(
                seed=settings["rng_seed"] + 100
            ),
            name="mu_z_unit",
        )(layers_mu)

        mu_unit = tf.keras.layers.Rescaling(
            scale=y_std,
            offset=y_avg,
            name="mu_unit",
        )(mu_z_unit)

        # sigma_unit. The network predicts the log of the scaled sigma_z, then
        # the resclaing layer scales it up to log of sigma y, and the custom
        # Exponentiate layer converts it to sigma_y.
        log_sigma_z_unit = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            use_bias=True,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.keras.initializers.Zeros(),
            name="log_sigma_z_unit",
        )(layers_sigma)

        log_sigma_unit = tf.keras.layers.Rescaling(
            scale=1.0,
            offset=np.log(y_std),
            name="log_sigma_unit",
        )(log_sigma_z_unit)

        sigma_unit = Exponentiate(
            name="sigma_unit",
        )(log_sigma_unit)

        output_layer = tf.keras.layers.concatenate(
            [mu_unit, sigma_unit], axis=1, name="output_layer"
        )

    elif settings["network_type"] == "shash3":
        print("here")

    else:
        raise NotImplementedError("no such network_type")

    # Constructing the model
    model = Model((input_maps, input_target_temp), output_layer)
    try:
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=settings["learning_rate"]
            ),
            loss=LOSS,
            metrics=metrics,
        )
    except:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
            loss=LOSS,
            metrics=metrics,
        )

    print(model.summary())

    return model


def make_transfer_model(settings, transfer_model, trainable_id, verbose=True):
    if settings["network_type"] == "reg":
        loss = "mae"
    else:
        loss = RegressLossExpSigma

    transfer_model.trainable = True
    for layer in transfer_model.layers:
        if (trainable_id not in layer.name):
            layer.trainable = False
        if verbose:
            print(layer.name, layer.trainable)

    if verbose:
        transfer_model.summary()

    return transfer_model, loss
