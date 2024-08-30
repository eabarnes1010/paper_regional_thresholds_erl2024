"""Functions for XAI analysis.

Functions
---------
get_gradients(inputs, top_pred_idx=None)
get_integrated_gradients(inputs, baseline=None, num_steps=50, top_pred_idx=None)
random_baseline_integrated_gradients(inputs, num_steps=50, num_runs=5, top_pred_idx=None)

"""

import tensorflow as tf
import numpy as np
import shap
import gc


def get_gradient_xai(model, inputs, top_pred_idx=None):
    input_1 = tf.cast(inputs[0], tf.float32)
    input_2 = tf.cast(inputs[1], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_2)
        tape.watch(input_1)

        # Run the forward pass of the layer and record operations
        # on GradientTape.
        preds = model([input_1, input_2], training=False)

        # For classification, grab the top class
        if top_pred_idx is not None:
            preds = preds[:, top_pred_idx]

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(preds, [input_1, input_2])
    return grads


def get_gradients(model, inputs, top_pred_idx=None):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        inputs: 2D/3D/4D matrix of samples
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    inputs = tf.cast(inputs, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)

        # Run the forward pass of the layer and record operations
        # on GradientTape.
        preds = model(inputs, training=False)

        # For classification, grab the top class
        if top_pred_idx is not None:
            preds = preds[:, top_pred_idx]

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(preds, inputs)
    return grads


def get_integrated_gradients(
    model, inputs, baseline=None, num_steps=50, top_pred_idx=None
):
    """Computes Integrated Gradients for a prediction.

    Args:
        inputs (ndarray): 2D/3D/4D matrix of samples
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.

    Returns:
        Integrated gradients w.r.t input image
    """
    # If baseline is not provided, start with zeros
    # having same size as the input image.
    if baseline is None:
        input_size = np.shape(inputs)[1:]
        baseline = np.zeros(input_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    inputs = inputs.astype(np.float32)
    interpolated_inputs = [
        baseline + (step / num_steps) * (inputs - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_inputs = np.array(interpolated_inputs).astype(np.float32)

    # 3. Get the gradients
    grads = []
    for i, x_data in enumerate(interpolated_inputs):
        grad = get_gradients(model, x_data, top_pred_idx=top_pred_idx)
        # grads.append(grad[0]) WRONG
        grads.append(grad)
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    # 5. Calculate integrated gradients and return
    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads


def random_baseline_integrated_gradients(
    model, inputs, num_steps=50, num_runs=5, top_pred_idx=None
):
    """Generates a number of random baseline images.

    Args:
        inputs (ndarray): 2D/3D/4D matrix of samples
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        num_runs: number of baseline images to generate
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.

    Returns:
        Averaged integrated gradients for `num_runs` baseline images
    """
    # 1. List to keep track of Integrated Gradients (IG) for all the images
    integrated_grads = []

    # 2. Get the integrated gradients for all the baselines
    for run in range(num_runs):
        baseline = np.zeros(np.shape(inputs)[1:])
        for i in np.arange(0, np.shape(baseline)[0]):
            j = np.random.choice(np.arange(0, np.shape(inputs)[0]))
            baseline[i] = inputs[j, i]

        igrads = get_integrated_gradients(
            inputs=inputs,
            baseline=baseline,
            num_steps=num_steps,
        )
        integrated_grads.append(igrads)

    # 3. Return the average integrated gradients for the image
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)


def calculate_shap_values(
    model,
    transfer_model,
    x_obs,
    x_obs_years,
    x_test,
    y_yrs_test,
    xai_settings,
    baseline_factor=1,
):
    shap_target_temp = xai_settings["target_temp"]

    i_yrobs = np.where(
        (x_obs_years >= xai_settings["obs_start"])
        & (x_obs_years <= xai_settings["obs_end"])
    )[0]

    rng = np.random.default_rng(xai_settings["rng_seed"])
    i_yrbase = np.where(
        (y_yrs_test >= xai_settings["baseline_start"])
        & (y_yrs_test <= xai_settings["baseline_end"])
    )[0]
    i_yrbase = rng.choice(i_yrbase, xai_settings["n_base_samples"], replace=False)

    i_yrcmip = np.where(
        (y_yrs_test >= xai_settings["obs_start"])
        & (y_yrs_test <= xai_settings["obs_end"])
    )[0]
    if xai_settings["n_cmip_samples"] is None:
        i_yrcmip = rng.choice(i_yrcmip, size=len(i_yrobs), replace=False)
    else:
        i_yrcmip = rng.choice(
            i_yrcmip, size=xai_settings["n_cmip_samples"], replace=False
        )

    # set deep-shape settings so there are no errors
    tf.keras.utils.set_random_seed(xai_settings["rng_seed"])
    shap.explainers._deep.deep_tf.op_handlers[
        "AddV2"
    ] = shap.explainers._deep.deep_tf.passthrough

    # deep-shap on the transfer model first
    explainer_transfer = shap.DeepExplainer(
        transfer_model,
        [
            baseline_factor * x_test[i_yrbase, :, :, :],
            shap_target_temp * np.ones((len(i_yrbase), 1)),
        ],
    )

    transfer_shap = explainer_transfer.shap_values(
        [x_obs[i_yrobs, :, :, :], shap_target_temp * np.ones((i_yrobs.shape[0], 1))],
        check_additivity=True,
    )
    _ = gc.collect()

    # deep-shap on the original base model
    explainer = shap.DeepExplainer(
        model,
        [
            baseline_factor * x_test[i_yrbase, :, :, :],
            shap_target_temp * np.ones((len(i_yrbase), 1)),
        ],
    )

    original_shap = explainer.shap_values(
        [x_obs[i_yrobs, :, :, :], shap_target_temp * np.ones((i_yrobs.shape[0], 1))],
        check_additivity=True,
    )
    cmip_shap = explainer.shap_values(
        [
            x_test[i_yrcmip, :, :, :],
            shap_target_temp * np.ones((i_yrcmip.shape[0], 1)),
        ],
        check_additivity=True,
    )

    # print(
    #     transfer_model.predict(
    #         [
    #             x_obs[i_yrobs, :, :, :],
    #             shap_target_temp * np.ones((i_yrobs.shape[0], 1)),
    #         ],
    #         verbose=None,
    #     )[-1]
    # )
    # print(
    #     transfer_model(
    #         [x_obs[i_yrobs, :, :, :], shap_target_temp * np.ones((i_yrobs.shape[0], 1))]
    #     ).numpy()[-1]
    # )

    print(f"{explainer.expected_value.numpy() = }")
    print(f"{explainer_transfer.expected_value.numpy() = }")

    return (
        original_shap,
        transfer_shap,
        cmip_shap,
        explainer.expected_value.numpy(),
        explainer_transfer.expected_value.numpy(),
    )
