"""Experimental settings

Functions
---------
get_settings(experiment_name)
"""

__author__ = "Noah Diffenbaugh and Elizabeth A. Barnes"
__date__ = "28 December 2023"


def get_settings(experiment_name):
    experiments = {

        # MAIN EXP: only using current year, no cumulative, high learning rate
        "exp134": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,  # 0.00001
            "kernel_size": [(5, 5), (3, 3), (3, 3)],
            "filters": [32, 32, 32],
            "hiddens": [10, 10, 10],
            "dropout_rate": [0.0, 0.0, 0.0],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },

        # NO HOT MODELS: like main experiment but no hot models
        "exp133": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "NOHOT",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,  # 0.00001
            "kernel_size": [(5, 5), (3, 3), (3, 3)],
            "filters": [32, 32, 32],
            "hiddens": [10, 10, 10],
            "dropout_rate": [0.0, 0.0, 0.0],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },

        # ROBUSTNESS TESTS: testing a few different architectures

        "exp061": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005 * 10.,
            "kernel_size": [(5, 5), (3, 3), (3, 3)],
            "filters": [32, 32, 32],
            "hiddens": [10, 10, 10],
            "dropout_rate": [0.0, 0.0, 0.0],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },
        "exp062": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005 / 2.,
            "kernel_size": [(5, 5), (3, 3), (3, 3)],
            "filters": [32, 32, 32],
            "hiddens": [10, 10, 10],
            "dropout_rate": [0.0, 0.0, 0.0],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },
        "exp063": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,
            "kernel_size": [(5, 5), (3, 3), (3, 3)],
            "filters": [32, 32, 32],
            "hiddens": [10,],
            "dropout_rate": [0.0,],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },
        "exp064": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,
            "kernel_size": [(5, 5), (3, 3), (3, 3)],
            "filters": [32, 32, 32],
            "hiddens": [100,],
            "dropout_rate": [0.0,],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },
        "exp065": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,  # 0.00001
            "kernel_size": [(5, 5), (3, 3), (3, 3)],
            "filters": [32, 32, 32],
            "hiddens": [],
            "dropout_rate": [],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },
        "exp066": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,  # 0.00001
            "kernel_size": [(5, 5), (3, 3)],
            "filters": [64, 64],
            "hiddens": [10, 10, 10],
            "dropout_rate": [0.0, 0.0, 0.0],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },
        "exp067": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,  # 0.00001
            "kernel_size": [(5, 5), (3, 3), (3, 3), (3, 3)],
            "filters": [32, 32, 32, 32],
            "hiddens": [10, 10, 10],
            "dropout_rate": [0.0, 0.0, 0.0],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },
        "exp068": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,  # 0.00001
            "kernel_size": [(5, 5), (3, 3), (3, 3), (3, 3)],
            "filters": [64, 64, 64],
            "hiddens": [10, 10, 10],
            "dropout_rate": [0.0, 0.0, 0.0],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },

        # -----------------------------------------------------------------
        # two dense layers after convolutional layers, plus other tweaks
        "exp081": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00005,  # 0.00001
            "kernel_size": [(5, 5), (3, 3)],
            "filters": [32, 32],
            "hiddens": [25, 25],
            "dropout_rate": [0.0, 0.0],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 10,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },

        # one dense layer after convolutional layers, plus other tweaks
        "exp082": {
            "save_model": True,
            "ssp": "370",  # [options: '126' or '370']
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "target_temp": None,
            "target_temps": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_train_val_test": (7, 2, 1),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1970, 2100),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "remove_poles": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]
            "target_region": None,  # ("ipcc_wna",),
            "land_only": False,
            "cumulative_history": False,
            "cumulative_history_only": False,
            "cumulative_sum": False,

            "final_year_of_obs": 2023,
            "rolling_mean_len": 5,
            "fit_start_year": 1970,

            "transfer_patience": 50,
            "transfer_min_delta": 0.25,
            "transfer_max_year": 2100,
            "transfer_temp_vec": (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),

            "architecture": "cnn",
            "depthwise": False,
            "learning_rate": 0.00001,  # 0.00001
            "kernel_size": [(5, 5), (3, 3), (3, 3)],
            "filters": [32, 32, 32],
            "hiddens": [25,],
            "dropout_rate": [0.0,],
            "ridge_param": [0.0,],

            "network_type": 'shash2',
            "penultimate_hiddens": 25,
            "normalizer_index": None,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [44, 55, 66],  # [11, 22, 33],
            "activation": ["relu", "relu", "tanh"],
            "n_epochs": 1_000,
            "patience": 10,
        },

    }

    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name

    return exp_dict
