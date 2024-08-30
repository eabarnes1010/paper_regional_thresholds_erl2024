## Overview
Code for the paper _Combining climate models and observations to predict the time remaining until regional warming thresholds are reached_ submitted to Environmental Research Letters.
by Elizabeth A. Barnes, Noah S. Diffenbaugh, and Sonia Seneviratne


## Tensorflow Code
***
This code was written in python 3.9.7, tensorflow 2.7.0, tensorflow-probability 0.15.0 and numpy 1.22.2.

### Python Environment
The following python environment was used to implement this code.
```
- conda create --name env-tfp-2.7-nometal python=3.9
- conda activate env-tfp-2.7-nometal
- conda install -c apple tensorflow-deps==2.7
- python -m pip install tensorflow-macos==2.7
- pip install tensorflow-probability==0.15 silence-tensorflow tqdm
- conda install numpy scipy pandas matplotlib seaborn statsmodels palettable progressbar2 tabulate icecream flake8 jupyterlab black isort jupyterlab_code_formatter xarray scikit-learn
- pip install cmocean cmasher
- pip install -U pytest
- conda install -c conda-forge cartopy dask netCDF4 geopandas pygeos regionmask
- pip install ipython-autotime
- pip install cmaps
- pip install shap
```

## Credits
***
This work is a collaborative effort between [Dr. Elizabeth A. Barnes](https://barnes.atmos.colostate.edu), [Dr. Noah Diffenbaugh](https://earth.stanford.edu/people/noah-diffenbaugh#gs.runods) and [Dr. Sonia Seneviratne](https://usys.ethz.ch/en/people/profile.sonia-seneviratne.html).

## Paper citation
_coming soon_

### License
This project is licensed under an MIT license.

MIT © [Elizabeth A. Barnes](https://github.com/eabarnes1010)




