# Programming Project Cosmology

## **`help_functions.py`**
This file contains the functions for luminosity distance, angular distance and other various function from the task, namely eq. (1) - (5). This also contains the function to make an equally large new dataset from a given dataset by sampling, used for bootstrapping.

The curves to fit to the data are also defined in this file. For joint parameter estimation, the positions of the SNIa points are also extracted as the SNIa data has the apparent magnitude, instead of the distance modulus. This is taken into account in the function ```curve```. For individual estimation, different curves are defined for the SNIa data and the GRB and galaxy cluster data, namely ```curve_SNIa``` and ```curve_GRB_clusters```.

A function for getting the cofidence interval is also defined.

```astropy``` is used to convert to different units!

## **`read_data.py`**
This file contains the functions to get the (m-M) vs. z data from the three datasets. To get the data from SNIa and galaxy clusters dataset, the initial guess for the parameters isn't used. For the GRB data, the $L((1+z)E_{peak})$ calibration detailed in the task is done in the function ```mod_z_GRB```.

The function to plot the original (m-M) vs. z data is also defined in this file.

## **`estimate.py`**
In this file, the function ```param_est``` estimates the cosmological parameters using ```curve_fit```, given some initial guess for the parameters. This function is then run $N$ times. To get a value for the cosmological parameters.

The parameters can be estimated jointly by fitting one curve to the datapoints from the three datasets (bootstrapping using all the datapoints), or individually for the three datasets (bootstrapping for the individual datasets). This given by the boolean $combined$.

The boolean $plot$ is used to make corner plots. This is done using the ```corner``` python package.

To run the code, the following parameters need to be selected
```python
H0 = 70                 # Hubble constant
Omega_M = 0.3           # Matter density
Omega_lambda = 0.7      # Dark enrgy density
w = -1                  # Dark energy EOS
M = -18.3               # Absolute luminosity of SNIa

N_runs = 1000           # Number of runs
combined=True           # Whether to do joint estimation
plot=True               # Whether to make corner plots
```

The results are saved in text files with the naming convetion:
``estimate_{type}_Nruns_{N_runs}.txt``
with `type` one of the following:
``combined, SNIa, GRB, clusters``

Similary, the corner plots are saved as:
``param_est_{type}_Nruns_{N_runs}.png``