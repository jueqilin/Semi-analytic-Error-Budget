# Semi-analytic-Error-Budget

A semi-analytic simulator for the Pyramid Wavefront Sensor (P-WFS) Error Budget, based on the theoretical framework presented in *Agapito et al. 2019* ("Semianalytical error budget for adaptive optics systems with pyramid wavefront sensors").

## Overview

This code computes the total residual variance of an Adaptive Optics (AO) system by evaluating its four main components:
1. **Fitting Error**
2. **Temporal Error** (Atmospheric turbulence + Windshake)
3. **Aliasing Error**
4. **Measurement / Noise Error**

The project has been modularized to separate core mathematical functions, visualization tools, and execution scripts, and it is fully configurable via YAML files.

## Project Structure

* **`src/Functions.py`**: The core library containing all mathematical and physical definitions (transfer functions, PSD calculations, optical gain interpolation, variance integrations, etc.).
* **`src/plots.py`**: Visualization library containing functions to generate PSD plots, transfer function plots, and closed-loop summary displays.
* **`src/fits_file`**: Directory with data files.
* **`examples/plot_psds.py`**: A standalone script dedicated to performing and plotting the spectral analysis (PSDs) in closed loop for specific Zernike modes.
* **`examples/verify_aliasing_energy.py`**: A validation script that checks the aliasing energy conservation by comparing the variance calculated from slopes vs. the numerical integral of the SA PSD.
* **`examples/main_saeb.py`**: Command-line entrypoint to run the error budget simulation with a YAML file.
* **`Total_variance.py`**: A script designed to run the error budget simulation.

### Configuration Files


*These files control telescope properties, atmospheric conditions (seeing, wind speed, $r_0$, $L_0$), WFS characteristics, paths to FITS files (reconstruction matrices, optical gains), and AO loop parameters.*

## Dependencies

The code relies on several standard scientific Python libraries. Ensure you have the following installed:
* `numpy`
* `scipy`
* `matplotlib`
* `astropy` (for reading `.fits` files)
* `pyyaml` (for parsing `.yaml` configurations)
* `sympy` (for symbolic transfer function derivations)
* `arte` (for atmospheric and turbulence PSD generation)

## How to Run

1. **Configure your simulation**: Edit `params_mod4.yaml` (or your chosen config file) to set up the telescope, atmosphere, and file paths. Make sure your `.fits` files are correctly located in the `src/file_fits/...` directories.
2. **Run a basic test**:
   ```bash
   python Total_Variance.py
3. **Run a full-scale simulation (e.g., 4000 modes)**:
   ```bash
   python script/main_saeb.py run params_mod4_4000modes.yaml
4. **Plot specific PSDs**:
   ```bash
   python plot_psds.py
5. **Verify Aliasing**:
   ```bash
    python verify_aliasing_energy.py
