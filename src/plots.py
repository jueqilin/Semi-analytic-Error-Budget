#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:11:30 2026

@author: greta
"""

# pylint: disable=C


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.io import fits                                                    

from src.Functions import total_variance
from src.Functions import compute_andes_optical_gain
#from src.Functions import compute_soul_optical_gain
from src.Functions import extract_propagation_coefficients
from src.Functions import PSD_aliasing
from src.Functions import double_interpolation_sigma_slope
from src.Functions import read_sigma_slopes

###########
from src.Functions import fitting_variance
from src.Functions import temporal_variance
from src.Functions import aliasing_variance
from src.Functions import measure_variance
from src.Functions import build_transfer_function
from src.Functions import interpolate_and_normalize_psd
from src.Functions import align_psd_modes

##########


# Function to compute the total residual variance for a set of gain values,
# by combining fitting, temporal, aliasing and measurement error contributions 
  
def variance_total_for_test(number_of_actuators, gain_values, omega_temp_freq_interval, t_freqs, f,
                            t_0, num1, num2, num3, den1, den2, den3, telescope_diameter, fried_parameter,
                            excess_noise_factor, sky_background, dark_current, readout_noise,
                            photon_flux, frame_rate, magnitude, n_subaperture, collecting_area,
                            slope_computer_weights, fitting_coeff, alpha, seeing, modulation_radius,
                            wind_speed, maximum_radial_order_corrected, reconstruction_matrix_path,
                            optical_gain_models, psd_turbulence, psd_windshake, sigma_slopes_path):
    
                                                            
    tot_variance = np.zeros_like(gain_values, dtype=float)
    
    for i in range(len(gain_values)):
        
        g = gain_values[i]
        gain_val = np.array([g])                                               

###########################       
        
        H_r_temp = build_transfer_function(gain_val, omega_temp_freq_interval, t_0, number_of_actuators, num1, num2, num3, den1, den2, den3,"H_r")
        H_n_meas = build_transfer_function(gain_val, omega_temp_freq_interval, t_0, number_of_actuators, num1, num2, num3, den1, den2, den3,"H_n")
        H_n_alias = build_transfer_function(gain_val, omega_temp_freq_interval, t_0, number_of_actuators, num1, num2, num3, den1, den2, den3,"H_n")
        
        
        variance_fit = fitting_variance(fitting_coeff, number_of_actuators, telescope_diameter, fried_parameter)
        
         
        if np.array_equal(t_freqs, f): 
            
            variance_temporal,_ , _ = temporal_variance(psd_turbulence, psd_windshake, H_r_temp, number_of_actuators,
                                                              omega_temp_freq_interval)

        else: 
            
            PSD_wind_vib_interp_norm = interpolate_and_normalize_psd(t_freqs, f, psd_windshake, number_of_actuators)
            variance_temporal,_ , _ = temporal_variance(psd_turbulence, PSD_wind_vib_interp_norm,
                                                            H_r_temp, number_of_actuators, omega_temp_freq_interval)

        
        
        
        variance_aliasing, _, _ = aliasing_variance(H_n_alias, number_of_actuators, omega_temp_freq_interval, 
                                                    alpha, telescope_diameter, seeing, modulation_radius, wind_speed,
                                                    maximum_radial_order_corrected, reconstruction_matrix_path, gain_val,
                                                    sigma_slopes_path)
        
        
        variance_measurement, _, _ = measure_variance(excess_noise_factor, slope_computer_weights,
                                   sky_background, dark_current, readout_noise,
                                   photon_flux, telescope_diameter,
                                   frame_rate, magnitude, n_subaperture,
                                   collecting_area, reconstruction_matrix_path,
                                                       omega_temp_freq_interval, H_n_meas, number_of_actuators)
        
      
        tot_variance[i] = total_variance(np.real(variance_fit), np.real(variance_temporal), 
                                         np.real(variance_measurement), np.real(variance_aliasing))            
    
    return tot_variance


# Function to plot the total residual variance of the system as a function 
# of the gain, considering only the first mode.

def plot_total_variance_mode_0(gain_min, gain_max, omega_temp_freq_interval, t_freqs, f, t_0, num1, num2, 
                               num3, den1, den2, den3, telescope_diameter, fried_parameter, excess_noise_factor,
                               sky_background, dark_current, readout_noise, photon_flux, frame_rate, magnitude,
                               n_subaperture, collecting_area, slope_computer_weights, fitting_coeff, alpha, seeing,
                               modulation_radius, wind_speed, maximum_radial_order_corrected,
                               reconstruction_matrix_path, optical_gain_models, psd_turbulence, psd_windshake,
                               sigma_slopes_path):

       
    print ('TEST') 
     
    actuators_number = 1                                                  
     
    gain_value = np.arange(gain_min, gain_max, 0.1)
    variance_total = variance_total_for_test(actuators_number, gain_value, omega_temp_freq_interval, t_freqs, f,
                                             t_0, num1, num2, num3, den1, den2, den3, telescope_diameter, 
                                             fried_parameter, excess_noise_factor, sky_background, dark_current,
                                             readout_noise, photon_flux, frame_rate, magnitude, n_subaperture,
                                             collecting_area, slope_computer_weights, fitting_coeff, alpha, seeing,
                                             modulation_radius, wind_speed, maximum_radial_order_corrected,
                                             reconstruction_matrix_path, optical_gain_models, psd_turbulence,
                                             psd_windshake, sigma_slopes_path)
        
    plt.plot(gain_value, variance_total, marker='o')    
    plt.xlabel('Gain')
    plt.ylabel('Total variance')
    plt.title('Total variance as a function of the gain')
    plt.grid()
    plt.show()
       
        
# Defines a function that allows, when needed, to plot PSD_in, PSD_out, and the transfer 
# function for the variances (temp, alias, meas)

def plot(f, H_r_t, H_n_m, H_n_a, PSD_in_t, PSD_out_t, PSD_in_m, PSD_out_m, PSD_in_a, PSD_out_a):
    
    PSD_in = [PSD_in_t, PSD_in_m, PSD_in_a]                  
    PSD_out = [PSD_out_t, PSD_out_m, PSD_out_a]              
    H = [H_r_t, H_n_m, H_n_a]
       
    labels_PSD = ["temp", "meas","alias"]
    labels_H = ["r", "n", "n"]
       
    for i in range(len(PSD_in)):                                                           
          
        plt.loglog(f, PSD_in[i][0, :], label=f"PSD_in_{labels_PSD[i]} (mode 0)")       
        plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels_PSD[i]} (mode 0)")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("PSD")
        plt.title(f"PSD {labels_PSD[i]} (modo 0)")
        plt.legend()
        plt.grid()
        plt.show()
           
        plt.loglog(f, np.abs(H[i][0, :])**2)     
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel(f"|H_{labels_H[i]}|^2")
        plt.title(f"Transfert function H_{labels_H[i]} (mode 0)")
        plt.grid()
        plt.show()
       
        
# Function to plot the three output PSDs (temporal, measurement and aliasing)
# for mode 0 on the same graph.
       
def plot_all_PSD(f, PSD_out_t, PSD_out_m, PSD_out_a):

    PSD_out = [PSD_out_t, PSD_out_m, PSD_out_a]

    labels = ["temp", "meas", "alias"]

    for i in range(len(PSD_out)):

        plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels[i]} (mode 0)")

    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("PSD")
    plt.title("PSDs – mode 0")
    plt.legend()
    plt.grid()

    plt.show()


# Function to print a compact closed-loop summary and generate a basic set of plots:
# (1) variance vs mode, (2) total CL PSD, (3) CL PSD for selected modes,
# (4) transfer functions for selected modes.

def summary_display(var_fit_modes, var_temp_modes, var_alias_modes, var_meas_modes,
                    PSD_out_temp, PSD_out_alias, PSD_out_meas, frequencies,
                    H_r_temp, H_n_meas, PSD_input_atmos=None, PSD_input_wind=None,
                    modes_to_plot=None):

    frequencies = np.asarray(frequencies, dtype=float).ravel()
    n_frequencies = frequencies.size

    if n_frequencies == 0:
        raise ValueError("Frequency vector must not be empty")

    def _as_mode_frequency_matrix(array_in, array_name):
        array_out = np.asarray(array_in)

        if array_out.ndim != 2:
            raise ValueError(f"{array_name} must be a 2D array")

        if array_out.shape[1] == n_frequencies:
            return array_out

        if array_out.shape[0] == n_frequencies:
            return array_out.T

        raise ValueError(
            f"{array_name} has shape {array_out.shape}; expected (N_modes, N_frequencies) "
            f"or (N_frequencies, N_modes) with N_frequencies={n_frequencies}"
        )

    PSD_out_temp = _as_mode_frequency_matrix(PSD_out_temp, "PSD_out_temp")
    PSD_out_alias = _as_mode_frequency_matrix(PSD_out_alias, "PSD_out_alias")
    PSD_out_meas = _as_mode_frequency_matrix(PSD_out_meas, "PSD_out_meas")
    H_r_temp = _as_mode_frequency_matrix(H_r_temp, "H_r_temp")
    H_n_meas = _as_mode_frequency_matrix(H_n_meas, "H_n_meas")

    PSD_out_temp = np.real_if_close(PSD_out_temp, tol=1000)
    PSD_out_alias = np.real_if_close(PSD_out_alias, tol=1000)
    PSD_out_meas = np.real_if_close(PSD_out_meas, tol=1000)

    if np.iscomplexobj(PSD_out_temp):
        PSD_out_temp = np.real(PSD_out_temp)
    if np.iscomplexobj(PSD_out_alias):
        PSD_out_alias = np.real(PSD_out_alias)
    if np.iscomplexobj(PSD_out_meas):
        PSD_out_meas = np.real(PSD_out_meas)

    n_modes = PSD_out_temp.shape[0]

    if n_modes == 0:
        raise ValueError("Empty modal vectors are not allowed")

    def _as_mode_vector(vector_in, vector_name):
        vector_out = np.real_if_close(np.asarray(vector_in), tol=1000).ravel()

        if np.iscomplexobj(vector_out):
            vector_out = np.real(vector_out)

        vector_out = vector_out.astype(float)

        if vector_out.size == 1:
            return np.full(n_modes, float(vector_out.item()))

        if vector_out.size == n_modes:
            return vector_out

        raise ValueError(
            f"{vector_name} length is {vector_out.size}, expected 1 or N_modes={n_modes}"
        )

    var_fit_modes = _as_mode_vector(var_fit_modes, "var_fit_modes")
    var_temp_modes = _as_mode_vector(var_temp_modes, "var_temp_modes")
    var_alias_modes = _as_mode_vector(var_alias_modes, "var_alias_modes")
    var_meas_modes = _as_mode_vector(var_meas_modes, "var_meas_modes")

    vectors = [var_fit_modes, var_temp_modes, var_alias_modes, var_meas_modes]

    for vec in vectors:
        if vec.size != n_modes:
            raise ValueError("All variance vectors must have length N_modes")

    expected_shape = (n_modes, frequencies.size)

    arrays_2d = [PSD_out_temp, PSD_out_alias, PSD_out_meas, H_r_temp, H_n_meas]

    for arr in arrays_2d:
        if arr.shape != expected_shape:
            raise ValueError("PSD and transfer-function arrays must have shape (N_modes, N_frequencies)")

    H_n_alias = H_n_meas

    PSD_input_atmos_proc = None
    PSD_input_wind_proc = None

    if PSD_input_atmos is not None:
        try:
            PSD_input_atmos_proc = np.asarray(PSD_input_atmos, dtype=float)
            PSD_input_atmos_proc = _as_mode_frequency_matrix(PSD_input_atmos_proc, "PSD_input_atmos")
            PSD_input_atmos_proc = align_psd_modes(PSD_input_atmos_proc, n_modes)
            PSD_input_atmos_proc = np.real_if_close(PSD_input_atmos_proc, tol=1000)
            if np.iscomplexobj(PSD_input_atmos_proc):
                PSD_input_atmos_proc = np.real(PSD_input_atmos_proc)
        except Exception:
            PSD_input_atmos_proc = None

    if PSD_input_wind is not None:
        try:
            PSD_input_wind_proc = np.asarray(PSD_input_wind, dtype=float)
            PSD_input_wind_proc = _as_mode_frequency_matrix(PSD_input_wind_proc, "PSD_input_wind")
            PSD_input_wind_proc = align_psd_modes(PSD_input_wind_proc, n_modes)
            PSD_input_wind_proc = np.real_if_close(PSD_input_wind_proc, tol=1000)
            if np.iscomplexobj(PSD_input_wind_proc):
                PSD_input_wind_proc = np.real(PSD_input_wind_proc)
        except Exception:
            PSD_input_wind_proc = None

    var_total_modes = var_fit_modes + var_temp_modes + var_alias_modes + var_meas_modes

    fit_total = np.sum(var_fit_modes)
    temp_total = np.sum(var_temp_modes)
    alias_total = np.sum(var_alias_modes)
    meas_total = np.sum(var_meas_modes)
    total_variance_sum = np.sum(var_total_modes)

    print("\n===== CLOSED-LOOP SUMMARY =====")
    print(f"N_modes: {n_modes}")
    print(f"N_frequencies: {frequencies.size}")
    print(f"Fitting total variance [nm^2]:      {fit_total:.6e}")
    print(f"Temporal total variance [nm^2]:     {temp_total:.6e}")
    print(f"Aliasing total variance [nm^2]:     {alias_total:.6e}")
    print(f"Measurement total variance [nm^2]:  {meas_total:.6e}")
    print(f"Total output variance [nm^2]:       {total_variance_sum:.6e}")

    if modes_to_plot is None:
        modes_to_plot = [0, n_modes // 2, n_modes - 1]

    mode_indices = np.array(modes_to_plot, dtype=int)
    mode_indices = mode_indices[(mode_indices >= 0) & (mode_indices < n_modes)]
    mode_indices = np.unique(mode_indices)

    if mode_indices.size == 0:
        mode_indices = np.array([0], dtype=int)

    mode_axis = np.arange(n_modes)+1

    plt.figure()
    plt.plot(mode_axis, var_total_modes, label="total")
    plt.plot(mode_axis, var_fit_modes, label="fitting")
    plt.plot(mode_axis, var_temp_modes, label="temporal")
    plt.plot(mode_axis, var_alias_modes, label="aliasing")
    plt.plot(mode_axis, var_meas_modes, label="measurement")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mode index")
    plt.ylabel("Variance [nm^2]")
    plt.title("Variance vs mode")
    plt.grid()
    plt.legend()
    plt.show()

    PSD_cl_total = PSD_out_temp + PSD_out_alias + PSD_out_meas
    PSD_cl_total_all_modes = np.sum(PSD_cl_total, axis=0)

    plt.figure()
    plt.loglog(frequencies, PSD_cl_total_all_modes)
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("PSD [nm^2/Hz]")
    plt.title("Closed-loop total PSD")
    plt.grid()
    plt.show()

    plt.figure()
    for mode in mode_indices:
        plt.loglog(frequencies, PSD_cl_total[mode, :], label=f"mode {mode}")
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("PSD [nm^2/Hz]")
    plt.title("Closed-loop PSD for selected modes")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    for mode in mode_indices:
        plt.loglog(frequencies, np.abs(H_r_temp[mode, :])**2, label=f"|H_r|² mode {mode}")
        plt.loglog(frequencies, np.abs(H_n_meas[mode, :])**2, '--', label=f"|H_n|² mode {mode}")
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("Magnitude squared")
    plt.title("Transfer functions for selected modes")
    plt.grid()
    plt.legend()
    plt.show()

    if PSD_input_atmos_proc is not None or PSD_input_wind_proc is not None:
        PSD_input_total_all_modes = np.zeros(frequencies.size)
        if PSD_input_atmos_proc is not None:
            PSD_input_total_all_modes += np.sum(PSD_input_atmos_proc, axis=0)
        if PSD_input_wind_proc is not None:
            PSD_input_total_all_modes += np.sum(PSD_input_wind_proc, axis=0)

        plt.figure()
        if PSD_input_atmos_proc is not None:
            plt.loglog(frequencies, np.sum(PSD_input_atmos_proc, axis=0), label="Atmospheric")
        if PSD_input_wind_proc is not None:
            plt.loglog(frequencies, np.sum(PSD_input_wind_proc, axis=0), label="Windshake")
        plt.loglog(frequencies, PSD_input_total_all_modes, 'k--', linewidth=2, label="Total input")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("PSD [nm^2/Hz]")
        plt.title("Input PSD (all modes)")
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure()
        if PSD_input_atmos_proc is not None:
            for mode in mode_indices:
                plt.loglog(frequencies, PSD_input_atmos_proc[mode, :], label=f"Atmos mode {mode}")
        if PSD_input_wind_proc is not None:
            for mode in mode_indices:
                plt.loglog(frequencies, PSD_input_wind_proc[mode, :], '--', label=f"Wind mode {mode}")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("PSD [nm^2/Hz]")
        plt.title("Input PSD (selected modes)")
        plt.grid()
        plt.legend()
        plt.show()

    return {
        "var_total_modes": var_total_modes,
        "totals": {
            "fit": fit_total,
            "temp": temp_total,
            "alias": alias_total,
            "meas": meas_total,
            "total": total_variance_sum
        },
        "modes_plotted": mode_indices,
        "H_n_alias": H_n_alias
    }


# Function to verify the consistency of the aliasing variance calculation.
# It computes the analytical aliasing variance (summing the selected modes)
# and compares it with the variance obtained by integrating the corresponding 
# aliasing PSD. 
# The function also print the variance of the first mode alone in both cases.
    
def check(reconstruction_matrix_path, telescope_diameter, seeing, modulation_radius,
                    actuators_number, alpha, omega_temp_freq_interval, wind_speed,
                    maximum_radial_order_corrected, optical_gain_models, sigma_slopes_path,
          system="ANDES"):

    if system == "ANDES":
                c_optg = compute_andes_optical_gain(optical_gain_models[0], optical_gain_models[1],
                                                                                        seeing, modulation_radius)
    # TODO not supported yet
    #elif system == "SOUL":
    #    gain = compute_soul_optical_gain(file_optg, mod_modes, binning, magnitude)
    
    p_coefficient = extract_propagation_coefficients(reconstruction_matrix_path)
    
    if p_coefficient is None:   
                                              
        raise RuntimeError("Propagation coefficients not loaded") 
   
    print("Propagation coefficients loaded successfully.")
    
    data_slopes = read_sigma_slopes(sigma_slopes_path)
    seeing_vals = data_slopes[0,0,:]                                           
  
    modal_radius_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0]) 
    
    
    
    sigma_slope_alias = double_interpolation_sigma_slope(modal_radius_vals, seeing_vals, data_slopes, 
                                                         modulation_radius, seeing)
  
    sigma_alias_2_two_modes = 0.0
    
    for i in range (actuators_number):
        
        sigma_alias_2 = p_coefficient[i] * (sigma_slope_alias ** 2) / c_optg ** 2
        
        sigma_alias_2_two_modes += sigma_alias_2
        
    print("ALIASING VARIANCE:", sigma_alias_2_two_modes)
    
    PSD_al = PSD_aliasing (actuators_number, omega_temp_freq_interval, alpha, 
                           telescope_diameter, seeing, modulation_radius, wind_speed,
                           maximum_radial_order_corrected,
                           reconstruction_matrix_path, c_optg, sigma_slopes_path)
    
    
    integral_per_mode = integrate.simpson(PSD_al, omega_temp_freq_interval)
    sigma_alias_2_PSD_total = np.sum(integral_per_mode)
    
    print("ALIASING VARIANCE FROM PSD:", sigma_alias_2_PSD_total)
    
    sigma_alias_2_one_mode = p_coefficient * (sigma_slope_alias ** 2) / c_optg ** 2
    
    print("ALIASING VARIANCE ONE MODE:", sigma_alias_2_one_mode[0])
    
    print("ALIASING VARIANCE FROM PSD ONE MODE:", integral_per_mode[0])
    
    
def plot_PSD_alias_mode_0(actuators_number, omega_temp_freq_interval, alpha, telescope_diameter,
                          seeing, modulation_radius, wind_speed, maximum_radial_order_corrected,
                          reconstruction_matrix_path, optical_gain_models, sigma_slopes_path,
                          system="ANDES"):
    
    with fits.open("src/file_fits/ANDES/modal_psd_aliasing.fits") as hdul:
        data = hdul[0].data # pylint: disable=no-member
        
        freq_hz = data[:, 0]
        mode_0 = data[:, 1]
        
        freq_rad_s = 2 * np.pi * freq_hz
        if system == "ANDES":
            c_optg = compute_andes_optical_gain(optical_gain_models[0], optical_gain_models[1],
                                                seeing, modulation_radius)
        # TODO not supported yet
        #elif system == "SOUL":
        #    gain = compute_soul_optical_gain(file_optg, mod_modes, binning, magnitude)
        PSD_aliasing_mode0_given = mode_0 / (c_optg ** 2 * 2 * np.pi)             
        
    PSD_alising_mine = PSD_aliasing (actuators_number, omega_temp_freq_interval, alpha,  
                                     telescope_diameter, seeing, modulation_radius, wind_speed,
                                     maximum_radial_order_corrected, reconstruction_matrix_path, c_optg,
                                     sigma_slopes_path)
    
    PSD_alising_mine_mode0 = PSD_alising_mine[0,:]
    
    plt.loglog(omega_temp_freq_interval, PSD_alising_mine_mode0, label="PSD alias mine mode 0")
    plt.loglog(freq_rad_s, PSD_aliasing_mode0_given, label="PSD alias data mode 0")
    

    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("PSD")  
    plt.legend()
    plt.grid()
    plt.show()

    # N = len(freq_rad_s)   # = 501

    # omega_cut = omega_temp_freq_interval[:N]
    # PSD_mine_cut = PSD_alising_mine_mode0[:N]

    # ratio = PSD_mine_cut / PSD_aliasing_mode0_given
    # plt.figure()

    # plt.semilogx(omega_cut, ratio)
    # plt.xlabel("Frequency [rad/s]")
    # plt.ylabel("Mine / Given")
    # plt.grid()
    # plt.title("PSD ratio")

    # plt.show()
    
    
    
    
    
  

    