#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

== Optimize Control Gain ==


Created on 2026-04-15 14:46

@author: Jueqi Lin
"""

import os
import numpy as np
import control as ct

DEFAULT_SRC_PATH = os.path.dirname(__file__)

from src.Functions import (
    DEFAULT_ALIASING_ALPHA,
    load_parameters,
    turbulence_psd,
    temporal_variance,
    aliasing_variance,
    measure_variance,
    build_transfer_function,
    compute_andes_optical_gain,
    funct_d2,
    radial_order_from_n_modes,
)

def optimize_control_gain(param_dir = 'params_mod4.yaml'):
    # 1. Load parameters
    param = load_parameters(os.path.join(DEFAULT_SRC_PATH, param_dir))
    print("Parameters loaded successfully.")
    
    n_actuators = param['control']['n_modes']
    D = param['telescope']['telescope_diam']
    aperture_radius = D / 2.0
    aperture_center = [0, 0, 0]

    L0 = param['atmosphere']['outer_scale']
    layers_altitude = 0.0
    wind_direction = 0.0
    wind_speed = param['atmosphere']['wind_speed']
    seeing = param['atmosphere']['seeing']
    r0 = 0.98 * 500 / seeing

    F_excess_noise = np.sqrt(param['wavefront_sensor']['value_for_F_excess_noise'])
    sky_background = param['wavefront_sensor']['sky_backgr']
    dark_current = param['wavefront_sensor']['dark_curr']
    readout_noise = param['wavefront_sensor']['noise_readout']

    file_path_R1 = param['data']['reconstruction_matrix']
    file_mod0 = param['data']['optical_gain_models'][0]
    file_mod4 = param['data']['optical_gain_models'][1]
    sigma_slopes_path = param['data']['sigma_slopes']

    d1 = param['plant']['d_1']
    d3 = param['plant']['d_3']
    n1 = param['plant']['n_1']
    n2 = param['plant']['n_2']
    n3 = param['plant']['n_3']

    t_0 = param['control']['sampling_time']
    frame_rate = 1.0 / t_0
    temporal_freqs = np.logspace(-3, np.log10(frame_rate / 2.0), 1000)
    omega = 2 * np.pi * temporal_freqs

    spatial_freqs = np.logspace(-4, 4, 100)

    T_tot = param['control']['total_delay']
    modulation_radius = param['wavefront_sensor']['modulation_radius']
    maximum_radial_order = radial_order_from_n_modes(n_actuators)
    alpha_ = DEFAULT_ALIASING_ALPHA

    phot_flux = float(param['guide_star']['flux_photons'])
    magnitude = param['guide_star']['magn']
    n_subapert = param['wavefront_sensor']['number_of_sub']
    collecting_area = param['telescope']['collect_area']
    x_pixel = param['control']['slope_computer_weights']

    gain_value = param['control'].get('gain_value')
    if gain_value is not None:
        gain_array = np.full(n_actuators, float(np.asarray(gain_value).ravel()[0]))
    else:
        gain_array = np.full(n_actuators, float(param['control']['gain_min']))

    d2 = funct_d2(T_tot)
    
    # 2. Generate Atmospheric Input
    PSD_atmosf = turbulence_psd(0, 0, aperture_radius, aperture_center,
                                r0, L0, layers_altitude,
                                wind_speed, wind_direction,
                                spatial_freqs, temporal_freqs,
                                n_modes=n_actuators)
    PSD_vibration_zeros = np.zeros_like(PSD_atmosf)

    # 3. Build Transfer Functions
    plant_num = np.polymul(np.polymul(np.asarray(n1), np.asarray(n2)), np.asarray(n3))
    plant_den = np.polymul(np.polymul(np.asarray(d1), np.asarray(d2)), np.asarray(d3))
    H_r, H_n = build_transfer_function(
        omega,
        t_0,
        n_actuators,
        plant_num,
        plant_den,
        gain=gain_array,
    )

    # 4. Compute Optical Gain (needed for aliasing)
    c_optg = compute_andes_optical_gain(file_mod0=file_mod0,
                                        file_mod4=file_mod4,
                                        seeing=seeing,
                                        modulation_radius=modulation_radius,
                                        actuators_number=n_actuators)
    
    # 5. Compute error budget components
    _, variance_vibr_CL, PSD_out_temp, PSD_in_temp = temporal_variance(
        PSD_atmosf, PSD_vibration_zeros, H_r, n_actuators, omega
    )
    
    _, variance_alias_CL, PSD_out_alias, PSD_in_alias = aliasing_variance(
        transf_funct=H_n,
        actuators_number=n_actuators,
        omega_temp_freq_interval=omega,
        c_optg=c_optg,
        alpha=alpha_,
        telescope_diameter=D,
        seeing=seeing,
        modulation_radius=modulation_radius,
        windspeed=wind_speed,
        maximum_radial_order_corrected=maximum_radial_order,
        file_path_matrix_R=file_path_R1,
        file_path_sigma_slopes=sigma_slopes_path
    )

    _, variance_meas_CL, PSD_out_meas, PSD_in_meas = measure_variance(
        F_excess=F_excess_noise,
        pixel_pos=x_pixel,
        sky_bkg=sky_background,
        dark_curr=dark_current,
        read_out_noise=readout_noise,
        photon_flux=phot_flux,
        telescope_diameter=D,
        frame_rate=frame_rate,
        magnitudo=magnitude,
        n_subaperture=n_subapert,
        collecting_area=collecting_area,
        file_path_matrix_R=file_path_R1,
        omega_temp_freq_interval=omega,
        transf_funct=H_n,
        actuators_number=n_actuators,
        c_optg=c_optg
    )
    
    # 6. Cost function for optimization (minimize total variance and ensure stability)
    total_variance = variance_vibr_CL + variance_alias_CL + variance_meas_CL
    
    
    
    return optimal_gain

def compute_penalty_term(close_loop_function_poly,t_0):
    
        
    stability_margin = np.min(np.abs(1 - H_r))  # Ensure stability (H_r should not approach 1)
    
    return cost_value