#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

== Start Script for Semi-analytic Error Budget Optimization ==

Created on 2026-04-15 16:48

@author: Jueqi Lin
"""
# required control slycot matplotlib
import os
import numpy as np
import control as ct
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from src.control_plot import bodeplot_Hz

DEFAULT_SRC_PATH = os.path.dirname(__file__)
DEFAULT_FITTING_COEFF = 0.28

from src.Functions import (
    DEFAULT_ALIASING_ALPHA,
    load_parameters,
    turbulence_psd,
    fitting_variance,
    temporal_variance,
    aliasing_variance,
    measure_variance,
    build_transfer_function,
    compute_andes_optical_gain,
    funct_d2,
    radial_order_from_n_modes,
    prepare_single_mode_control_optimization,
    seeing_to_r0,
)

from src.control_utils import (
    control_CL_tf_margin,
    cost,
)

def initial(param_dir = 'params_mod4.yaml'):
    # 1. Load parameters
    # param = load_parameters(os.path.join(DEFAULT_SRC_PATH, param_dir))
    param = load_parameters(param_dir)
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
    r0 = seeing_to_r0(seeing)

    F_excess_noise = np.sqrt(param['wavefront_sensor']['value_for_F_excess_noise'])
    sky_background = param['wavefront_sensor']['sky_backgr']
    dark_current = param['wavefront_sensor']['dark_curr']
    readout_noise = param['wavefront_sensor']['noise_readout']

    file_path_R1 = param['data']['reconstruction_matrix']
    file_mod0 = param['data']['optical_gain_models'][0]
    file_mod4 = param['data']['optical_gain_models'][1]
    sigma_slopes_path = param['data']['sigma_slopes']

    d1 = param['plant']['d_1']      # WFS's den
    d3 = param['plant']['d_3']      # RTC's den
    n1 = param['plant']['n_1']
    n2 = param['plant']['n_2']      # ASM's num
    n3 = param['plant']['n_3']

    t_0 = param['control']['sampling_time']
    frame_rate = 1.0 / t_0
    temporal_freqs = np.logspace(-3, np.log10(frame_rate / 2.0), 1000)
    omega = 2 * np.pi * temporal_freqs

    spatial_freqs = np.logspace(-4, 4, 300)

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

    # d2 = funct_d2(T_tot)
    d2 = funct_d2(T_tot)
    
    # 2. Generate Atmospheric Input
    PSD_atmosf = turbulence_psd(0, 0, aperture_radius, aperture_center,
                                r0, L0, layers_altitude,
                                wind_speed, wind_direction,
                                spatial_freqs, temporal_freqs,
                                n_modes=n_actuators)
    PSD_vibration_zeros = np.zeros_like(PSD_atmosf)
    
    # 3. Build Transfer Functions for mode 0 (as an example)
    mode_index = 0    
    
    # The plant (WFS * Reconstructor+Delay * DM) is represented by its pre-multiplied polynomials
    plant_W_tf = ct.tf(n1, d1, t_0)
    plant_R_tf = ct.tf(n2, d2, t_0)
    plant_M_tf = ct.tf(n3, d3, t_0)
    plant_tf = ct.series(plant_W_tf, plant_R_tf, plant_M_tf)
    plant_num = plant_tf.num[mode_index][mode_index]
    plant_den = plant_tf.den[mode_index][mode_index]

    # 4. Compute Optical Gain (needed for aliasing)
    c_optg = compute_andes_optical_gain(file_mod0=file_mod0,
                                        file_mod4=file_mod4,
                                        seeing=seeing,
                                        modulation_radius=modulation_radius,
                                        actuators_number=n_actuators)
    
    # 5. initialize the optimization context for single mode control optimization
    
    # fitting variance
    static_fit_variance = fitting_variance(
        fitting_coeff=DEFAULT_FITTING_COEFF, 
        actuators_number=n_actuators, 
        telescope_diameter=D, 
        r0=r0)
    
    # obj_to_optimize is an instance of SingleModeControllerOptimizationContext
    obj_to_optimize = prepare_single_mode_control_optimization(
        mode_index=mode_index,
        omega_temp_freq_interval=omega,
        t_0=t_0,
        PSD_atmo_turb=PSD_atmosf,
        PSD_vibration=PSD_vibration_zeros,
        telescope_diameter=D,
        seeing=seeing,
        modulation_radius=modulation_radius,
        windspeed=wind_speed,
        maximum_radial_order_corrected=maximum_radial_order,
        c_optg=c_optg,
        F_excess=F_excess_noise,
        pixel_pos=x_pixel,
        sky_bkg=sky_background,
        dark_curr=dark_current,
        read_out_noise=readout_noise,
        photon_flux=phot_flux,
        frame_rate=frame_rate,
        magnitudo=magnitude,
        n_subaperture=n_subapert,
        collecting_area=collecting_area,
        file_path_matrix_R=file_path_R1,
        alpha=alpha_,
        file_path_sigma_slopes=sigma_slopes_path,
        static_fit_variance=static_fit_variance,
        num1=n1,
        num2=n2,
        num3=n3,
        den1=d1,
        den2=d2,
        den3=d3,
    )
    
    # generate the system with the initial controller (without optimization)   
    res_cost_no_opti, res_evaluate_no_opti, _, _, _, _, _ = cost(obj_to_optimize, 
        actuators_number=n_actuators,
        WFS_num=n1,
        WFS_den=d1,
        ASM_num=n2,
        ASM_den=d2,
        RTC_num=n3,
        RTC_den=d3, 
        gain = gain_array,
        controller_num = None, 
        controller_den = None)    
    
    # 6. Optimization
    # evaluate function just for one mode!!!  
    # build up the cost function to be optimized with dual_annealing method
    opti_cost_func = lambda x: cost(obj_to_optimize,
                                actuators_number=n_actuators,
                                WFS_num=n1,
                                WFS_den=d1,
                                ASM_num=n2,
                                ASM_den=d2,
                                RTC_num=n3,
                                RTC_den=d3, 
                                gain = x,
                                controller_num = None, 
                                controller_den = None)[0]  
    opti_bounds = [(0, 2) for _ in range(len(gain_array))]  # Example bounds for each gain parameter
    res_opti_dual_annealing = dual_annealing(opti_cost_func, opti_bounds, maxiter=100, seed=50)
    
    print()
    print()  
    print("=========before optimization============")
    print("plant num and den:", plant_num, plant_den)
    print("Total cost result:", res_cost_no_opti)
    print("Evaluation result (variance terms):", res_evaluate_no_opti[0].variance_terms)
    print("Initial Gain array:", gain_array)
    
    print()
    print() 
    print("=========Optimization============")
    print("Optimal gain array found:", res_opti_dual_annealing.x)
    
    res_cost_optimized, res_evaluate_optimized, stability_penalty, stability_margin_penalty, H_r_tf, H_n_tf, cl_peak_penalty= cost(obj_to_optimize, 
        actuators_number=n_actuators,
        WFS_num=n1,
        WFS_den=d1,
        ASM_num=n2,
        ASM_den=d2,
        RTC_num=n3,
        RTC_den=d3, 
        gain = res_opti_dual_annealing.x,
        controller_num = None, 
        controller_den = None)
    
    print()
    print()  
    print("============ After Optimization ===========") 
    print("Total cost result:", res_cost_optimized)
    print("Optimized evaluation result:", res_evaluate_optimized[0].variance_terms, res_evaluate_optimized[1].variance_terms)
    
    print("Optimized stability penalty:", stability_penalty)
    print("Optimized stability margin penalty:", stability_margin_penalty)
    print('Close_loop_peak_penalty of H_n:', cl_peak_penalty)
    
    print("Optimized controller num and den:", 
          res_evaluate_optimized[0].controller_num, 
          res_evaluate_optimized[0].controller_den)
    print('H_r_tf:',H_r_tf[0])
    print('H_n_tf:',H_n_tf[0])
    
    
    # 7. Plotting
    # bode figure for the transfer functions
    fig1, ax3 = bodeplot_Hz(
    transfer_functions_ct=H_n_tf[mode_index],
    omega_limits=[1e-5,frame_rate/2],
    omega_num=1000,
    labels="H_n ",
    title="Transfer function H_n & H_r")
    
    bodeplot_Hz(
    transfer_functions_ct=H_r_tf[mode_index],
    omega_limits=[1e-5,frame_rate/2],
    omega_num=1000,
    labels="H_r ",
    styles={'linestyle':'--'},
    fig=fig1,
    ax1=ax3[0],
    ax2=ax3[1])
    
    # compare psd
    gain_array_optimized = res_opti_dual_annealing.x
    H_r_optimized, H_n_optimized = build_transfer_function(
        omega,
        t_0,
        n_actuators,
        plant_num,
        plant_den,
        gain=gain_array_optimized,
    )
    
    _, variance_vibr_CL, PSD_out_temp, PSD_in_temp = temporal_variance(
        PSD_atmosf, PSD_vibration_zeros, H_r_optimized, n_actuators, omega
    )
    
    _, variance_alias_CL, PSD_out_alias, PSD_in_alias = aliasing_variance(
        transf_funct=H_n_optimized,
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
        transf_funct=H_n_optimized,
        actuators_number=n_actuators,
        c_optg=c_optg
    )
    
    PSD_in_temp = np.real(PSD_in_temp)
    PSD_in_alias = np.real(PSD_in_alias)
    PSD_in_meas = np.real(PSD_in_meas)
    PSD_out_temp = np.real(PSD_out_temp)
    PSD_out_alias = np.real(PSD_out_alias)
    PSD_out_meas = np.real(PSD_out_meas)
    
    plt.figure(figsize=(12, 7))
    freqs = temporal_freqs

    plot_inputs = True
    if plot_inputs:
        plt.loglog(freqs, PSD_in_temp[mode_index, :], label='Turbulence (Open Loop)',
                   color='tab:blue', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_alias[mode_index, :], label='Aliasing (Open Loop)',
                   color='tab:orange', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_meas[mode_index, :], label='Noise (Open Loop)',
                   color='tab:green', linestyle='--', alpha=0.6)

    plt.loglog(freqs, PSD_out_temp[mode_index, :], label='Turbulence Residual (Servo-lag)',
               color='tab:blue', linewidth=2.5)
    plt.loglog(freqs, PSD_out_alias[mode_index, :], label='Aliasing Residual',
               color='tab:orange', linewidth=2.5)
    plt.loglog(freqs, PSD_out_meas[mode_index, :], label='Noise Residual',
               color='tab:green', linewidth=2.5)

    psd_total = (PSD_out_temp[mode_index, :] +
                 PSD_out_alias[mode_index, :] +
                 PSD_out_meas[mode_index, :])

    plt.loglog(freqs, psd_total, label='Total PSD (Sum)',
               color='black', linewidth=3, linestyle='-.')

    plt.title(f"Spectral Analysis (PSD) in Closed Loop - Zernike Mode {mode_index}"
              f"\nLoop Gain = {gain_array_optimized[mode_index]:.2f}", fontsize=14)
    plt.xlabel("Temporal Frequency [Hz]", fontsize=12)
    plt.ylabel("Power Spectral Density [nm² / Hz]", fontsize=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend(loc='lower left', fontsize=11)

    plt.tight_layout()
 
    plt.show()
    
    return obj_to_optimize
 

if __name__ == "__main__":
    obj_to_optimize = initial(param_dir = 'params_mod4.yaml')
  