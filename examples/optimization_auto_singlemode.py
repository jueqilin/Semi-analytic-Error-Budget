#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

== Start Script for Semi-analytic Error Budget Optimization ==

NOTE: 1. ONLY FOR SINGLE MODE
      2. The controller type and parameters will be determined by the YAML file automatically:
         - controller type==1: integral controller, only optimize the gain value
         - controller type==2: polynomial controller, optimize the coefficients of the numerator and denominator polynomials
     
=========== content ===========

    1. Load parameters
    2. Generate Atmospheric Input
    3. Build Transfer Functions for mode 0 (as an example)
    4. Compute Optical Gain (needed for aliasing)
    5. Initialize the optimization context for single mode control optimization
    6. Optimization
    7. Plotting
        7.1 bode figure for the transfer functions
        7.2 Compare PSDs and plot
        7.3 Nyquist plot for the open-loop transfer function
        
===============================
        
Created on 2026-05-03 14:59

@author: Jueqi Lin
"""

import os
import numpy as np
import control as ct
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from src.control_plot import (
    bodeplot_Hz,
    set_psd_plot_title_text,
    plot_psds_single_mode,
    plot_nyquist)

DEFAULT_SRC_PATH = os.path.dirname(__file__)
DEFAULT_FITTING_COEFF = 0.28

from src.Functions import (
    DEFAULT_ALIASING_ALPHA,
    load_parameters,
    turbulence_psd,
    fitting_variance,
    build_transfer_function_single_mode,
    compute_optical_gain,
    funct_d2,
    radial_order_from_n_modes,
    prepare_single_mode_control_optimization,
    seeing_to_r0,
)

from src.control_utils import cost

def optmization_auto_singlemode(param_dir = 'params_mod4_polynomial.yaml'):
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
    
    controller_type = param['optimization'].get('ctrl_type')

    if controller_type is not None:
        if controller_type == 1:  # integral controller
            gain_value = param['control'].get('gain_value')
            if gain_value is not None:
                gain_array = np.full(n_actuators, float(np.asarray(gain_value).ravel()[0]))
                ctrl_num_array = None
                ctrl_den_array = None
            else:
                gain_array = np.full(n_actuators, float(param['control']['gain_min']))
                ctrl_num_array = None
                ctrl_den_array = None
        elif controller_type == 2: # polynomial controller
            ctrl_order = param['optimization'].get('order')
            if ctrl_order is not None:
                gain_array = None
                n_num_poly = ctrl_order[0] + 1
                n_den_poly = ctrl_order[1] + 1
                ctrl_num_array = np.zeros(n_num_poly, dtype=float)
                ctrl_num_array[0] = 1.0
                ctrl_den_array = np.zeros(n_den_poly, dtype=float)
                ctrl_den_array[0] = 1.0  
            else:
                raise ValueError("Provide controller's order (polynomial controller)")
        else:
            raise ValueError("Provide wrong 'ctrl_type'. Please check. ")
    else:
        raise ValueError("Provide 'ctrl_type' ")    

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
    c_optg = compute_optical_gain(file_mod0=file_mod0,
                                file_mod1=file_mod4,
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
    
    # 6. Optimization
    # evaluate function just for one mode!!!  
    # build up the cost function to be optimized with dual_annealing method
    n_iter_optimization = 100
    seed_optimization = 50
    
    if controller_type== 1:
        # generate the system with the initial controller (without optimization)
        res_cost_initial =  cost(obj_to_optimize, 
                            gain=gain_array[mode_index])    
    
        cost_function_value_no_opti = res_cost_initial['cost_function_value']
        evaluate_no_opti = res_cost_initial['evaluate_result']
        
        opti_cost_func = lambda x: cost(obj_to_optimize,
                                gain = x)['cost_function_value']
        
        gain_min = float(param['control']['gain_min'])
        gain_max = 1.0
        
        if gain_min >= gain_max:
            raise ValueError(f"gain_min ({gain_min}) must be less than gain_max ({gain_max})")

        x0_integral = np.array([gain_array[mode_index]])
        opti_bounds = [(gain_min, gain_max)] # Example bounds for each gain parameter
        res_opti_dual_annealing = dual_annealing(opti_cost_func, 
                                                opti_bounds,
                                                x0=x0_integral,
                                                maxiter=n_iter_optimization, 
                                                seed=seed_optimization)
        
        print()  
        print("=========before optimization============")
        print()
        print("plant num and den:", plant_num, plant_den)
        print("Cost function value:", cost_function_value_no_opti)
        print("Evaluation result (variance terms):", evaluate_no_opti.variance_terms)
        print("Initial Gain array:", gain_array)
    
        print() 
        print("=========Optimization============")
        print()
        print("Optimal gain array found:", res_opti_dual_annealing.x)
    
        # res_cost_optimized, res_evaluate_optimized, stability_penalty, stability_margin_penalty, H_r_tf, H_n_tf, H_n_peak_penalty
        res_cost_optimized  = cost(obj_to_optimize, 
                                    gain=res_opti_dual_annealing.x[0],
                                    controller_num=None, 
                                    controller_den=None)
        
        gain_optimized = res_opti_dual_annealing.x[0]
        print(type(gain_optimized))
        H_r_optimized, H_n_optimized = build_transfer_function_single_mode(
                                            omega,
                                            t_0,
                                            plant_num,
                                            plant_den,
                                            gain=gain_optimized)
        
        title_text = set_psd_plot_title_text(controller_type, 
                                         mode_index=mode_index,
                                         gain=gain_optimized) 
               
    elif controller_type == 2:
        # generate the system with the initial controller (without optimization)
        res_cost_initial =  cost(obj_to_optimize, 
                            controller_num=ctrl_num_array, 
                            controller_den=ctrl_den_array)    
    
        cost_function_value_no_opti = res_cost_initial['cost_function_value']
        evaluate_no_opti = res_cost_initial['evaluate_result']
        
        x0_poly = np.r_[ctrl_num_array, ctrl_den_array]
        opti_cost_func = lambda x: cost(obj_to_optimize,
                                gain=None,
                                controller_num = x[:n_num_poly], 
                                controller_den = x[n_num_poly:])['cost_function_value']
        opti_bounds = [(-10, 10) for _ in range(len(x0_poly))]
        res_opti_dual_annealing = dual_annealing(opti_cost_func, 
                                                 opti_bounds, 
                                                 x0=x0_poly, 
                                                 maxiter=n_iter_optimization, 
                                                 seed=seed_optimization)
    
        print()  
        print("=========before optimization============")
        print()
        print("plant num and den:", plant_num, plant_den)
        print("Cost function value:", cost_function_value_no_opti)
        print("Evaluation result (variance terms):", evaluate_no_opti.variance_terms)
        print("Initial Num:", ctrl_num_array)
        print("Initial Den:", ctrl_den_array)
        
        print() 
        print("=========Optimization============")
        print()
        print("Optimal controller num & den found:", res_opti_dual_annealing.x)
        
        ctrl_num_optimized = res_opti_dual_annealing.x[:n_num_poly]
        ctrl_den_optimized = res_opti_dual_annealing.x[n_num_poly:]
                                
        res_cost_optimized  = cost(obj_to_optimize, 
                                    gain = None,
                                    controller_num = ctrl_num_optimized, 
                                    controller_den = ctrl_den_optimized)
        H_r_optimized, H_n_optimized = build_transfer_function_single_mode(
                                            omega,
                                            t_0,
                                            plant_num,
                                            plant_den,
                                            controller_num=ctrl_num_optimized,
                                            controller_den=ctrl_den_optimized)
        title_text = set_psd_plot_title_text(controller_type, 
                                         mode_index=mode_index,
                                         ctrl_num=ctrl_num_optimized,
                                         ctrl_den=ctrl_den_optimized
                                         )
            
    cost_function_value_optimized = res_cost_optimized['cost_function_value']
    evaluate_optimized = res_cost_optimized['evaluate_result']
    stability_penalty, sm_penalty, H_n_tf_peak_penalty, H_r_tf_peak_penalty, gm_penalty = res_cost_optimized['penalty']
    H_n_tf_optimized = res_cost_optimized['H_n_tf']
    H_r_tf_optimized = res_cost_optimized['H_r_tf']
    H_ol_tf_optimized = res_cost_optimized['H_ol_tf']
    H_ol_margins_optimized = res_cost_optimized['H_ol_margins']
    H_n_bandwidth_optimized_Hz = res_cost_optimized['bandwidth_H_n'] * 180 / np.pi
    
    print()  
    print("============ After Optimization ===========") 
    print()  
    print("Cost function value:", cost_function_value_optimized )
    print("Evaluation result (variance terms):\n",evaluate_optimized.variance_terms)
    
    print("Optimized stability penalty:", stability_penalty)
    print("Optimized stability margin penalty:", sm_penalty)
    print("Optimized gain margin penalty:", gm_penalty)
    print('H_n peak penalty:', H_n_tf_peak_penalty)
    print('H_r peak penalty:', H_r_tf_peak_penalty)
    
    print("Optimized controller num and den:", 
          evaluate_optimized.controller_num, 
          evaluate_optimized.controller_den)
    print()
    print('H_r_tf:',H_r_tf_optimized)
    print('H_n_tf:',H_n_tf_optimized)
    
    # 7. Plotting
    
    # 7.1 bode figure for the transfer functions
    
    gm = H_ol_margins_optimized[0]
    pm = H_ol_margins_optimized[1]
    
    fig1, ax3 = bodeplot_Hz(
    transfer_functions_ct=H_n_tf_optimized,
    omega_limits=[1e-5,frame_rate/2],
    omega_num=1000,
    labels="H_n ",
    title="Transfer function H_n & H_r",
    subtitle=f"[GM: {gm:.2f} dB, PM: {pm:.2f} deg]")
    
    bodeplot_Hz(
    transfer_functions_ct=H_r_tf_optimized,
    omega_limits=[1e-5,frame_rate/2],
    omega_num=1000,
    labels="H_r ",
    styles={'linestyle':'--'},
    fig=fig1,
    ax1=ax3[0],
    ax2=ax3[1])
    
    # 7.2 Compare PSDs and plot
    
    PSD_in_atmos = evaluate_optimized.psd_input["atmosphere"][0, :]
    PSD_in_alias = evaluate_optimized.psd_input["aliasing"][0, :]
    PSD_in_meas  = evaluate_optimized.psd_input["measurement"][0, :]
    PSD_in_total = evaluate_optimized.psd_input["total"][0, :]
    
    PSD_out_atmos = evaluate_optimized.psd_output["atmosphere"][0, :]
    PSD_out_alias = evaluate_optimized.psd_output["aliasing"][0, :]
    PSD_out_meas  = evaluate_optimized.psd_output["measurement"][0, :]
    PSD_out_total = evaluate_optimized.psd_output["total"][0, :]
    
    plot_psds_single_mode(mode_index,
              temporal_freqs,
              PSD_in_atmos,
              PSD_in_alias,
              PSD_in_meas,
              PSD_in_total,
              PSD_out_atmos,
              PSD_out_alias,
              PSD_out_meas,
              PSD_out_total,
              plot_inputs=True,
              title_text=title_text)
    
    # 7.3 Nyquist plot for the open-loop transfer function
    
    freqs_nyquist = np.logspace(0, np.log10(frame_rate / 2.0), 2000)    
    nyquist_count = plot_nyquist(H_ol_tf_optimized,
                                 freqs_Hz=freqs_nyquist)    

    plt.show()
    
    return obj_to_optimize

if __name__ == "__main__":
    obj_to_optimize = optmization_auto_singlemode(param_dir = 'params_mod4_polynomial.yaml')