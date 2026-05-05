#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-04-16 14:03

Utility functions for control system calculations

@author: Jueqi Lin
"""

import numpy as np
import control as ct

def control_CL_tf_margin(
    SingleModeControlOptimization,
    **controller_param):
    # gain=None,
    # controller_num=None, 
    # controller_den=None,
    
    # compute the close-loop transfer function, stability margins for single mode
    t_0 = SingleModeControlOptimization.t_0
    
    WFS_num=SingleModeControlOptimization.num1
    WFS_den=SingleModeControlOptimization.den1
    ASM_num=SingleModeControlOptimization.num2
    ASM_den=SingleModeControlOptimization.den2
    RTC_num=SingleModeControlOptimization.num3
    RTC_den=SingleModeControlOptimization.den3

    WFS_num=np.asarray(WFS_num)
    WFS_den=np.asarray(WFS_den)
    RTC_num=np.asarray(RTC_num)
    RTC_den=np.asarray(RTC_den)
    ASM_num=np.asarray(ASM_num)
    ASM_den=np.asarray(ASM_den)
    
    if 'gain' in controller_param and controller_param['gain'] is not None:
        gain = controller_param['gain']
        if isinstance(gain, np.ndarray):
            gain = gain[0] 
        print(type(gain))
        print(f"Shape if array: {np.shape(gain) if hasattr(gain, '__len__') else 'scalar'}")
        ctrl_num = np.array([gain, 0.0])
        ctrl_den = np.array([1.0, -1.0])
        
    elif ('controller_num' in controller_param and 'controller_den' in controller_param 
         and controller_param['controller_num'] is not None and controller_param['controller_den'] is not None):
        controller_num = controller_param['controller_num']
        controller_den = controller_param['controller_den']
        ctrl_num = np.array(controller_num, dtype=float, copy=True)
        ctrl_den = np.array(controller_den, dtype=float, copy=True)
        
    else:
        raise ValueError(
            "Provide either 'gain' (integrator) or both "
            "'controller_num' and 'controller_den'")      
    
    # transfer functions for all modes
    WFS_tf = ct.tf(WFS_num, WFS_den, t_0)
    RTC_tf = ct.tf(RTC_num, RTC_den, t_0)
    ASM_tf = ct.tf(ASM_num, ASM_den, t_0)
    plant_tf = ct.series(WFS_tf, RTC_tf, ASM_tf)
    
    ctrl_tf = ct.tf(ctrl_num, ctrl_den, t_0)

    H_ol_tf = ct.series(plant_tf, ctrl_tf)
        
    # Closed-loop transfer functions
    # sensitivity function H_n and complementary sensitivity function H_r
    H_r_tf = ct.feedback(1, H_ol_tf)
    H_n_tf = ct.feedback(ct.series(RTC_tf, ctrl_tf, ASM_tf), WFS_tf)
           
    # gm - gain margin, pm - phase margin, sm - stability margin, wpc - phase crossover frequency,
    # wgc - gain crossover frequency, wms - stability margin crossover frequency
    H_ol_gm, H_ol_pm, H_ol_sm, _, _, _ = ct.stability_margins(H_ol_tf, returnall=False)
        
    # Check stability (boolean)
    H_n_is_stable = all(np.abs(H_n_tf.poles()) < 1)
    H_r_is_stable = all(np.abs(H_r_tf.poles()) < 1)
        
    sensitivity_penalty = ct.dcgain(H_r_tf)
        
    try:
        bw_n = ct.bandwidth(H_n_tf)
    except Exception:  # if bandwidth cannot be computed, set it to infinity
        bw_n = 0.0             # bw_r = float('inf')
    
    bandwidth_Hn = bw_n
         
    return {
        "ctrl_tf": 
            ctrl_tf,
        "H_n_tf":
            H_n_tf,  
        "H_r_tf":    
            H_r_tf,
        "H_ol_tf":
            H_ol_tf, 
        "H_ol_margins":
            [H_ol_gm, H_ol_pm, H_ol_sm],
        "CL_stability": 
            [H_n_is_stable, H_r_is_stable],
        "sensitivity_penalty":
            sensitivity_penalty,
        "bandwidth_H_n":
            bandwidth_Hn}

def cost(obj_to_optimize, 
         sm_target=None,
         gm_target=None,
         weight_cost=None,   
         **controller_param):
    
    WFS_num=obj_to_optimize.num1
    WFS_den=obj_to_optimize.den1
    ASM_num=obj_to_optimize.num2
    ASM_den=obj_to_optimize.den2
    RTC_num=obj_to_optimize.num3
    RTC_den=obj_to_optimize.den3
    
    # ensure numpy array style
    WFS_num=np.asarray(WFS_num)
    WFS_den=np.asarray(WFS_den)
    RTC_num=np.asarray(RTC_num)
    RTC_den=np.asarray(RTC_den)
    ASM_num=np.asarray(ASM_num)
    ASM_den=np.asarray(ASM_den)
    
    result_control_CL_tf = control_CL_tf_margin(
        obj_to_optimize,
        **controller_param)
    
    ctrl_tf = result_control_CL_tf["ctrl_tf"]
    H_ol_tf = result_control_CL_tf["H_ol_tf"]
    H_ol_margins = result_control_CL_tf["H_ol_margins"]
    
    H_ol_gm = H_ol_margins[0]
    H_ol_pm = H_ol_margins[1]
    H_ol_sm = H_ol_margins[2]    
    
    CL_stability = result_control_CL_tf["CL_stability"]

    bandwidth_H_n = result_control_CL_tf["bandwidth_H_n"]

    H_n_tf = result_control_CL_tf["H_n_tf"]
    H_r_tf = result_control_CL_tf["H_r_tf"]
    H_ol_tf = result_control_CL_tf["H_ol_tf"]
    
    H_n_peak_limitation = 3 # dB
    H_r_peak_limitation = 2 # dB    
    
    ctrl_num_evaluate = ctrl_tf.num[0][0]
    ctrl_den_evaluate = ctrl_tf.den[0][0]
    
    evaluate_result = obj_to_optimize.evaluate(
        controller_num=ctrl_num_evaluate, 
        controller_den=ctrl_den_evaluate, 
        store_history=True)   
    
    cost_variance_without_fitting = evaluate_result.cost - evaluate_result.variance_terms["fitting"]
    
    H_n_tf_peak_penalty = compute_close_loop_peak_penalty(H_n_tf, H_n_peak_limitation) 
    H_r_tf_peak_penalty = compute_close_loop_peak_penalty(H_r_tf, H_r_peak_limitation)    
    
    if not all(CL_stability):
        stability_penalty = 1e9  # A large penalty for instability
    else:
        stability_penalty = 0
    
    if sm_target is None:
        sm_target = 0.65 # Target stability margin (example value)
    sm_penalty = np.maximum(0, sm_target - H_ol_sm) ** 2  # Penalty for not meeting stability margin target   
    
    if gm_target is None:
        gm_target = 3
    gm_penalty = np.maximum(0, gm_target - H_ol_gm)** 2   
    
    """
    cost function = cost_variance_without_fitting * weight[0]
                    + penalty for stability * weight[1]
                    + stability margin * weight[2]
                    + penalty for H_n's peak * weight[3]
                    + penalty for H_r's peak * weight[4]
                    + gain margin * weight[5]
    where
    cost_variance_without_fitting = variance_vibr_CL + variance_alias_CL + variance_meas_CL;
    penalty for stability = 0 if stable, else a large number;
    margin to ensure stability = a large number * (max(0, target stability margin - actual stability margin));
    """
    
    if weight_cost is None:
        weight_cost = np.array([1, 1, 1e1, 1e2, 1e4, 1e3], dtype=float)
        
    cost_function = (cost_variance_without_fitting * weight_cost[0] 
                    + stability_penalty * weight_cost[1]
                    + sm_penalty * weight_cost[2]
                    + H_n_tf_peak_penalty * weight_cost[3]
                    + H_r_tf_peak_penalty * weight_cost[4]
                    + gm_penalty * weight_cost[5])
    
    print(f"\nTotal cost: {cost_function}")
    print(f"- Error variance (without fitting): {cost_variance_without_fitting}")
    print(f"- Stability penalty: {stability_penalty}")
    print(f"- Stability margin penalty: {sm_penalty}")
    print(f"- Gain margin penalty: {gm_penalty}")
    
    # return [cost_function, cost_variance_result, stability_penalty, sm_penalty, H_r_tf, H_n_tf, H_n_tf_peak_penalty]
    return {
        "cost_function_value":
            cost_function,
        "evaluate_result":
            evaluate_result,
        "penalty":
            [stability_penalty, sm_penalty, H_n_tf_peak_penalty, H_r_tf_peak_penalty, gm_penalty],
        "weight_penalty":
            weight_cost,
        "H_n_tf":
            H_n_tf,
        "H_r_tf":
            H_r_tf,  
        "H_ol_tf":
            H_ol_tf,
        "H_ol_margins":
            H_ol_margins,
        "bandwidth_H_n":
            bandwidth_H_n,
        "close_loop_stability":
            CL_stability
    }

def compute_close_loop_peak_penalty(
    H_cl_tf=None,
    close_loop_peak_target_dB=None):
    
    if H_cl_tf is None or close_loop_peak_target_dB is None:
        raise ValueError("Provide both close-loop transfer function and close loop peak limitation")
    
    H_cl_linf = ct.norm(H_cl_tf, p='inf')
    close_loop_peak_target_times = 10 ** (close_loop_peak_target_dB/20)
    
    if np.isinf(H_cl_linf):
        close_loop_peak_penalty = 1e9
    else:
        close_loop_peak_penalty = max(0, H_cl_linf/close_loop_peak_target_times-1)
    
    return(close_loop_peak_penalty)