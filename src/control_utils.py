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
    actuators_number,
    WFS_num=None,
    WFS_den=None,
    RTC_num=None,
    RTC_den=None,
    ASM_num=None,
    ASM_den=None,
    gain=None,
    controller_num=None, 
    controller_den=None):
    
    t_0 = SingleModeControlOptimization.t_0
        
    if gain is not None:
        ctrl_num = np.zeros((actuators_number, 2), dtype=float)
        ctrl_den = np.zeros((actuators_number, 2), dtype=float)
        ctrl_num = np.column_stack((gain, np.zeros((actuators_number, 1), dtype=float)))
        ctrl_den[:] = [1, -1]
    elif controller_num is not None and controller_den is not None:
        ctrl_num = np.array(controller_num, dtype=float, copy=True)
        ctrl_den = np.array(controller_den, dtype=float, copy=True)
    else:
        raise ValueError(
            "Provide either 'gain' (integrator) or both "
            "'controller_num' and 'controller_den'")
    
    # ensure numpy array style
    WFS_num=np.asarray(WFS_num)
    WFS_den=np.asarray(WFS_den)
    RTC_num=np.asarray(RTC_num)
    RTC_den=np.asarray(RTC_den)
    ASM_num=np.asarray(ASM_num)
    ASM_den=np.asarray(ASM_den)
    
    # transfer functions for all modes
    WFS_tf = ct.tf(WFS_num, WFS_den, t_0)
    RTC_tf = ct.tf(RTC_num, RTC_den, t_0)
    ASM_tf = ct.tf(ASM_num, ASM_den, t_0)
    plant_tf = ct.series(WFS_tf, RTC_tf, ASM_tf)
    
    H_n_tf = np.zeros(actuators_number, dtype=object)
    H_r_tf = np.zeros(actuators_number, dtype=object)
    H_ol_tf = np.zeros(actuators_number, dtype=object)
    
    H_ol_gm = np.zeros(actuators_number, dtype=float)
    H_ol_pm = np.zeros(actuators_number, dtype=float)
    H_ol_sm = np.zeros(actuators_number, dtype=float)
    
    H_n_is_stable = np.zeros(actuators_number, dtype=bool)
    H_r_is_stable = np.zeros(actuators_number, dtype=bool)
    
    ctrl_tf = np.zeros(actuators_number, dtype=object)
    bandwidth_Hn = np.zeros(actuators_number, dtype=float)
    sensitivity_penalty = np.zeros(actuators_number, dtype=float)
    
    for i in range(actuators_number):        
        
        ctrl_tf[i] = ct.tf(ctrl_num[i, :], ctrl_den[i, :], t_0)

        H_ol_tf[i] = ct.series(plant_tf, ctrl_tf[i])
        
        # Closed-loop transfer functions
        # sensitivity function H_n and complementary sensitivity function H_r
        H_r_tf[i] = ct.feedback(1, H_ol_tf[i])
        H_n_tf[i] = ct.feedback(ct.series(RTC_tf, ctrl_tf[i], ASM_tf), WFS_tf)
        # H_n_tf[i] = ct.feedback(H_ol_tf[i], 1)    
        
        # gm - gain margin, pm - phase margin, sm - stability margin, wpc - phase crossover frequency,
        # wgc - gain crossover frequency, wms - stability margin crossover frequency
        H_ol_gm[i], H_ol_pm[i], H_ol_sm[i], _, _, _ = ct.stability_margins(H_ol_tf[i], returnall=False)
       
        print(f"Actuator {i}: Gain Margin = {H_ol_gm[i]}, Phase Margin = {H_ol_pm[i]}, Stability Margin = {H_ol_sm[i]}")
        
        # Check stability (boolean)
        H_n_is_stable[i] = all(np.abs(H_n_tf[i].poles()) < 1)
        H_r_is_stable[i] = all(np.abs(H_r_tf[i].poles()) < 1)
        
        sensitivity_penalty[i] = ct.dcgain(H_r_tf[i])
        # print(f"sensitivity_penalty {i}: {sensitivity_penalty[i]}")        
        
        try:
            bw_n = ct.bandwidth(H_n_tf[i])
        except Exception:  # if bandwidth cannot be computed, set it to infinity
            bw_n = 1e-6             # bw_r = float('inf')
        bandwidth_Hn[i] = bw_n
         
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
        actuators_number,
        WFS_num=None,
        WFS_den=None,
        RTC_num=None,
        RTC_den=None,
        ASM_num=None,
        ASM_den=None,
        gain = None,
        controller_num = None, 
        controller_den = None):
    
    if WFS_num is None or WFS_den is None:
        raise ValueError("Provide both 'WFS_num' and 'WFS_den'")
    if RTC_num is None or RTC_den is None:
        raise ValueError("Provide both 'RTC_num' and 'RTC_den'")
    if ASM_num is None or ASM_den is None:
        raise ValueError("Provide both 'ASM_num' and 'ASM_den'")
    
    # ensure numpy array style
    WFS_num=np.asarray(WFS_num)
    WFS_den=np.asarray(WFS_den)
    RTC_num=np.asarray(RTC_num)
    RTC_den=np.asarray(RTC_den)
    ASM_num=np.asarray(ASM_num)
    ASM_den=np.asarray(ASM_den)
    
    result_control_CL_tf = control_CL_tf_margin(
        obj_to_optimize,
        actuators_number,
        WFS_num,
        WFS_den,
        RTC_num,
        RTC_den,
        ASM_num,
        ASM_den,
        gain,
        controller_num, 
        controller_den)
    
    ctrl_tf = result_control_CL_tf["ctrl_tf"]
    H_ol_margins = result_control_CL_tf["H_ol_margins"]
    
    H_ol_gm = H_ol_margins[0]
    H_ol_pm = H_ol_margins[1]
    H_ol_sm = H_ol_margins[2]    
    
    CL_stability = result_control_CL_tf["CL_stability"]

    bandwidth_H_n = result_control_CL_tf["bandwidth_H_n"]

    H_n_tf = result_control_CL_tf["H_n_tf"]
    H_r_tf = result_control_CL_tf["H_r_tf"]
    H_ol_tf = result_control_CL_tf["H_ol_tf"]

    ctrl_num_evaluate = np.zeros((actuators_number, 2), dtype=float)
    ctrl_den_evaluate = np.zeros((actuators_number, 2), dtype=float)
    
    # cost_variance_result = []
    cost_variance = np.zeros(actuators_number, dtype=float)
    cost_variance_result = np.zeros(actuators_number, dtype=object)
    H_n_peak_penalty = np.zeros(actuators_number, dtype=float)
    H_r_peak_penalty = np.zeros(actuators_number, dtype=float)
    
    # 
    H_n_peak_limitation = 3 # dB
    H_r_peak_limitation = 2 # dB
    
    for i, ctrl_tf_i in enumerate(ctrl_tf):
        ctrl_num_evaluate[i] = ctrl_tf_i.num[0][0]
        ctrl_den_evaluate[i] = ctrl_tf_i.den[0][0]
        
        cost_variance_result_i = obj_to_optimize.evaluate(
        controller_num=ctrl_num_evaluate[i], 
        controller_den=ctrl_den_evaluate[i], 
        store_history=True)           
        
        cost_variance_result[i] = cost_variance_result_i
        cost_variance[i] = cost_variance_result_i.cost
        
        H_n_peak_penalty[i] = compute_close_loop_peak_penalty(H_n_tf[i], H_n_peak_limitation)
        H_r_peak_penalty[i] = compute_close_loop_peak_penalty(H_r_tf[i], H_r_peak_limitation)
    
    if not np.all(CL_stability):
        stability_penalty = 1e9  # A large penalty for instability
    else:
        stability_penalty = 0
    
    sm_target = 0.65 # Target stability margin (example value)
    sm_penalty = np.maximum(0, sm_target - H_ol_sm)  # Penalty for not meeting stability margin target
    sm_penalty = np.sum(sm_penalty) ** 2 # Sum penalty across all actuators
    
    H_n_tf_peak_penalty = np.sum(H_n_peak_penalty) ** 2
    H_r_tf_peak_penalty = np.sum(H_r_peak_penalty) ** 2
    
    gm_target = 3 
    gm_penalty = np.maximum(0, gm_target - H_ol_gm)
    gm_penalty = np.sum(gm_penalty) ** 2    
    
    """
    cost function = cost_variance + penalty for stability + margin to ensure stability
    where
    cost_variance = variance_vibr_CL + variance_alias_CL + variance_meas_CL;
    penalty for stability = 0 if stable, else a large number;
    margin to ensure stability = a large number * (max(0, target stability margin - actual stability margin));
    """
    
    # cost_function = cost_variance + stability_penalty + sm_penalty + close_loop_peak_penalty
    weight_cost = np.array([1, 1, 1e1, 1e2, 1e3, 1e3],dtype=float)
    cost_function = (np.sum(cost_variance) * weight_cost[0] 
                    + stability_penalty * weight_cost[1]
                    + sm_penalty * weight_cost[2]
                    + H_n_tf_peak_penalty * weight_cost[3]
                    + H_r_tf_peak_penalty * weight_cost[4]
                    + gm_penalty * weight_cost[5])
    print(f"Total cost: {cost_function}, Cost variance: {cost_variance}, Stability penalty: {stability_penalty}, Stability margin penalty: {sm_penalty}")
    
    # return [cost_function, cost_variance_result, stability_penalty, sm_penalty, H_r_tf, H_n_tf, H_n_tf_peak_penalty]
    return {
        "cost_function_value":
            cost_function,
        "total_variance":
            cost_variance_result,
        "penalty":
            [stability_penalty, sm_penalty, H_n_tf_peak_penalty, H_r_tf_peak_penalty, gm_penalty],
        "weight_penalty":
            weight_cost,
        "H_n_tf":
            H_n_tf,
        "H_r_tf":
            H_r_tf,  
        "H_ol_margins":
            H_ol_margins,
        "bandwidth_H_n":
            bandwidth_H_n,
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