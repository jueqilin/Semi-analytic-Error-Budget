#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-04-24 18:12

@author: Jueqi Lin

======= control plot =======

"""

import numpy as np
import control as ct
import matplotlib.pyplot as plt
from src.Functions import build_transfer_function

# def bodeplot_Hz(
#     transfer_function_ct=None,
#     omega_limits=None,
#     omega_num=1000,
#     label=None,
#     title=None):
    
#     # after using this function, "plt.show()" must be used to display the figure
    
#     if transfer_function_ct is None:
#         raise ValueError("Transfer function must not be empty")
#     if omega_limits is None:
#         raise ValueError("omega_limits must not be empty")
#     if len(omega_limits) != 2:
#         raise ValueError("omega_limits must be a 2-element array/list")
    
#     omega_limits = np.asarray(omega_limits, dtype=float)
#     omega_limits_rad  = omega_limits * 2 * np.pi

#     mag, phase, omega = ct.frequency_response(
#         transfer_function_ct, 
#         omega_limits=omega_limits_rad,
#         omega_num=omega_num)    
    
#     phase_unwrap = np.unwrap(phase) 
#     phase_deg = np.rad2deg(phase_unwrap)  
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))
    
#     omega_Hz = omega / (2*np.pi)
#     ax1.semilogx(omega_Hz, 20 * np.log10(mag), linewidth=2, label=label)
#     ax1.set_ylabel('Magnitude [dB]')
#     if title:
#         ax1.set_title(title)
#     if label:
#         ax1.legend()
#     ax1.grid(True, linewidth=0.5, alpha=0.6)
    
#     ax2.semilogx(omega_Hz, phase_deg, linewidth=2, label=label)
#     ax2.set_xlabel('Frequency [Hz]')
#     ax2.set_ylabel('Phase [deg]')
#     if label:
#         ax2.legend()
#     ax2.grid(True, linewidth=0.5, alpha=0.6)
    
#     fig.set_tight_layout(True)
    
#     return fig    

def bodeplot_Hz(
    transfer_functions_ct=None,
    omega_limits=None,
    omega_num=1000,
    labels=None,
    styles=None,
    title=None,
    fig=None,
    ax1=None,
    ax2=None):
    
    # after using this function, "plt.show()" must be used to display the figure
    
    if transfer_functions_ct is None:
        raise ValueError("Transfer function(s) must not be empty")
    if omega_limits is None:
        raise ValueError("omega_limits must not be empty")
    
    # Determine if it's a single transfer function or collection
    try:
        n_systems = len(transfer_functions_ct)
        first_elem = transfer_functions_ct[0]
        if hasattr(first_elem, 'num') or hasattr(first_elem, 'poles'):
            pass
        else:
            n_systems = 1
            transfer_functions_ct = [transfer_functions_ct]
    except (TypeError, AttributeError):
        n_systems = 1
        transfer_functions_ct = [transfer_functions_ct]
    
    # Handle omega_limits
    omega_limits = np.asarray(omega_limits, dtype=float)
    if omega_limits.ndim == 1:
        if len(omega_limits) != 2:
            raise ValueError("omega_limits must be a 2-element array or Nx2 array for N systems")
        omega_limits_list = [omega_limits] * n_systems
    elif omega_limits.ndim == 2:
        if omega_limits.shape[0] != n_systems:
            raise ValueError(f"omega_limits has {omega_limits.shape[0]} rows but there are {n_systems} transfer functions")
        if omega_limits.shape[1] != 2:
            raise ValueError("omega_limits must have 2 columns (min and max frequency)")
        omega_limits_list = [omega_limits[i, :] for i in range(n_systems)]
    else:
        raise ValueError("omega_limits must be a 1D or 2D array")
    
    # Handle labels
    if labels is None:
        if n_systems == 1:
            labels = [None]
        else:
            labels = [f'System {i+1}' for i in range(n_systems)]
    elif isinstance(labels, str):
        if n_systems == 1:
            labels = [labels]
        else:
            labels = [f'{labels} [{i+1}]' for i in range(n_systems)]
    elif isinstance(labels, list):
        if len(labels) == 1 and n_systems > 1:
            base_label = labels[0] if labels[0] is not None else 'System'
            labels = [f'{base_label} [{i+1}]' for i in range(n_systems)]
        elif len(labels) != n_systems:
            raise ValueError(f"Number of labels ({len(labels)}) must match number of transfer functions ({n_systems})")
    else:
        if n_systems == 1:
            labels = [str(labels)]
        else:
            labels = [f'{labels} [{i+1}]' for i in range(n_systems)]
    
    # Handle styles
    if styles is None:
        # default style table
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        styles = [{'color': default_colors[i % len(default_colors)], 
                   'linestyle': '-', 
                   'linewidth': 2} for i in range(n_systems)]
    elif isinstance(styles, dict):
        styles = [styles] * n_systems
    elif isinstance(styles, list):
        if len(styles) == 1 and n_systems > 1:
            styles = styles * n_systems
        elif len(styles) != n_systems:
            raise ValueError(f"Number of styles ({len(styles)}) must match number of transfer functions ({n_systems})")
    else:
        raise ValueError("styles must be a dict, list of dicts, or None")
    
    # Create figure and axes if not provided
    if fig is None or ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))
    
    # Plot each transfer function
    for i in range(n_systems):
        tf_ct = transfer_functions_ct[i]
        label = labels[i]
        omega_lim = omega_limits_list[i]
        style = styles[i]
        
        omega_limits_rad = omega_lim * 2 * np.pi
        
        mag, phase, omega = ct.frequency_response(
            tf_ct, 
            omega_limits=omega_limits_rad,
            omega_num=omega_num)    
        
        phase_unwrap = np.unwrap(phase) 
        phase_deg = np.rad2deg(phase_unwrap)  
        
        omega_Hz = omega / (2*np.pi)
        
        ax1.semilogx(omega_Hz, 20 * np.log10(mag), 
                     color=style.get('color'),
                     linestyle=style.get('linestyle', '-'),
                     linewidth=style.get('linewidth', 2),
                     marker=style.get('marker'),
                     markersize=style.get('markersize', 4),
                     markevery=style.get('markevery', None),
                     alpha=style.get('alpha', 1.0),
                     label=label)
        
        ax2.semilogx(omega_Hz, phase_deg, 
                     color=style.get('color'),
                     linestyle=style.get('linestyle', '-'),
                     linewidth=style.get('linewidth', 2),
                     marker=style.get('marker'),
                     markersize=style.get('markersize', 4),
                     markevery=style.get('markevery', None),
                     alpha=style.get('alpha', 1.0),
                     label=label)
    
    # Configure axes
    ax1.set_ylabel('Magnitude [dB]')
    if title:
        ax1.set_title(title)
    if any(label is not None for label in labels) or n_systems > 1:
        ax1.legend()
    ax1.grid(True, linewidth=0.5, alpha=0.6)
    
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    ax2.grid(True, linewidth=0.5, alpha=0.6)
    
    fig.set_tight_layout(True)
    
    return fig, (ax1, ax2)

def psd_compare( 
        omega,
        t_0,
        n_actuators,
        plant_num,
        plant_den,
        gain=None,
        controller_num=None,
        controller_den=None,
        ):
    
    if gain is not None:
        H_r, H_n = build_transfer_function(
        omega,
        t_0,
        n_actuators,
        plant_num,
        plant_den,
        gain=gain,
    )
    elif controller_num is not None and controller_den is not None:
        H_r, H_n = build_transfer_function(
        omega,
        t_0,
        n_actuators,
        plant_num,
        plant_den,
        controller_num=controller_num,
        controller_den=controller_den,
    )
    else:
        raise ValueError(
            "Provide either 'gain' (integrator) or both "
            "'controller_num' and 'controller_den'")
    