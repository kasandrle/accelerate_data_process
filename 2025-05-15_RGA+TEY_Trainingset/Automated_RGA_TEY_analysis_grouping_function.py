#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:19:17 2025

@author: BJLuttgenau
"""

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import time
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt



def get_beam_intervals_in_seconds(time_array, beam_on_indices,BEAM_ON_USED_S=20,BEAM_OFF_BEFORE_S=20,BEAM_OFF_AFTER_S=30):
    """
    Given an array of times (seconds) and the indices where the shutter is on,
    return [start_idx, end_idx] for the beam-on region
    and two beam-off regions in seconds.
    """
    # Indices of first/last beam-on
    first_on_idx = beam_on_indices[0]
    last_on_idx  = beam_on_indices[-1]

    # Times (in seconds) of the first/last beam on
    time_first_on = time_array[first_on_idx]
    time_last_on  = time_array[last_on_idx]

    # --- Beam ON interval ---
    beam_on_start_time = time_first_on  # no extra margin at the start
    beam_on_end_time   = time_first_on + BEAM_ON_USED_S  # use only first X seconds
    on_start_idx = np.searchsorted(time_array, beam_on_start_time, side='left')
    on_end_idx   = np.searchsorted(time_array, beam_on_end_time,   side='right')
    beam_on_interval = [on_start_idx, on_end_idx]

    # --- Beam OFF 1 interval ---
    # from t=0 up to (time_first_on - BEAM_OFF_BEFORE_S)
    beam_off1_end_time = time_first_on - BEAM_OFF_BEFORE_S
    off1_end_idx = np.searchsorted(time_array, beam_off1_end_time, side='right')
    beam_off1_interval = [1, off1_end_idx]

    # --- Beam OFF 2 interval ---
    # from (time_last_on + BEAM_OFF_AFTER_S) to the end
    beam_off2_start_time = time_last_on + BEAM_OFF_AFTER_S
    off2_start_idx = np.searchsorted(time_array, beam_off2_start_time, side='left')
    beam_off2_interval = [off2_start_idx, len(time_array) - 1]

    return beam_on_interval, beam_off1_interval, beam_off2_interval

def convert_time_to_seconds(time_str_array):
    """
    Convert an array of time strings (format '%Y/%m/%d %H:%M:%S.%f') to seconds relative to the first timestamp.
    """
    start_date = datetime.strptime(time_str_array[0], '%Y/%m/%d %H:%M:%S.%f')
    return np.array([
        (datetime.strptime(dt, '%Y/%m/%d %H:%M:%S.%f') - start_date).total_seconds()
        for dt in time_str_array
    ])

# =================== FILE AND DATA FUNCTIONS ===================

def scan_folder(folder):
    """ Scan folder for TEY and RGA files, returning full paths. """
    TEY_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".txt") and "_TEY_" in file]
    rga_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".txt") and "_RGA_" in file]
    return TEY_files, rga_files


def extract_sample_name(filename):
    """
    Extract the sample name from the filename.
    Assumes the pattern: <SampleName>_RGA_ or <SampleName>_TEY_
    """
    filename = filename.split('/')[-1]
    if "_RGA_" in filename:
        return filename.split("_RGA_")[0].strip()
    elif "_TEY_" in filename:
        return filename.split("_TEY_")[0].strip()
    return None

def parse_photodiode_current(TEY_filename):
    """
    Extract the photodiode current (in microamps) from the TEY filename.
    Looks for a substring between '_PD_' and 'uA'. If not found, returns None.
    Example: "Sample1_TEY_PD_0.3uA_" -> returns 0.3
    """
    if "_PD_" in TEY_filename and "uA" in TEY_filename:
        part = TEY_filename.split("_PD_")[1]
        pd_str = part.split("uA")[0]
        try:
            return float(pd_str)
        except ValueError:
            return None
    return None

def determine_intervals(TEY_file, rga_file):
    """
    Determine the beam-on indices by comparing the TEY (pressure-current) data and RGA timestamps.
    """
    TEY_data = np.loadtxt(TEY_file, skiprows=1, delimiter='\t', dtype=float)
    rga_data = np.loadtxt(rga_file, skiprows=2, delimiter='\t', dtype=str)
    
    TEY_time = TEY_data[:, 0]
    shutter  = TEY_data[:, 2]
    
    rga_time_str = rga_data[:, 0]
    rga_time = np.array([datetime.strptime(t, '%Y/%m/%d %H:%M:%S.%f') for t in rga_time_str])
    
    beam_on_indices = []
    for idx, rga_time_val in enumerate(rga_time):
        # Find the index of the nearest TEY time value
        nearest_idx = np.abs(TEY_time - (rga_time_val - rga_time[0]).total_seconds()).argmin()
        if shutter[nearest_idx] == 1:
            beam_on_indices.append(idx)
                         
    return beam_on_indices

# =================== PLOTTING & PROCESSING FUNCTIONS ===================

def plot_TEY_data(TEY_files, sample_name,SAVE_IMAGES=False,save_path=None):
    """
    Plot the TEY (pressure-current) data. 
    Also returns the photodiode current for each TEY file (if found).
    """
    pd_currents = []
    for idx, TEY_file in enumerate(TEY_files):
        fig, ax = plt.subplots()
        data = np.loadtxt(TEY_file, skiprows=1, delimiter='\t', dtype=float)
        time = data[:, 0]
        TEY_signal = data[:, 1]
        
        # Extract photodiode current from filename
        pd_current = parse_photodiode_current(os.path.basename(TEY_file))
        pd_currents.append(pd_current)
        
        ax.plot(time, TEY_signal, label=f"TEY signal ({sample_name})", linewidth=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (A)')
        plt.title(f'TEY-Shutter Data - {sample_name}')
        plt.grid(True)
        plt.tight_layout()
        
        if SAVE_IMAGES:
            temp_fig_path = os.path.join(save_path, f'{sample_name}_TEY_shutter_{idx}.png')
            plt.savefig(temp_fig_path, bbox_inches='tight', dpi=300)
            plt.show()
        else:
            plt.show()
            #plt.close(fig)
    return pd_currents

def plot_all_columns_and_sum(data, sample_name,SAVE_IMAGES=False,save_path=None):
    """
    Sum the pressure values from all available mass channels and plot total pressure vs. time.
    The number of m/z channels is detected automatically.
    This is a raw sum (no background subtraction).
    """
    ncols = data.shape[1]  # first column is time; remaining columns are m/z channels
    total_pressure_sum = np.zeros(data.shape[0], dtype=float)
    for col in range(1, ncols):
        total_pressure_sum += data[:, col].astype(float)
    
    first_column = data[:, 0]
    first_column_adjusted = convert_time_to_seconds(first_column)
    
    # Filtering out negative values
    total_pressure_sum[total_pressure_sum < 0] = 0
    
    plt.figure(figsize=(10, 6))
    plt.plot(first_column_adjusted, total_pressure_sum, label='Sum of Pressures')
    plt.xlabel('Time (s)')
    plt.ylabel('Total ion signal (Torr)')
    plt.title(f'Total ion signal for {sample_name} (raw sum, no bkgd correction)')
    plt.grid(True)
    
    if SAVE_IMAGES:
        temp_fig_path = os.path.join(save_path, f'{sample_name}_sum_of_pressures.png')
        plt.savefig(temp_fig_path, bbox_inches='tight', dpi=300)
        plt.show()
    else:
        plt.show()
        #plt.close()

def process_and_plot_column(data, column_to_plot, sample_name, beam_on_indices,SAVE_IMAGES=False,save_path=None):
    """
    Process a given RGA mass channel (column), perform background subtraction based on beam-off intervals,
    and plot the corrected time trace. Returns:
       - beam_on_interval, beam_off1_interval, beam_off2_interval,
       - [avg_value]  (avg over beam-on region)
       - [std_value]  (std from beam-off region)
       - time_array: time (in seconds) for the measurement
       - corrected_data: background-corrected pressure time series
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    first_column = data[:, 0]
    column_data = data[:, column_to_plot].astype(float)
    time_array = convert_time_to_seconds(first_column)
    
    # Compute beam intervals
    beam_on_interval, beam_off1_interval, beam_off2_interval = get_beam_intervals_in_seconds(time_array, beam_on_indices)
    
    # Fit linear background using beam-off data
    x_fit = np.concatenate((
        time_array[beam_off1_interval[0]:beam_off1_interval[1]],
        time_array[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    y_fit = np.concatenate((
        column_data[beam_off1_interval[0]:beam_off1_interval[1]],
        column_data[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    regression_coefficients = np.polyfit(x_fit, y_fit, 1)
    background_line = np.polyval(regression_coefficients, time_array)
    
    corrected_data = column_data - background_line
    ax.plot(time_array, corrected_data, label=f"m/z = {column_to_plot}")
    
    # Compute average (over beam-on region) and std (from beam-off region)
    data_beam_on = corrected_data[beam_on_interval[0]: beam_on_interval[1]]
    data_beam_off = np.concatenate((
        corrected_data[beam_off1_interval[0]:beam_off1_interval[1]],
        corrected_data[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    avg_value = np.mean(data_beam_on)
    std_value = np.std(data_beam_off)
    
    # Highlight beam intervals on the plot
    ax.axvspan(time_array[beam_on_interval[0]], time_array[beam_on_interval[1]], color='orange', alpha=0.3)
    ax.axvspan(time_array[beam_off1_interval[0]], time_array[beam_off1_interval[1]], color='gray', alpha=0.3)
    ax.axvspan(time_array[beam_off2_interval[0]], time_array[beam_off2_interval[1]], color='gray', alpha=0.3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Ion signal (Torr)')
    plt.title(f'Isolated ion signal m/z = {column_to_plot} - {sample_name}')
    plt.grid(True)
    plt.tight_layout()
    
    
    if SAVE_IMAGES:
        temp_fig_path = os.path.join(save_path, f'{sample_name}_mass_{column_to_plot}_background_correction.png')
        plt.savefig(temp_fig_path, bbox_inches='tight', dpi=300)
        plt.show()
    else:
        plt.show()
        #plt.close(fig)

    
    return beam_on_interval, beam_off1_interval, beam_off2_interval, [avg_value], [std_value], time_array, corrected_data

def process_column(data, column_to_plot, sample_name, beam_on_indices,SAVE_IMAGES=False,save_path=None):
    """
    Process a given RGA mass channel (column), perform background subtraction based on beam-off intervals,
    and plot the corrected time trace. Returns:
       - beam_on_interval, beam_off1_interval, beam_off2_interval,
       - [avg_value]  (avg over beam-on region)
       - [std_value]  (std from beam-off region)
       - time_array: time (in seconds) for the measurement
       - corrected_data: background-corrected pressure time series
    """
    
    first_column = data[:, 0]
    column_data = data[:, column_to_plot].astype(float)
    time_array = convert_time_to_seconds(first_column)
    
    # Compute beam intervals
    beam_on_interval, beam_off1_interval, beam_off2_interval = get_beam_intervals_in_seconds(time_array, beam_on_indices)
    
    # Fit linear background using beam-off data
    x_fit = np.concatenate((
        time_array[beam_off1_interval[0]:beam_off1_interval[1]],
        time_array[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    y_fit = np.concatenate((
        column_data[beam_off1_interval[0]:beam_off1_interval[1]],
        column_data[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    regression_coefficients = np.polyfit(x_fit, y_fit, 1)
    background_line = np.polyval(regression_coefficients, time_array)
    
    corrected_data = column_data - background_line
    
    # Compute average (over beam-on region) and std (from beam-off region)
    data_beam_on = corrected_data[beam_on_interval[0]: beam_on_interval[1]]
    data_beam_off = np.concatenate((
        corrected_data[beam_off1_interval[0]:beam_off1_interval[1]],
        corrected_data[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    avg_value = np.mean(data_beam_on)
    std_value = np.std(data_beam_off) 

    
    return beam_on_interval, beam_off1_interval, beam_off2_interval, [avg_value], [std_value], time_array, corrected_data


def process_sample(rga_file, TEY_file, directory, save=True, plot=True,SAVE_IMAGES=False):
    """
    Processes a sample by extracting data, normalizing TEY signal, 
    processing RGA channels, and optionally saving & plotting results.

    Args:
        rga_file (str): Path to RGA file.
        TEY_file (str): Path to TEY file.
        directory (str): Base directory for saving data.
        save (bool): Whether to save results to disk.
        plot (bool): Whether to generate plots.

    Returns:
        dict: Processed data including TEY signals and outgassing statistics.
    """
    sample_name = extract_sample_name(rga_file)
    print(f"Processing sample: {sample_name}")

    # Load TEY & RGA data
    TEY_data = np.loadtxt(TEY_file, skiprows=1, delimiter='\t', dtype=float)
    rga_data = np.genfromtxt(rga_file, delimiter='\t', skip_header=2, dtype=str)
    ncols = rga_data.shape[1]

    # Normalize TEY data
    TEY_time, TEY_signal = TEY_data[:, 0], TEY_data[:, 1]
    pd_current = parse_photodiode_current(TEY_file)
    TEY_signal_normalized = TEY_signal / (pd_current * 1.0e-6) if pd_current else TEY_signal
    sample_TEY = {sample_name: (TEY_time, TEY_signal_normalized)}

    # Save TEY data
    if save:
        save_path = Path(directory) / "outgassing_data" / f"{sample_name}_TEY_normalized.txt"
        np.savetxt(save_path, np.column_stack((TEY_time, TEY_signal_normalized)), fmt="%.6e", delimiter="\t",
                   header=f"TEY data for {sample_name}\nTime(s)\tNormalized_TEY", comments="")

    # Plot total raw sum (no background correction)
    if plot:
        plot_all_columns_and_sum(rga_data, sample_name)

    # Process mass channels efficiently using dictionary comprehension
    sample_ion = {
        col: process_and_plot_column(rga_data, col, sample_name, determine_intervals(TEY_file, rga_file))
        for col in range(1, ncols)
    }

    # Extract outgassing averages and standard deviations
    outgassing_avg = np.array([sample_ion[col][3][0] for col in range(1, ncols)])
    outgassing_std = np.array([sample_ion[col][4][0] for col in range(1, ncols)])

    sample_outgassing = {sample_name: {'avg': outgassing_avg, 'std': outgassing_std}}

    # Save outgassing data summary
    if save:
        save_path = Path(directory) / "outgassing_data" / f"{sample_name}_outgassing_data_mean_std.txt"
        np.savetxt(save_path, np.column_stack((np.arange(1, ncols), outgassing_avg, outgassing_std)), fmt="%d\t%.6e\t%.6e",
                   delimiter="\t", header=f"Outgassing data {sample_name}\nMass number\tAvg Values (Torr)\tStd Values (Torr)", comments="")

        print(f"Data saved for {sample_name}")

    # ======= Plot Outgassing Spectra =======
    if plot:
        plt.figure(figsize=(8, 6))
        outgassing_avg[outgassing_avg < 0] = 0
        outgassing_std[outgassing_std > outgassing_avg] = 0

        plt.bar(np.arange(1, ncols), outgassing_avg, alpha=0.5, color='gray', edgecolor='black', label=sample_name)
        plt.errorbar(np.arange(1, ncols), outgassing_avg, yerr=outgassing_std, fmt='none', capsize=3, ecolor='black')

        for x_val in range(5, ncols, 5):
            plt.axvline(x=x_val, color='gray', linestyle='--', alpha=0.7, linewidth=0.5)

        plt.xlabel("m/z")
        plt.ylabel("Ion signal (Torr)")
        plt.title(f"{sample_name} Outgassing Spectrum (Linear Scale)")
        plt.grid(True)
        plt.tight_layout()

        if SAVE_IMAGES:
            save_path = Path(directory) / "plots" / f"{sample_name}_outgassing_spectrum_linear.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.show()
        else:
            plt.show()
            #plt.close()

    return {'TEY': sample_TEY, 'Ion': sample_ion, 'Outgassing': sample_outgassing}

def only_process_sample(rga_file, TEY_file, directory, save=True):
    """
    Processes a sample by extracting data, normalizing TEY signal, 
    processing RGA channels, and optionally saving & plotting results.

    Args:
        rga_file (str): Path to RGA file.
        TEY_file (str): Path to TEY file.
        directory (str): Base directory for saving data.
        save (bool): Whether to save results to disk.
        plot (bool): Whether to generate plots.

    Returns:
        dict: Processed data including TEY signals and outgassing statistics.
    """

    # Dictionaries to store per-sample data
    sample_outgassing = {}  # sample_name -> {'avg': array, 'std': array, 'sum_avg': float, 'sum_std': float}
    sample_TEY = {}         # sample_name -> (time_array, TEY_normalized) 
    sample_ion = {}         # sample_name -> { m/z : (time_array, corrected_data), 'sum': (time_array, sum_corrected_data) }
    

    sample_name = extract_sample_name(rga_file)
    print(f"Processing sample: {sample_name}")

    # Load TEY & RGA data
    TEY_data = np.loadtxt(TEY_file, skiprows=1, delimiter='\t', dtype=float)
    rga_data = np.genfromtxt(rga_file, delimiter='\t', skip_header=2, dtype=str)
    ncols = rga_data.shape[1]

    # Normalize TEY data
    TEY_time, TEY_signal = TEY_data[:, 0], TEY_data[:, 1]
    pd_current = parse_photodiode_current(TEY_file)
    TEY_signal_normalized = TEY_signal / (pd_current * 1.0e-6) if pd_current else TEY_signal
    sample_TEY = {sample_name: (TEY_time, TEY_signal_normalized)}

    # Save TEY data
    if save:
        save_path = Path(directory) / "outgassing_data" / f"{sample_name}_TEY_normalized.txt"
        np.savetxt(save_path, np.column_stack((TEY_time, TEY_signal_normalized)), fmt="%.6e", delimiter="\t",
                   header=f"TEY data for {sample_name}\nTime(s)\tNormalized_TEY", comments="")


    # Process mass channels efficiently using dictionary comprehension
    sample_ion = {
        col: process_column(rga_data, col, sample_name, determine_intervals(TEY_file, rga_file))
        for col in range(1, ncols)
    }

    # Extract outgassing averages and standard deviations
    outgassing_avg = np.array([sample_ion[col][3][0] for col in range(1, ncols)])
    outgassing_std = np.array([sample_ion[col][4][0] for col in range(1, ncols)])

    sample_outgassing = {sample_name: {'avg': outgassing_avg, 'std': outgassing_std}}

    # Save outgassing data summary
    if save:
        save_path = Path(directory) / "outgassing_data" / f"{sample_name}_outgassing_data_mean_std.txt"
        np.savetxt(save_path, np.column_stack((np.arange(1, ncols), outgassing_avg, outgassing_std)), fmt="%d\t%.6e\t%.6e",
                   delimiter="\t", header=f"Outgassing data {sample_name}\nMass number\tAvg Values (Torr)\tStd Values (Torr)", comments="")

        print(f"Data saved for {sample_name} in {save_path}")



    return sample_TEY, {sample_name : sample_ion}, sample_outgassing