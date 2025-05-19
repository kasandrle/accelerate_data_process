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

# =================== SETTINGS & FOLDERS ===================

# Set directory where the data files are located
directory = r'/Users/BJLuttgenau/Documents/Experiments/RGA/data_files/2024-12-15_RGA+TEY/'  # <-- adapt this to your file name/path
os.chdir(directory)

# Path to the tab-delimited txt file that contains the two columns "sample_name" and "group_name":
#   "sample_name" (matching the filenames) and "group_name"
txt_file = '/Users/BJLuttgenau/Documents/Experiments/RGA/data_files/2024-12-15_RGA+TEY/sample_holder_position_readout_241215.txt'  # <-- adapt this path

# =================== GLOBAL BEAM INTERVAL PARAMETERS ===================
# Define absolute times (in seconds)
BEAM_ON_USED_S          = 20.0  # how many seconds of beam-on intervals to include for determining values for RGA spectrum
BEAM_OFF_BEFORE_S       = 20.0  # how many seconds (before beam-on) to exclude for the first beam-off region
BEAM_OFF_AFTER_S        = 30.0  # how many seconds (after beam-on) to exclude for the second beam-off region

# ===================================================================================

SAVE_IMAGES = True  # True if you want to save or False if you don't want to save


# ===================================================================================

# Read Excel to define sample groups
df_groups = pd.read_csv(txt_file, sep='\t')  # specify tab delimiter
# Build a dictionary: group_name -> list of sample_names
sample_groups = {}
for grp, subdf in df_groups.groupby('group_name'):
    sample_groups[grp] = list(subdf['sample_name'].unique())

# Plotting parameters
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plot_size = (8, 6)  # Width, height in inches
# You may need to adapt the style if 'seaborn-v0_8-whitegrid' is not available
plot_style = 'seaborn-v0_8-whitegrid'
colormap = 'viridis'

plt.rcParams['figure.figsize'] = plot_size
plt.style.use(plot_style)
plt.rcParams['image.cmap'] = colormap
plt.rcParams.update({'lines.markeredgewidth': 1})
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Create folders for saving outputs
folders = {
    'raw_data': 'rawdataplots',
    'plots': 'plots',
    'outgassing': 'outgassing_data'
}
for key, folder in folders.items():
    try:
        os.makedirs(folder)
        print(f"Folder '{folder}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder}' already exists.")

def get_beam_intervals_in_seconds(time_array, beam_on_indices):
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
    """
    Scan the given folder for TEY and RGA files.
    Returns two lists: one for TEY files and one for RGA files.
    """
    TEY_files = []
    rga_files = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            if "_TEY_" in file:
                TEY_files.append(file)
            elif "_RGA_" in file:
                rga_files.append(file)
    return TEY_files, rga_files

def extract_sample_name(filename):
    """
    Extract the sample name from the filename.
    Assumes the pattern: <SampleName>_RGA_ or <SampleName>_TEY_
    """
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

def plot_TEY_data(TEY_files, sample_name):
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
        temp_fig_path = os.path.join(folders['plots'], f'{sample_name}_TEY_shutter_{idx}.png')
        if SAVE_IMAGES:
            plt.savefig(temp_fig_path, bbox_inches='tight', dpi=300)
            plt.show()
        else:
            plt.close(fig)
    return pd_currents

def plot_all_columns_and_sum(data, sample_name):
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
    temp_fig_path = os.path.join(folders['raw_data'], f'{sample_name}_sum_of_pressures.png')
    if SAVE_IMAGES:
        plt.savefig(temp_fig_path, bbox_inches='tight', dpi=300)
        plt.show()
    else:
        plt.close()

def process_and_plot_column(data, column_to_plot, sample_name, beam_on_indices):
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
    
    temp_fig_path = os.path.join(folders['raw_data'], f'{sample_name}_mass_{column_to_plot}_background_correction.png')
    if SAVE_IMAGES:
        plt.savefig(temp_fig_path, bbox_inches='tight', dpi=300)
        plt.show()
    else:
        plt.close(fig)

    
    return beam_on_interval, beam_off1_interval, beam_off2_interval, [avg_value], [std_value], time_array, corrected_data

# =================== MAIN SCRIPT ===================

if __name__ == "__main__":
    start_time = time.time()  # <-- Start the timer
    # Scan folder for TEY and RGA files
    TEY_files, rga_files = scan_folder(directory)
    TEY_files.sort()
    rga_files.sort()
    
    # Dictionaries to store per-sample data
    sample_outgassing = {}  # sample_name -> {'avg': array, 'std': array, 'sum_avg': float, 'sum_std': float}
    sample_TEY = {}         # sample_name -> (time_array, TEY_normalized) 
    sample_ion = {}         # sample_name -> { m/z : (time_array, corrected_data), 'sum': (time_array, sum_corrected_data) }
    
    # Process each sample (each pair of TEY and RGA files)
    for rga_file, TEY_file in zip(rga_files, TEY_files):
        rga_file_path = os.path.join(directory, rga_file)
        TEY_file_path = os.path.join(directory, TEY_file)
        
        # Extract sample name from the file name
        sample_name = extract_sample_name(rga_file)
        print(f"Processing sample: {sample_name}")
        
        # Determine beam-on indices based on TEY and RGA files
        beam_on_indices = determine_intervals(TEY_file_path, rga_file_path)
        
        # Process TEY data
        TEY_data = np.loadtxt(TEY_file_path, skiprows=1, delimiter='\t', dtype=float)
        TEY_time = TEY_data[:, 0]
        TEY_signal = TEY_data[:, 1]  # drain current
        # Get photodiode current (µA)
        pd_current = parse_photodiode_current(os.path.basename(TEY_file_path))
        if pd_current is None or pd_current == 0:
            # If not found or zero, just store raw TEY
            TEY_signal_normalized = TEY_signal
            print("Warning: Photodiode current not found (or zero). Storing raw TEY data.")
        else:
            # Normalize the TEY signal by the photodiode current (in A)
            TEY_signal_normalized = TEY_signal / (pd_current * 1.0e-6)
        
        sample_TEY[sample_name] = (TEY_time, TEY_signal_normalized)
        
        # Also quickly plot the TEY data (this helps keep your old plot)
        _ = plot_TEY_data([TEY_file_path], sample_name)
        
        # Save TEY data for this sample as .txt
        tey_out_path = os.path.join(
            folders['outgassing'], f'{sample_name}_TEY_normalized.txt'
        )
        with open(tey_out_path, 'w') as f_tey:
            f_tey.write(f"# TEY data for sample {sample_name}\n")
            f_tey.write(f"# time(s)\tnormalized_TEY\n")
            for tval, teyval in zip(TEY_time, TEY_signal_normalized):
                f_tey.write(f"{tval:.6e}\t{teyval:.6e}\n")
        print(f"Saved TEY data (normalized) to {tey_out_path}")
        
        # Load RGA data
        rga_data = np.genfromtxt(rga_file_path, delimiter='\t', skip_header=2, dtype=str)
        ncols = rga_data.shape[1]  # total columns; first column is time, remaining are m/z channels
        
        # Plot total raw sum (no background subtraction)
        plot_all_columns_and_sum(rga_data, sample_name)
        
        # Prepare placeholders for outgassing averages
        outgassing_avg_list = []
        outgassing_std_list = []
        
        # We'll also keep a running sum of the background-corrected data for each time point
        # so we can have a "sum-of-all-channels" background-corrected trace.
        # Initialize after we know the time array from the first channel.
        sum_corrected_data = None
        sum_beam_on_interval = None
        sum_beam_off1_interval = None
        sum_beam_off2_interval = None
        
        # Process each mass channel and collect outgassing + ion signal data
        sample_ion[sample_name] = {}
        for col in range(1, ncols):
            (beam_on_interval, beam_off1_interval, beam_off2_interval,
             avg_values, std_values, time_array, corrected_data) = process_and_plot_column(rga_data, col, sample_name, beam_on_indices)
            
            outgassing_avg_list.append(avg_values[0])
            outgassing_std_list.append(std_values[0])
            
            # Store the time + corrected data for individual sample, per m/z
            # As requested: time(s), corrected ion signal, and repeated std
            # (the same std_value for each time point)
            data_with_std = np.column_stack((
                time_array, 
                corrected_data, 
                np.full_like(corrected_data, std_values[0])
            ))
            rga_out_path = os.path.join(
                folders['outgassing'], f'{sample_name}_mz_{col}_corrected_signal.txt'
            )
            np.savetxt(
                rga_out_path,
                data_with_std,
                fmt='%.6e',
                delimiter='\t',
                header=(f"Ion signal for {sample_name}, m/z={col}\n"
                        "Time(s)\tCorrected Signal(Torr)\tStd(Torr)"),
                comments=''
            )
            
            # Store in the dictionary
            sample_ion[sample_name][col] = (time_array, corrected_data)
            
            # Accumulate the sum
            if sum_corrected_data is None:
                sum_corrected_data = np.copy(corrected_data)
                sum_beam_on_interval = beam_on_interval
                sum_beam_off1_interval = beam_off1_interval
                sum_beam_off2_interval = beam_off2_interval
            else:
                sum_corrected_data += corrected_data
        
        # After processing all columns, store the average outgassing data
        sample_outgassing[sample_name] = {
            'avg': np.array(outgassing_avg_list),  # shape (n_mz_channels,)
            'std': np.array(outgassing_std_list)
        }
        
        # Save individual sample outgassing data (average + std across m/z)
        # i.e. the typical "mass_number, avg, std"
        mass_numbers = np.arange(1, ncols)  # mass channels 1 ... (ncols-1)
        data_to_save = np.column_stack((mass_numbers, 
                                        sample_outgassing[sample_name]['avg'],
                                        sample_outgassing[sample_name]['std']))
        header = (f"Outgassing data {sample_name}\n"
                  "Mass number\tAvg Values (Torr)\tStd Values (Torr)")
        file_path = os.path.join(folders['outgassing'], f'{sample_name}_outgassing_data_mean_std.txt')
        np.savetxt(file_path, data_to_save, delimiter='\t', header=header, 
                   fmt='%d\t%.6e\t%.6e', comments='')
        print(f"Data saved to {file_path}")
        
        # ======= Also save the sum of the corrected data for each sample =======
        if sum_corrected_data is not None:
            # Compute a single standard deviation from beam-off region of the sum
            sum_data_beam_off = np.concatenate((
                sum_corrected_data[sum_beam_off1_interval[0]:sum_beam_off1_interval[1]],
                sum_corrected_data[sum_beam_off2_interval[0]:sum_beam_off2_interval[1]]
            ))
            sum_std_value = np.std(sum_data_beam_off)
            sum_data_with_std = np.column_stack((
                time_array, 
                sum_corrected_data, 
                np.full_like(sum_corrected_data, sum_std_value)
            ))
            sum_file = os.path.join(folders['outgassing'], f'{sample_name}_sum_corrected_signal.txt')
            np.savetxt(
                sum_file,
                sum_data_with_std,
                fmt='%.6e',
                delimiter='\t',
                header=(f"Sum of corrected signals for {sample_name}\n"
                        "Time(s)\tSumCorrected(Torr)\tStd(Torr)"),
                comments=''
            )
            print(f"Sum of corrected signals saved to {sum_file}")
            
            # Also store average over beam on region, and that std, in sample_outgassing:
            sum_data_beam_on = sum_corrected_data[sum_beam_on_interval[0]: sum_beam_on_interval[1]]
            sum_avg_value = np.mean(sum_data_beam_on)
            sample_outgassing[sample_name]['sum_avg'] = sum_avg_value
            sample_outgassing[sample_name]['sum_std'] = sum_std_value
        else:
            sample_outgassing[sample_name]['sum_avg'] = 0
            sample_outgassing[sample_name]['sum_std'] = 0
        
        # ======= Plot Individual Outgassing Spectrum (Linear Scale) =======
        plt.figure(figsize=plot_size)
        outgassing_avg = sample_outgassing[sample_name]['avg']
        outgassing_std = sample_outgassing[sample_name]['std']
        
        # Filter out negative avg values or when errorbar is larger than the avg
        outgassing_avg[outgassing_avg < 0] = 0
        outgassing_std[outgassing_std > outgassing_avg] = 0
        
        plt.bar(mass_numbers, outgassing_avg, alpha=0.5, color='gray', 
                edgecolor='black', label=sample_name)
        plt.errorbar(mass_numbers, outgassing_avg, 
                     yerr=outgassing_std, fmt='none', capsize=3, ecolor='black')
        
        # Add vertical grid lines every 5 m/z
        for x_val in range(5, ncols, 5):
            plt.axvline(x=x_val, color='gray', linestyle='--', 
                        alpha=0.7, linewidth=0.5)

        plt.xlabel('m/z')
        plt.ylabel('Ion signal (Torr)')
        plt.title(f'{sample_name} Outgassing Spectrum (Linear Scale)')
        plt.ylim(0,)
        plt.xlim(0, ncols)
        plt.grid(True)
        plt.tight_layout()
        single_spectrum_linear_file = os.path.join(
            folders['plots'], f'{sample_name}_outgassing_spectrum_linear.png'
        )
        if SAVE_IMAGES:
            plt.savefig(single_spectrum_linear_file, bbox_inches='tight', dpi=300)
            plt.show()
        else:
            plt.close()
        
        # ======= Plot Individual Outgassing Spectrum (Log Scale) =======
        plt.figure(figsize=plot_size)
        plt.bar(mass_numbers, outgassing_avg, alpha=0.5, color='gray', 
                edgecolor='black', label=sample_name)
        plt.errorbar(mass_numbers, outgassing_avg, 
                     yerr=outgassing_std, fmt='none', capsize=3, ecolor='black')
        
        # Add vertical grid lines every 5 m/z
        for x_val in range(5, ncols, 5):
            plt.axvline(x=x_val, color='gray', linestyle='--', 
                        alpha=0.7, linewidth=0.5)

        plt.xlabel('m/z')
        plt.ylabel('Ion signal (Torr)')
        plt.title(f'{sample_name} Outgassing Spectrum (Log Scale)')
        plt.yscale('log')
        plt.ylim(1e-12,)
        plt.xlim(0, ncols)
        plt.grid(True, which='both')
        plt.tight_layout()
        single_spectrum_log_file = os.path.join(
            folders['plots'], f'{sample_name}_outgassing_spectrum_log.png'
        )
        if SAVE_IMAGES:
            plt.savefig(single_spectrum_log_file, bbox_inches='tight', dpi=300)
            plt.show()
        else:
            plt.close()


    # =================== GROUP AVERAGING ===================
    # Take sample groups from Excel sheet.
    for group_name, sample_list in sample_groups.items():
        print(f"\nProcessing group: {group_name}")
        
        # ---- Outgassing Spectrum (per m/z channel) ----
        group_outgassing = []
        common_mz = None
        for s in sample_list:
            if s in sample_outgassing:
                n_mz = sample_outgassing[s]['avg'].size
                if common_mz is None:
                    common_mz = n_mz
                else:
                    common_mz = min(common_mz, n_mz)
                group_outgassing.append(sample_outgassing[s]['avg'][:common_mz])
        
        if group_outgassing:
            group_outgassing = np.array(group_outgassing)  # shape: (n_samples, common_mz)
            group_mean = np.mean(group_outgassing, axis=0)
            group_std  = np.std(group_outgassing, axis=0)
            
            # ---- Group Outgassing Spectrum (Linear Scale) ----
            plt.figure(figsize=plot_size)
            
            # Filter out negative avg values or where errorbar is larger than the avg
            group_mean[group_mean < 0] = 0
            group_std[group_std > group_mean] = 0
            
            mass_numbers = np.arange(1, common_mz + 1)
            plt.bar(mass_numbers, group_mean, alpha=0.5, color='gray', 
                    edgecolor='black', label=group_name)
            plt.errorbar(mass_numbers, group_mean, yerr=group_std, 
                         fmt='none', capsize=3, ecolor='black')
            
            # Vertical grid lines
            for x_val in range(5, common_mz + 1, 5):
                plt.axvline(x=x_val, color='gray', linestyle='--', 
                            alpha=0.7, linewidth=0.5)

            plt.xlabel('m/z')
            plt.ylabel('Ion signal (Torr)')
            plt.title(f'{group_name} Outgassing Spectrum (Linear Scale)')
            plt.ylim(0,)
            plt.xlim(0, common_mz + 1)
            plt.grid(True)
            plt.tight_layout()
            group_spectrum_linear_file = os.path.join(
                folders['plots'], f'{group_name}_outgassing_spectrum_linear.png'
            )
            if SAVE_IMAGES:
                plt.savefig(group_spectrum_linear_file, bbox_inches='tight', dpi=300)
                plt.show()
            else:
                plt.close()

            
            # ---- Group Outgassing Spectrum (Log Scale) ----
            plt.figure(figsize=plot_size)
            plt.bar(mass_numbers, group_mean, alpha=0.5, color='gray', 
                    edgecolor='black', label=group_name)
            plt.errorbar(mass_numbers, group_mean, yerr=group_std, 
                         fmt='none', capsize=3, ecolor='black')
            
            # Vertical grid lines
            for x_val in range(5, common_mz + 1, 5):
                plt.axvline(x=x_val, color='gray', linestyle='--', 
                            alpha=0.7, linewidth=0.5)

            plt.xlabel('m/z')
            plt.ylabel('Ion signal (Torr)')
            plt.title(f'{group_name} Outgassing Spectrum (Log Scale)')
            plt.yscale('log')
            plt.ylim(1e-12,)
            plt.xlim(0, common_mz + 1)
            plt.grid(True, which='both')
            plt.tight_layout()
            group_spectrum_log_file = os.path.join(
                folders['plots'], f'{group_name}_outgassing_spectrum_log.png'
            )
            if SAVE_IMAGES:
                plt.savefig(group_spectrum_log_file, bbox_inches='tight', dpi=300)
                plt.show()
            else:
                plt.close()



            # Save group outgassing data
            data_to_save = np.column_stack((mass_numbers, group_mean, group_std))
            header = (f"Group Outgassing data {group_name}\n"
                      "Mass number\tAvg Values (Torr)\tStd Values (Torr)")
            group_save_path = os.path.join(
                folders['outgassing'], f'{group_name}_outgassing_data_mean_std.txt'
            )
            np.savetxt(group_save_path, data_to_save, delimiter='\t', 
                       header=header, fmt='%d\t%.6e\t%.6e', comments='')
            print(f"Group outgassing data saved to {group_save_path}")
        else:
            print(f"No outgassing data for group {group_name}")
        
        # ---- TEY Signal Averaging ----
        group_TEY_signals = []
        common_TEY_time = None
        for s in sample_list:
            if s in sample_TEY:
                tey_time, tey_signal_norm = sample_TEY[s]
                group_TEY_signals.append(tey_signal_norm)
                if common_TEY_time is None:
                    common_TEY_time = tey_time
        
        if group_TEY_signals:
            # Determine the minimum length among the TEY signals
            min_length = min(len(signal) for signal in group_TEY_signals)
            # Truncate all TEY signals and the common time array to that length
            group_TEY_signals = np.array([signal[:min_length] for signal in group_TEY_signals])
            common_TEY_time = common_TEY_time[:min_length]
            
            tey_mean = np.mean(group_TEY_signals, axis=0)
            tey_std  = np.std(group_TEY_signals, axis=0)
            plt.figure(figsize=(8,6))
            plt.plot(common_TEY_time, tey_mean, 
                     label=f'{group_name} Average TEY', linewidth=3)
            plt.fill_between(common_TEY_time, tey_mean - tey_std, tey_mean + tey_std, alpha=0.3)
            plt.xlabel('Time (s)')
            plt.ylabel('DrainCurrent/DiodeCurrent')
            plt.title(f'{group_name} TEY Signal (Normalized)')
            plt.grid(True)
            plt.tight_layout()
            group_tey_file = os.path.join(
                folders['plots'], f'{group_name}_TEY_signal.png'
            )
            if SAVE_IMAGES:
                plt.savefig(group_tey_file, bbox_inches='tight', dpi=300)
                plt.show()
            else:
                plt.close()

            print(f"Group TEY signal saved to {group_tey_file}")
            
            # ---- Save group TEY data as txt ----
            tey_data_to_save = np.column_stack((common_TEY_time, tey_mean, tey_std))
            tey_header = (f"Averaged TEY data for {group_name}\n"
                          "Time(s)\tMeanTEY\tStd")
            tey_save_path = os.path.join(
                folders['outgassing'], f'{group_name}_TEY_signal.txt'
            )
            np.savetxt(tey_save_path, tey_data_to_save, delimiter='\t',
                       header=tey_header, fmt='%.6e', comments='')
            print(f"Group TEY data saved to {tey_save_path}")
        else:
            print(f"No TEY data for group {group_name}")
        
        # ---- Ion Signal Averaging (per m/z channel) ----
        # (Same as before)
        if not sample_list:
            print(f"No samples for group {group_name}")
            continue
        
        # We need to pick a reference time array for each m/z from the samples
        # We'll do it the same way as in outgassing (only if the sample is in sample_ion).
        # Then we can average the corrected signals across the samples in that group.
        if common_mz is None:
            print(f"No ion signal data for group {group_name}")
        else:
            for col in range(1, common_mz + 1):
                group_ion_signals = []
                common_time = None
                for s in sample_list:
                    if s in sample_ion and col in sample_ion[s]:
                        time_arr, corr_data = sample_ion[s][col]
                        group_ion_signals.append(corr_data)
                        if common_time is None:
                            common_time = time_arr
                if group_ion_signals:
                    group_ion_signals = np.array(group_ion_signals)  # shape: (n_samples, len(time))
                    ion_mean = np.mean(group_ion_signals, axis=0)
                    ion_std  = np.std(group_ion_signals, axis=0)
                    
                    # ---- Plot averaged isolated ion signal ----
                    plt.figure(figsize=(8,6))
                    plt.plot(common_time, ion_mean, label=f'{group_name} m/z = {col}', linewidth=3)
                    plt.fill_between(common_time, ion_mean - ion_std, ion_mean + ion_std, alpha=0.3)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Ion signal (Torr)')
                    plt.title(f'{group_name} isolated ion Signal for m/z = {col}')
                    plt.grid(True)
                    plt.tight_layout()
                    group_ion_file = os.path.join(
                        folders['raw_data'], f'{group_name}_mass_{col}_background_correction.png'
                    )
                    if SAVE_IMAGES:
                        plt.savefig(group_ion_file, bbox_inches='tight', dpi=300)
                        plt.show()
                    else:
                        plt.close()
                    
                    # ---- Save the averaged isolated ion signal data over time ----
                    data_to_save_ion = np.column_stack((common_time, ion_mean, ion_std))
                    header_ion = (
                        f"Averaged ion signal for {group_name}, m/z={col}\n"
                        "Time (s)\tMean Ion Signal (Torr)\tStd (Torr)"
                    )
                    ion_save_file = os.path.join(
                        folders['outgassing'], f'{group_name}_mass_{col}_ion_signal.txt'
                    )
                    np.savetxt(
                        ion_save_file, data_to_save_ion, delimiter='\t', 
                        header=header_ion, fmt='%.6e', comments=''
                    )
                    print(f"Averaged ion signal for m/z={col} saved to {ion_save_file}")
                else:
                    print(f"No data for m/z {col} in group {group_name}")
            
            # ---- Now do the sum of corrected signals for the group ----
            # We'll look for 'sum' in each sample_ion[sample_name], then average them.
            sum_signals_list = []
            common_sum_time = None
            for s in sample_list:
                # We stored the sum of corrected data in the dictionary in a variable way,
                # but let's check if we can retrieve it. We didn't store it under sample_ion[sample_name]['sum'],
                # but we did store the array in a local variable. We placed it in sample_outgassing[s]['sum_avg']
                # and 'sum_std' but not the time trace. We'll fix that:
                # Let's also store the entire time trace in sample_ion[s]['sum'] for convenience:
                pass
            

            
        # Now that the loop is done, let's do one final pass to store the sum in sample_ion
    # END of the main for-loop over samples


    end_time = time.time()  # <-- Stop the timer
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
