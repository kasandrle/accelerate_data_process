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
from _functions import *


# =================== SETTINGS & FOLDERS ===================

# Set directory where the data files are located
directory = '/home/kas/Projects/accelerate_data_process/data example small/'  # <-- adapt this to your file name/path
os.chdir(directory)

# Path to the tab-delimited txt file that contains the two columns "sample_name" and "group_name":
#   "sample_name" (matching the filenames) and "group_name"
txt_file = '/home/kas/Projects/accelerate_data_process/data example small/sample_holder_position_readout_2025-06-08.txt'  # <-- adapt this path

# =================== GLOBAL BEAM INTERVAL PARAMETERS ===================
# Define absolute times (in seconds)
BEAM_ON_USED_S          = 30.0  # how many seconds of beam-on intervals to include for determining values for RGA spectrum
BEAM_OFF_BEFORE_S       = 20.0  # how many seconds (before beam-on) to exclude for the first beam-off region
BEAM_OFF_AFTER_S        = 30.0  # how many seconds (after beam-on) to exclude for the second beam-off region

# ===================================================================================

SAVE_IMAGES = False  # True if you want to save or False if you don't want to save

SAVE_EVERY_Pressure_IMAGE = False  # True if you want to save or False if you don't want to save Pressure(t) for every m/z


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


# =================== MAIN SCRIPT ===================

if __name__ == "__main__":
    start_time = time.time()  # <-- Start the timer
    # Scan folder for TEY and RGA files
    TEY_files = [f for f in os.listdir(directory) if f.endswith(".txt") and "_TEY_" in f]
    rga_files = [f for f in os.listdir(directory) if f.endswith(".txt") and "_RGA_" in f]
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
        #_ = plot_TEY_data([TEY_file_path], sample_name)
        
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
        #plot_all_columns_and_sum(rga_data, sample_name)
        
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
        #plt.figure(figsize=plot_size)
        outgassing_avg = sample_outgassing[sample_name]['avg']
        outgassing_std = sample_outgassing[sample_name]['std']
        
        # Filter out negative avg values or when errorbar is larger than the avg
        outgassing_avg[outgassing_avg < 0] = 0
        outgassing_std[outgassing_std > outgassing_avg] = 0
        
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
            #plt.figure(figsize=plot_size)
            
            # Filter out negative avg values or where errorbar is larger than the avg
            group_mean[group_mean < 0] = 0
            group_std[group_std > group_mean] = 0
            
            mass_numbers = np.arange(1, common_mz + 1)

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

    # ===== USER SETTINGS =====
    folder_path = r"C:/Oleg Kostko/GoogleDrive/python test/2025-05-17 BL12012 RGA-TEY CAR MeOx test short"  # change to your folder


    # ===== REGEX PATTERNS =====
    darkpd_pattern = re.compile(r"DarkPD_(-?\d+\.\d+)uA")
    pd_pattern = re.compile(r"_PD_(-?\d+\.\d+)uA")
    scanspeed_pattern = re.compile(r"scanspeed_(\d+)")
    scantime_pattern = re.compile(r"scantime_(\d+)")

    # ===== FIND ALL MAIN FILES =====
    main_files = [f for f in os.listdir(folder_path) if f.startswith("sample_holder_position_readout") and f.endswith(".txt")]

    if not main_files:
        raise FileNotFoundError("No files starting with 'sample_holder_position_readout' found.")

    # ===== PROCESS EACH MAIN FILE =====
    for main_fname in main_files:
        main_file = os.path.join(folder_path, main_fname)
        
        # Create copy for post-analysis
        base_name, ext = os.path.splitext(main_file)
        post_file = base_name + "_post_analysis" + ext
        shutil.copy2(main_file, post_file)
        print(f"Created copy: {post_file}")
        
        # Read the copied file
        df = pd.read_csv(post_file, sep="\t")
        
        # Ensure new columns exist
        df["DarkPD,A"] = None
        df["PD,A"] = None
        df["scanspeed"] = None
        df["scantime,s"] = None
        
        # ===== PROCESS TEY FILES =====
        for fname in os.listdir(folder_path):
            if "TEY" in fname and fname.endswith(".txt"):
                sample_name = fname.split("_TEY")[0]

                darkpd_match = darkpd_pattern.search(fname)
                pd_match = pd_pattern.search(fname)

                if darkpd_match and pd_match:
                    darkpd_val = float(darkpd_match.group(1)) * 1e-6
                    pd_val = float(pd_match.group(1)) * 1e-6

                    # Round to 5 significant digits
                    df.loc[df["sample_name"] == sample_name, "DarkPD,A"] = float(f"{darkpd_val:.5g}")
                    df.loc[df["sample_name"] == sample_name, "PD,A"] = float(f"{pd_val:.5g}")
        
        # ===== PROCESS RGA FILES =====
        for fname in os.listdir(folder_path):
            if "RGA_" in fname and fname.endswith(".txt"):
                sample_name = fname.split("_RGA_")[0]

                scanspeed_match = scanspeed_pattern.search(fname)
                scantime_match = scantime_pattern.search(fname)

                if scanspeed_match:
                    df.loc[df["sample_name"] == sample_name, "scanspeed"] = int(scanspeed_match.group(1))
                if scantime_match:
                    df.loc[df["sample_name"] == sample_name, "scantime,s"] = int(scantime_match.group(1))
        
        # ===== SAVE UPDATED COPY =====
        df.to_csv(post_file, sep="\t", index=False)
        print(f"Updated file saved to: {post_file}")

        compress_folder(folder_path)
