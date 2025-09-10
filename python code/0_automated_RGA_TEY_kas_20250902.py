#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 02 

@author: BJLuttgenau
@author: okostko
@author: kasandrle
"""

import os
import numpy as np
#from datetime import datetime
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import pandas as pd
#import time
from _functions import *
import argparse
import shutil

# ===== PARSE COMMAND-LINE ARGUMENTS =====
parser = argparse.ArgumentParser(description="Process outgassing data files.")
parser.add_argument(
    "directory",
    type=str,
    nargs="?",
    default="/home/kas/Projects/accelerate_data_process/data example small/",  #change here to folder you want to analysis
    help="Path to the folder containing data files (default: example folder)"
)
parser.add_argument(
    "--plot",
    action="store_true",
    default= False, #change here if you want to plot it
    help="Enable plotting of outgassing spectra"
)
args = parser.parse_args()


# ===== SET WORKING DIRECTORY =====
directory = args.directory
os.chdir(directory)

# =================== GLOBAL BEAM INTERVAL PARAMETERS ===================
# Define absolute times (in seconds)
BEAM_ON_USED_S          = 30.0  # how many seconds of beam-on intervals to include for determining values for RGA spectrum
BEAM_OFF_BEFORE_S       = 20.0  # how many seconds (before beam-on) to exclude for the first beam-off region
BEAM_OFF_AFTER_S        = 30.0  # how many seconds (after beam-on) to exclude for the second beam-off region

# ===================================================================================

SAVE_IMAGES = args.plot  # True if you want to save or False if you don't want to save

# =================== SETTINGS & FOLDERS ===================

# Set base directory
#directory = '/home/kas/Projects/accelerate_data_process/data example small/'
#os.chdir(directory)

# Regex pattern to match the desired file format
pattern = re.compile(r"sample_holder_position_readout_\d{4}-\d{2}-\d{2}\.txt")

#__________
# Find all matching files
matching_files = [f for f in os.listdir(directory) if f.startswith('sample_holder_position_readout_') and f.endswith('.txt')]

if not matching_files:
    raise FileNotFoundError("No matching 'sample_holder_position_readout_YYYY-MM-DD.txt' files found.")

# Sort files by date (assuming filenames are date-sorted)
matching_files.sort(reverse=True)

print(f"Using {len(matching_files)} matching files.")

# Initialize empty DataFrame to collect all entries
df_all_groups = pd.DataFrame()

# Read and concatenate all matching files
for fname in matching_files:
    fpath = os.path.join(directory, fname)
    df_temp = pd.read_csv(fpath, sep='\t')
    df_all_groups = pd.concat([df_all_groups, df_temp], ignore_index=True)

# Build dictionary: group_name -> list of unique sample_names
sample_groups = {}
for grp, subdf in df_all_groups.groupby('group_name'):
    sample_groups[grp] = sorted(subdf['sample_name'].unique())



#_____________

#txt_file = os.path.join(directory, matching_files[0])

# Plotting parameters
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

# Create folders for saving outputs

folders = ['Analysis_results-ascii',
           'Analysis_results-plots',
           "Analysis_results-ascii/TEY_normalized_averaged",
           "Analysis_results-ascii/TEY_normalized",
           "Analysis_results-ascii/MS",
           "Analysis_results-ascii/MS(t)",
           "Analysis_results-ascii/MS(t)_averaged",
           "Analysis_results-ascii/MS_averaged",
           "Analysis_results-ascii/Total_outgassing",
           "Analysis_results-ascii/Total_outgassing_averaged"]

for folder in folders:
    path = os.path.join(directory, folder)
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Folder '{folder}' is ready.")
    except Exception as e:
        print(f"Failed to create folder '{folder}': {e}")

output_folder = directory +"/Analysis_results-ascii"
output_folder_plots = directory+"/Analysis_results-plots"

# ===== REGEX PATTERNS =====
darkpd_pattern = re.compile(r"DarkPD_(-?\d+\.\d+)uA")
pd_pattern = re.compile(r"_PD_(-?\d+\.\d+)uA")
scanspeed_pattern = re.compile(r"scanspeed_(\d+)")
scantime_pattern = re.compile(r"scantime_(\d+)")

# ===== FIND ALL MAIN FILES =====
main_files = [
    f for f in os.listdir(directory)
    if f.startswith("sample_holder_position_readout")
    and f.endswith(".txt")
    and "_post_analysis" not in f
]


if not main_files:
    raise FileNotFoundError("No files starting with 'sample_holder_position_readout' found.")

# ===== SETUP OUTPUT SUBFOLDER =====
ascii_output_dir = output_folder #os.path.join(directory, "Analysis_results-ascii")
os.makedirs(ascii_output_dir, exist_ok=True)

# ===== PROCESS EACH MAIN FILE =====
for main_fname in main_files:
    main_file = os.path.join(directory, main_fname)

    # Create copy for post-analysis
    base_name, ext = os.path.splitext(main_fname)  # Use fname here to avoid duplicating path
    post_fname = base_name + "_post_analysis" + ext
    post_file = os.path.join(ascii_output_dir, post_fname)

    if os.path.exists(post_file):
        print(f"Post-analysis file already exists: {post_file}")
    else:
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
        for fname in os.listdir(directory):
            if "TEY" in fname and fname.endswith(".txt"):
                sample_name = fname.split("_TEY")[0]

                darkpd_match = darkpd_pattern.search(fname)
                pd_match = pd_pattern.search(fname)

                if darkpd_match and pd_match:
                    darkpd_val = float(darkpd_match.group(1)) * 1e-6
                    pd_val = float(pd_match.group(1)) * 1e-6

                    df.loc[df["sample_name"] == sample_name, "DarkPD,A"] = float(f"{darkpd_val:.5g}")
                    df.loc[df["sample_name"] == sample_name, "PD,A"] = float(f"{pd_val:.5g}")

        # ===== PROCESS RGA FILES =====
        for fname in os.listdir(directory):
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

#============save normalized TEY
pd_pattern = re.compile(r"_PD_([-+]?\d*\.?\d+)uA", re.IGNORECASE)
folder_path_TEY_norm = directory +"/Analysis_results-ascii/TEY_normalized"
output_data_folder = directory +"/Analysis_results-ascii/TEY_normalized_averaged"

for filename in os.listdir(directory):
    if "TEY_" in filename and filename.lower().endswith(".txt"):
        match = pd_pattern.search(filename)
        if not match:
            print(f"⚠ Skipping {filename}: PD value not found.")
            continue
        
        pd_value_uA = float(match.group(1))
        file_path = os.path.join(directory, filename)
        
        try:
            df = pd.read_csv(file_path, sep="\t")
            if "Time,s" not in df.columns or "TEY,A" not in df.columns:
                print(f"⚠ Skipping {filename}: Missing required columns.")
                continue
            
            norm_tey = df["TEY,A"] / pd_value_uA * 1e6
            
            # Fix spikes only after 62 seconds
            norm_tey_fixed = fix_spikes_with_time(df["Time,s"].values, norm_tey.values, start_time=62, threshold=5E-3)
            
            norm_tey_rounded = pd.Series(norm_tey_fixed).apply(lambda v: round_sig(v, 5))
            
            df_normalized = pd.DataFrame({
                "Time(s)": df["Time,s"],
                "Normalized_TEY": norm_tey_rounded
            })
            
            base_name = filename.split("_TEY_Dark")[0]
            new_filename = f"{base_name}_TEY_normalized.txt"
            output_path = os.path.join(folder_path_TEY_norm, new_filename)
            df_normalized.to_csv(output_path, sep="\t", index=False)
            print(f"✅ Saved normalized file: {new_filename}")
        
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

#-------------------TEY averaging of normalized data---------------------------
# Load and merge all mapping files
mapping_dfs = []

for fname in matching_files:
    fpath = os.path.join(directory, fname)
    df = pd.read_csv(fpath, sep="\t", dtype=str)
    df = df.iloc[:, [3, 6]]  # 4th and 7th columns (0-based index)
    df.columns = ["sample_name", "group_name"]
    mapping_dfs.append(df)

# Concatenate all mappings
mapping_df = pd.concat(mapping_dfs, ignore_index=True).drop_duplicates()

# Dictionary: group -> list of file paths
group_files = {}

for fname in os.listdir(folder_path_TEY_norm):
    if fname.endswith("_TEY_normalized.txt"):
        sample_name = fname.replace("_TEY_normalized.txt", "")
        group_name = mapping_df.loc[mapping_df["sample_name"] == sample_name, "group_name"]

        if not group_name.empty:
            gname = group_name.values[0]
            group_files.setdefault(gname, []).append(os.path.join(folder_path_TEY_norm, fname))

# Process each group
for group, files in group_files.items():
    data_arrays = []

    for fpath in files:
        df = pd.read_csv(fpath, sep="\t")
        df = df.iloc[:, [0, 1]]  # Only Time and TEY intensity
        df.columns = ["Time,s", "TEY"]
        data_arrays.append(df.to_numpy())

    # Find minimum number of rows (in case files differ in length)
    min_len = min(arr.shape[0] for arr in data_arrays)
    data_arrays = [arr[:min_len] for arr in data_arrays]

    # Stack into 3D array: shape (files, rows, 2)
    stacked = np.stack(data_arrays, axis=0)

    # Average times and TEYs line-by-line
    avg_time = stacked[:, :, 0].mean(axis=0)
    avg_tey = stacked[:, :, 1].mean(axis=0)
    std_tey = stacked[:, :, 1].std(axis=0, ddof=0)  # population std

    # Format time to avoid scientific notation and trailing zeros
    formatted_time = [f"{t:.6f}".rstrip('0').rstrip('.') for t in avg_time]

    # Build DataFrame
    result_df = pd.DataFrame({
        "Time(s)": formatted_time,
        "Averaged_TEY": avg_tey,
        "Std_TEY": std_tey
    })

    # Save in same folder
    output_path = os.path.join(output_data_folder, f"{group}_TEY_normalized_averaged.txt")
    result_df.to_csv(output_path, sep="\t", index=False, float_format="%.5g")

    print(f"Saved: {output_path}")


#-------------------- Search for Maximal TEY value--------------------------------------


results = []

for filename in os.listdir(folder_path_TEY_norm):
    if "TEY_" in filename and filename.endswith('.txt'):
        filepath = os.path.join(folder_path_TEY_norm, filename)
        try:
            df = pd.read_csv(filepath, sep=r"\s+", header=0)
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue

        # Check for the updated column names
        if 'Time(s)' not in df.columns or 'Normalized_TEY' not in df.columns:
            print(f"File {filename} missing required columns")
            continue

        subset = df[(df['Time(s)'] >= 59.5) & (df['Time(s)'] <= 60.5)]

        if subset.empty:
            print(f"No data in time window for {filename}")
            continue

        max_val = subset['Normalized_TEY'].max()

        sample_name = filename.split("_TEY_")[0]

        results.append((sample_name, max_val))

output_df = pd.DataFrame(results, columns=['sample', 'TEY_t=0'])

output_path = os.path.join(output_folder, 'TEY_at_t=0.txt')
output_df.to_csv(output_path, sep='\t', index=False)


print("Done! Results saved to 'TEY_at_t0.txt'.")

# Scan folder for TEY and RGA files
TEY_files = [f for f in os.listdir(directory) if f.endswith(".txt") and "_TEY_" in f]
rga_files = [f for f in os.listdir(directory) if f.endswith(".txt") and "_RGA_" in f]
TEY_files.sort()
rga_files.sort()

if len(TEY_files) != len(rga_files):
    raise ValueError(f"Mismatch in file counts: {len(TEY_files)} TEY files vs {len(rga_files)} RGA files.")

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

        #print(outgassing_avg_list)
        
        # Store the time + corrected data for individual sample, per m/z
        # As requested: time(s), corrected ion signal, and repeated std
        # (the same std_value for each time point)
        data_with_std = np.column_stack((
            time_array, 
            corrected_data, 
            np.full_like(corrected_data, std_values[0])
        ))
        
        # Store in the dictionary
        sample_ion[sample_name][col] = (time_array, corrected_data,np.full_like(corrected_data, std_values[0]))
        
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
    #file_path = os.path.join(folders['outgassing'], f'{sample_name}_outgassing_data_mean_std.txt')
    #np.savetxt(file_path, data_to_save, delimiter='\t', header=header, 
    #            fmt='%d\t%.6e\t%.6e', comments='')
    #print(f"Data saved to {file_path}")
    
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
        
        # Also store average over beam on region, and that std, in sample_outgassing:
        sum_data_beam_on = sum_corrected_data[sum_beam_on_interval[0]: sum_beam_on_interval[1]]
        sum_avg_value = np.mean(sum_data_beam_on)
        sample_outgassing[sample_name]['sum_avg'] = sum_avg_value
        sample_outgassing[sample_name]['sum_std'] = sum_std_value
        sample_ion[sample_name]['sum_std'] = sum_std_value
    else:
        sample_outgassing[sample_name]['sum_avg'] = 0
        sample_outgassing[sample_name]['sum_std'] = 0
        sample_ion[sample_name]['sum_std'] = 0
    

save_sample_ion_to_txt(sample_ion,os.path.join(output_folder,'MS(t)'))
save_mass_spectra_with_pandas(sample_outgassing, os.path.join(output_folder,'MS'))
save_sample_ion_to_total_outgassing_txt(sample_ion,os.path.join(output_folder,'Total_outgassing'))
save_grouped_mass_spectra(sample_outgassing, os.path.join(output_folder,'MS_averaged'))
save_gouped_sample_ion_to_txt(sample_ion,os.path.join(output_folder,'MS(t)_averaged'))
save_grouped_sample_ion_to_total_outgassing_txt(sample_ion,os.path.join(output_folder,'Total_outgassing_averaged'))

if SAVE_IMAGES:
    input_folder1 = output_folder+ "/TEY_normalized"
    output_folder1 = output_folder_plots+"/TEY_normalized"

    input_folder2 = output_folder+ "/TEY_normalized_averaged"
    output_folder2 = output_folder_plots+"/TEY_normalized_averaged"

    plot_ascii_files(input_folder1, output_folder1)
    plot_ascii_files(input_folder2, output_folder2)
    plot_MS(sample_outgassing,output_folder_plots+"/MS")
    plot_MS_from_folder(os.path.join(output_folder,'MS_averaged'),output_folder_plots+"/MS_averaged")
    plot_total_outgassing_from_folder(os.path.join(output_folder,'Total_outgassing_averaged'),output_folder_plots+"/Total_outgassing_averaged")