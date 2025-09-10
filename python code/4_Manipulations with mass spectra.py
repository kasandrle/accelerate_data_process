# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 15:52:32 2025

@author: okostko
"""


import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to folder containing files
folder_path = r"C:/Oleg Kostko/GoogleDrive/python test/2025-05-17 BL12012 RGA-TEY CAR MeOx test short"

folder_path = folder_path +"/outgassing_data"

###------------------Merge Time-dependent mass spectra---------------------------
# Regex to match both corrected_signal and ion_signal formats
mz_pattern = re.compile(r"(.+?)_mz_(\d+)_corrected_signal\.txt$")
mass_pattern = re.compile(r"(.+?)_mass_(\d+)_ion_signal\.txt$")

# Store data grouped by sample name
samples_data = {}

for filename in os.listdir(folder_path):
    if not filename.endswith(".txt"):
        continue

    mz_match = mz_pattern.match(filename)
    mass_match = mass_pattern.match(filename)

    if not mz_match and not mass_match:
        continue

    if mz_match:
        sample_name, mass_str = mz_match.groups()
    else:
        sample_name, mass_str = mass_match.groups()

    mass = int(mass_str)

    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path, sep="\t", skiprows=1)

    df = df.rename(columns={
        df.columns[0]: "Time(s)",
        df.columns[1]: f"mz{mass}(Torr)",
        df.columns[2]: f"std{mass}(Torr)"
    })

    if sample_name not in samples_data:
        samples_data[sample_name] = []

    samples_data[sample_name].append(df)

for sample_name, dfs in samples_data.items():
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="Time(s)", how="outer")

    merged_df = merged_df.sort_values(by="Time(s)")

    # Sort mass columns numerically
    time_col = ["Time(s)"]
    mass_cols = [col for col in merged_df.columns if col != "Time(s)"]

    # Extract mass numbers for sorting
    def get_mass_num(col):
        match = re.search(r"(\d+)", col)
        return int(match.group(1)) if match else float('inf')

    mass_cols_sorted = sorted(mass_cols, key=get_mass_num)

    merged_df = merged_df[time_col + mass_cols_sorted]

    output_file = os.path.join(folder_path, f"{sample_name}_MS-time_merged.txt")
    merged_df.to_csv(output_file, sep="\t", index=False)

    print(f"Merged file saved: {output_file}")

##----------Averaging Total outgassing signal ---------------------------------------

# Find all matching files
all_files = [f for f in os.listdir(folder_path) if f.endswith('_sum_corrected_signal.txt')]

# Group files by prefix before the number
pattern = re.compile(r'^(.*?)[ ]?\d+_sum_corrected_signal\.txt$')
groups = defaultdict(list)

for filename in all_files:
    match = pattern.match(filename)
    if match:
        prefix = match.group(1).strip()
        groups[prefix].append(filename)

# Process each group
for prefix, files in groups.items():
    time_data = None
    intensity_data = []

    for fname in files:
        path = os.path.join(folder_path, fname)
        data = np.loadtxt(path, usecols=(0, 1), skiprows=2)
        file_time = data[:, 0]
        file_intensity = data[:, 1]

        if time_data is None:
            time_data = file_time
        else:
            max_diff = np.max(np.abs(file_time - time_data))
            if max_diff > 0.5:
                raise ValueError(f"Time mismatch exceeds 0.5 s in file: {fname} (max diff: {max_diff:.3f} s)")

        intensity_data.append(file_intensity)

    intensity_data = np.array(intensity_data)
    intensity_avg = np.mean(intensity_data, axis=0)
    intensity_std = np.std(intensity_data, axis=0, ddof=1)

    # Combine time, average intensity, and standard deviation
    result = np.column_stack((time_data, intensity_avg, intensity_std))

    # Save to text file
    output_prefix = prefix.replace(' ', '_')
    txt_filename = f"{output_prefix}_total_outgassing_averaged.txt"
    txt_path = os.path.join(folder_path, txt_filename)
    header = "Time\tAveraged_Intensity\tStd_Deviation"
    np.savetxt(txt_path, result, fmt='%.6e', delimiter='\t', header=header, comments='')
    print(f"Saved data to: {txt_path}")

    # Create and save plot
    plt.figure(figsize=(8, 5))
    plt.plot(time_data, intensity_avg, label='Average Intensity', color='blue')
    plt.fill_between(time_data, intensity_avg - intensity_std, intensity_avg + intensity_std,
                     color='lightblue', alpha=0.5, label='±1 Std Dev')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.title(f'Averaged Signal for {prefix}')
    plt.legend()
    plt.tight_layout()

    png_filename = f"{output_prefix}_total_outgassing_averaged.png"
    png_path = os.path.join(folder_path, png_filename)
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {png_path}")
    
##--------------Renaming mass spectra-------------------

# Patterns to search for and their replacements
patterns = {
    "_outgassing_data_mean_std.txt": "_mass_spectrum_averaged.txt",
    "_sum_corrected_signal.txt": "_total_outgassing.txt"
}

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    for old_suffix, new_suffix in patterns.items():
        if filename.endswith(old_suffix):
            old_file_path = os.path.join(folder_path, filename)

            # Read the file skipping the first line
            with open(old_file_path, "r") as f:
                lines = f.readlines()[1:]  # skip the first line

            # Create new filename
            new_filename = filename.replace(old_suffix, new_suffix)
            new_file_path = os.path.join(folder_path, new_filename)

            # Write the remaining lines to the new file
            with open(new_file_path, "w") as f:
                f.writelines(lines)

            print(f"Processed: {filename} -> {new_filename}")
            break  # Stop checking other patterns for this file

print("All matching files have been processed.")