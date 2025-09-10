# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 17:51:39 2025

@author: okostko
"""

import os
import shutil
import pandas as pd
import re
import matplotlib.pyplot as plt

# ===== USER SETTINGS =====
folder_path = r"C:/Oleg Kostko/GoogleDrive/python test/2025-05-17 BL12012 RGA-TEY CAR MeOx test short"  # <-- change to your path
metadata_file = r"C:/Oleg Kostko/GoogleDrive/python test/2025-05-17 BL12012 RGA-TEY CAR MeOx test short/sample_holder_position_readout_2025-05-17.txt"  # <-- change to your metadata file path
# =========================

# Define the main output folder where results will be saved
output_folder_ascii = os.path.join(folder_path, "Analysis_results-ascii")


#-------------------------------Sorting ASCII files-------------------------------------------------

# Define subfolders to create
subfolders = [
    "TEY_normalized",
    "TEY_normalized_averaged",
    "MS",
    "MS_averaged",
    "MS(t)",
    "MS(t)_averaged",
    "Total_outgassing",
    "Total_outgassing_averaged"
]

# Create output folder and subfolders if they don't exist
os.makedirs(output_folder_ascii, exist_ok=True)
for sf in subfolders:
    os.makedirs(os.path.join(output_folder_ascii, sf), exist_ok=True)

# Define source folders
#tey_folder = os.path.join(folder_path, "Analysis results", "TEY-normalized")
outgassing_folder = os.path.join(folder_path, "outgassing_data")

# --- Read sample & group names from metadata file ---
sample_names = []
group_names = []
with open(metadata_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines[1:]:  # skip header
    parts = line.strip().split("\t")
    if len(parts) >= 7:
        sample_names.append(parts[3].strip())
        group_names.append(parts[6].strip())

# Remove duplicates
sample_names = list(set(sample_names))
group_names = list(set(group_names))

# --- TEY Files ---
#for fname in os.listdir(tey_folder):
#    fpath = os.path.join(tey_folder, fname)
#    if fname.endswith("_TEY_normalized.txt"):
#        shutil.copy(fpath, os.path.join(output_folder_ascii, "TEY_normalized"))
#    elif fname.endswith("_TEY_normalized_averaged.txt"):
#        shutil.copy(fpath, os.path.join(output_folder_ascii, "TEY_normalized_averaged"))
#    elif fname == "TEY at t=0.txt":
#        shutil.copy(fpath, output_folder_ascii)

# --- Outgassing Files ---
for fname in os.listdir(outgassing_folder):
    fpath = os.path.join(outgassing_folder, fname)

    # Sample-based (flexible start, suffix matching)
    for name in sample_names:
        if fname == f"{name}_MS-time_merged.txt":
            shutil.copy(fpath, os.path.join(output_folder_ascii, "MS(t)"))
        if fname == f"{name}_mass_spectrum_averaged.txt":
            shutil.copy(fpath, os.path.join(output_folder_ascii, "MS"))

    # Group-based (exact match required)
    for group in group_names:
        if fname == f"{group}_MS-time_merged.txt":
            shutil.copy(fpath, os.path.join(output_folder_ascii, "MS(t)_averaged"))
        if fname == f"{group}_mass_spectrum_averaged.txt":
            shutil.copy(fpath, os.path.join(output_folder_ascii, "MS_averaged"))

    # Total outgassing
    if fname.endswith("_total_outgassing.txt"):
        shutil.copy(fpath, os.path.join(output_folder_ascii, "Total_outgassing"))


    # Total outgassing averaged
    if fname.endswith("_total_outgassing_averaged.txt"):
        shutil.copy(fpath, os.path.join(output_folder_ascii, "Total_outgassing_averaged"))

# --- Copy sample_holder_position_..._post_analysis files ---
for fname in os.listdir(folder_path):
    if fname.startswith("sample_holder_position_") and fname.endswith("_post_analysis.txt"):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            shutil.copy(fpath, output_folder_ascii)

print("All files copied successfully!")



#--------------------------------Sorting plot files-------------------------------


# Define folders
output_main = os.path.join(folder_path, "Analysis_results-plots")
subfolders = [
#    "TEY",
#    "TEY_normalized_averaged",
    "MS",
    "MS_averaged",
    "Total_outgassing_averaged"
]

# Create output folders
for sub in subfolders:
    os.makedirs(os.path.join(output_main, sub), exist_ok=True)

# Helper function to copy matching files
def copy_matching_files(src_folder, pattern, dest_folder):
    for root, _, files in os.walk(src_folder):
        for f in files:
            if re.search(pattern, f):
                shutil.copy2(os.path.join(root, f), os.path.join(dest_folder, f))

# 1. Copy _total_outgassing_averaged.png
copy_matching_files(
    os.path.join(folder_path, "outgassing_data"),
    r"_total_outgassing_averaged\.png$",
    os.path.join(output_main, "Total_outgassing_averaged")
)


# Read metadata
df = pd.read_csv(metadata_file, sep="\t", header=0)
sample_names = df.iloc[:, 3].dropna().unique()  # 4th column (index 3)
sample_groups = df.iloc[:, 6].dropna().unique()  # 7th column (index 6) — not used here but read in case needed

# 4. Copy sample-based MS files
plots_folder = os.path.join(folder_path, "plots")

# 4. Copy sample-based MS files (any symbols before suffix)
for sample in sample_names:
    pattern = re.escape(sample) + r".*_outgassing_spectrum_log\.png$"
    copy_matching_files(plots_folder, pattern, os.path.join(output_main, "MS"))

# 5. Copy group-based MS_averaged files (no extra symbols before suffix)
for group in sample_groups:
    pattern = re.escape(group) + r"_outgassing_spectrum_log\.png$"
    copy_matching_files(plots_folder, pattern, os.path.join(output_main, "MS_averaged"))


print("✅ File copying complete.")

#-----------------------------------Plotting TEY data ------------------------------

input_folder1 = folder_path+ "/Analysis_results-ascii/TEY_normalized"
output_folder1 = folder_path+"/Analysis_results-plots/TEY_normalized"

input_folder2 = folder_path+ "/Analysis_results-ascii/TEY_normalized_averaged"
output_folder2 = folder_path+"/Analysis_results-plots/TEY_normalized_averaged"

#os.makedirs(output_folder, exist_ok=True)

def plot_ascii_files(input_folder, output_folder, extensions=(".txt", ".dat")):
    # Create output folder if it doesn’t exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(extensions):
            filepath = os.path.join(input_folder, filename)

            try:
                # Read file with pandas (handles tab/space delimiters)
                df = pd.read_csv(filepath, sep="\t")

                # Extract column names
                cols = df.columns.tolist()

                if len(cols) < 2:
                    print(f"Skipping {filename}, not enough columns")
                    continue

                x = df[cols[0]]
                y = df[cols[1]]

                plt.figure(figsize=(6,4))

                if len(cols) == 2:
                    plt.plot(x, y, linestyle="-", label=cols[1])
                elif len(cols) >= 3:
                    std = df[cols[2]]
                    plt.plot(x, y, color="blue", label=cols[1])
                    plt.fill_between(x, y-std, y+std, color="blue", alpha=0.3,
                                     label=f"{cols[1]} ± {cols[2]}")

                plt.xlabel(cols[0])
                plt.ylabel(cols[1])
                plt.title(filename)
                plt.legend()
                plt.tight_layout()

                # Save plot in output folder
                outpath = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
                plt.savefig(outpath, dpi=150)
                plt.close()
                print(f"Saved plot: {outpath}")

            except Exception as e:
                print(f"Could not process {filename}: {e}")


# Example usage:

plot_ascii_files(input_folder1, output_folder1)
plot_ascii_files(input_folder2, output_folder2)
