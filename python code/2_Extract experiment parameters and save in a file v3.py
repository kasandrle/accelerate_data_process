# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:14:52 2025

@author: okostko
"""

import os
import re
import shutil
import pandas as pd
import zipfile

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

#-----------------------------------------Saving a Zip of outgassing folder -----------------------

def compress_folder(base_folder):
    # Subfolders to compress
    subfolders = ["outgassing_data", "plots", "rawdataplots"]
    zip_path = os.path.join(base_folder, "outgassing_data_zip.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for sub in subfolders:
            folder_path = os.path.join(base_folder, sub)
            if os.path.isdir(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, base_folder)
                        zipf.write(file_path, arcname)
                print(f"Added '{sub}' to {zip_path}")
            else:
                print(f"No folder '{sub}' found in {base_folder}")

    print(f"All available subfolders compressed into {zip_path}")
    

# Example usage:
# change this to your target directory
compress_folder(folder_path)
