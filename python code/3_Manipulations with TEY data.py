# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 16:12:09 2025

@author: okostko
"""


import os
import re
import pandas as pd
import numpy as np

folder_path = r"C:/Oleg Kostko/GoogleDrive/python test/2025-05-17 BL12012 RGA-TEY CAR MeOx test short"
mapping_file = r"C:/Oleg Kostko/GoogleDrive/python test/2025-05-17 BL12012 RGA-TEY CAR MeOx test short/sample_holder_position_readout_2025-05-17.txt"  # Tab-separated file with sample/group info

#folder_path_TEY_norm = os.path.join(folder_path, "Analysis_results-ascii/TEY_normalized")
folder_path_TEY_norm = folder_path +"/Analysis_results-ascii/TEY_normalized"
os.makedirs(folder_path_TEY_norm, exist_ok=True)


output_data_folder = folder_path +"/Analysis_results-ascii/TEY_normalized_averaged"
os.makedirs(output_data_folder, exist_ok=True)

output_folder = folder_path +"/Analysis_results-ascii"

#--------------------TEY raw data processing_spike removal ------------------------ 
pd_pattern = re.compile(r"_PD_([-+]?\d*\.?\d+)uA", re.IGNORECASE)

def round_sig(x, sig=5):
    return np.round(x, sig - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0

def fix_spikes_with_time(time_array, data_array, start_time=62, threshold=3):
    """
    Fix spikes only for data points where time > start_time.
    """
    data_fixed = data_array.copy()
    # Get indices where time > start_time
    indices = np.where(time_array > start_time)[0]
    # Only check spikes from second to second-last in this subset to have neighbors
    for i in indices:
        if i == 0 or i == len(data_array) - 1:
            continue  # skip first and last index (no two neighbors)
        if time_array[i] <= start_time:
            continue
        
        prev_val = data_fixed[i - 1]
        curr_val = data_fixed[i]
        next_val = data_fixed[i + 1]
        
        neighbors_avg = (prev_val + next_val) / 2
        
        if neighbors_avg != 0:
            diff_ratio = abs(curr_val - neighbors_avg) / abs(neighbors_avg)
        else:
            diff_ratio = abs(curr_val - neighbors_avg)
        
        diff_prev = abs(curr_val - prev_val)
        diff_next = abs(curr_val - next_val)
        
        if diff_ratio > threshold and diff_prev > threshold * abs(prev_val) and diff_next > threshold * abs(next_val):
            data_fixed[i] = neighbors_avg
            
    return data_fixed

for filename in os.listdir(folder_path):
    if "TEY_" in filename and filename.lower().endswith(".txt"):
        match = pd_pattern.search(filename)
        if not match:
            print(f"⚠ Skipping {filename}: PD value not found.")
            continue
        
        pd_value_uA = float(match.group(1))
        file_path = os.path.join(folder_path, filename)
        
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
                "Time,s": df["Time,s"],
                "normalized_TEY": norm_tey_rounded
            })
            
            base_name = filename.split("_TEY_Dark")[0]
            new_filename = f"{base_name}_TEY_normalized.txt"
            output_path = os.path.join(folder_path_TEY_norm, new_filename)
            df_normalized.to_csv(output_path, sep="\t", index=False)
            print(f"✅ Saved normalized file: {new_filename}")
        
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            
#-------------------TEY averaging of normalized data---------------------------


# Load mapping file
mapping_df = pd.read_csv(mapping_file, sep="\t", dtype=str)
mapping_df = mapping_df.iloc[:, [3, 6]]  # 4th and 7th columns (0-based index)
mapping_df.columns = ["sample_name", "group_name"]

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
    stacked = np.stack(data_arrays, axis=0)  # files × rows × columns
    
    # Average times and TEYs line-by-line
    avg_time = stacked[:, :, 0].mean(axis=0)
    avg_tey = stacked[:, :, 1].mean(axis=0)
    std_tey = stacked[:, :, 1].std(axis=0, ddof=0)  # population std
    
    # Build DataFrame
    result_df = pd.DataFrame({
        "Time,s": avg_time,
        "average_TEY": avg_tey,
        "std_TEY": std_tey
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
        if 'Time,s' not in df.columns or 'normalized_TEY' not in df.columns:
            print(f"File {filename} missing required columns")
            continue

        subset = df[(df['Time,s'] >= 59.5) & (df['Time,s'] <= 60.5)]

        if subset.empty:
            print(f"No data in time window for {filename}")
            continue

        max_val = subset['normalized_TEY'].max()

        sample_name = filename.split("_TEY_")[0]

        results.append((sample_name, max_val))

output_df = pd.DataFrame(results, columns=['sample', 'TEY_t=0'])

output_path = os.path.join(output_folder, 'TEY_at_t=0.txt')
output_df.to_csv(output_path, sep='\t', index=False)


print("Done! Results saved to 'TEY_at_t0.txt'.")