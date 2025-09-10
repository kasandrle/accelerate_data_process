# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 18:19:15 2025

@author: okostko
"""

import os
import shutil
#import zipfile

folder_path = r"C:/Oleg Kostko/GoogleDrive/python test/2025-05-17 BL12012 RGA-TEY CAR MeOx test short"  # <-- change to your path

def clean_folder(base_folder):
    # Path to the "outgassing_data" folder
#    outgassing_path = os.path.join(folder_path, "outgassing_data")
#    zip_path = os.path.join(folder_path, "outgassing_data_zip.zip")
    
    # Compress "outgassing_data" into zip
    #if os.path.isdir(outgassing_path):
     #   with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
      #      for root, _, files in os.walk(outgassing_path):
       #         for file in files:
        #            file_path = os.path.join(root, file)
         #           arcname = os.path.relpath(file_path, folder_path)
          #          zipf.write(file_path, arcname)
#        print(f"Compressed {outgassing_path} into {zip_path}")
 #   else:
  #      print(f"No 'outgassing_data' folder found in {folder_path}")
    
    # Folders to delete
    for folder_name in ["plots", "rawdataplots", "outgassing_data"]:
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"No folder '{folder_name}' found in {folder_path}")

# Example usage:
# change this to your target directory
clean_folder(folder_path)
