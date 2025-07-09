###############################################################################
# This code loads the P02 maps computed via finite difference methods in a 
# volume simulated with VasuSynth, corresponding to various tumor vascular 
# architectures.
#
# Author:      Dr. Pamela Franco
# Time-stamp:  2025-04-10
# E-mail:      pamela.franco@unab.cl /pafranco@uc.cl
# Python 3.11 compatible
###############################################################################
import os
import numpy as np
import mat73
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Use LaTeX-like math rendering with mathtext
plt.rcParams['text.usetex'] = True  
plt.rcParams['font.family'] = 'serif'

# Try to create a parula-like colormap
try:
    from parula import parula_map
    parula = LinearSegmentedColormap.from_list("parula", parula_map)
except:
    parula = plt.get_cmap("viridis")  # fallback

dataset_path = r"C:\Users\pfran\Desktop\Vascular Response\Dataset"  # Change the folder path

def show_slices(volume, title='', save_path=None):
    axial = volume[volume.shape[0] // 2, :, :]
    sagittal = volume[:, :, volume.shape[2] // 2]
    coronal = volume[:, volume.shape[1] // 2, :]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
    fig.suptitle(title, fontsize=16)

    im0 = axes[0].imshow(np.rot90(axial), cmap=parula)
    axes[0].set_title(r"$\mathrm{Axial}$")
    axes[0].axis('off')

    im1 = axes[1].imshow(np.rot90(sagittal), cmap=parula)
    axes[1].set_title(r"$\mathrm{Sagittal}$")
    axes[1].axis('off')

    im2 = axes[2].imshow(np.rot90(coronal), cmap=parula)
    axes[2].set_title(r"$\mathrm{Coronal}$")
    axes[2].axis('off')

    cbar = plt.colorbar(im2, cax=axes[3])
    cbar.set_label(r"$\mathrm{PO}_2\,(\mathrm{mmHg})$")

    plt.tight_layout()
    
    # If a save path is provided, save the image
    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the image with high resolution
        print(f"✔ Image saved at {save_path}")

    plt.show()

# Process all .mat files in the dataset
for folder_name in os.listdir(dataset_path):
    subfolder_path = os.path.join(dataset_path, folder_name)

    if os.path.isdir(subfolder_path):
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('P.mat'):
                file_path = os.path.join(subfolder_path, file_name)

                mat_data = None
                try:
                    mat_data = mat73.loadmat(file_path)
                except:
                    try:
                        mat_data = scipy.io.loadmat(file_path)
                    except:
                        print(f"❌ Could not load {file_path}")
                        continue

                for key in mat_data:
                    if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 3:
                        volume = mat_data[key]
                        
                        # Define the save path for each image
                        save_path = os.path.join(subfolder_path, f"{folder_name}_{file_name}_slices.png")
                        
                        # Display and save the images
                        show_slices(volume, title=fr"Approximate Vascular Fraction (Folder Name): {folder_name}, File Name: {file_name}", save_path=save_path)
                        break
