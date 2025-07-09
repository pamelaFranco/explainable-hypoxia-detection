###############################################################################
# The code begins by loading the dataset from a specified directory, where it 
# searches for .mat files in subfolders containing 3D PO2 volumes. Each valid
# .mat file is loaded, and the script ensures that the data is in a 3D format.
# For each 3D volume, non-overlapping patches of a specified size (PATCH_SIZE)
# are extracted to process small regions of the volume independently. The 
# patches are analyzed to extract features that describe both the statistical 
# properties and spatial characteristics of the oxygen levels.
#
# Feature extraction includes several steps: first, basic statistics (mean,
# standard deviation, minimum, and maximum) are calculated for each patch. 
# Additionally, the spatial gradient of the patch is computed to capture
# variations in oxygen levels (∇PO2), followed by an evaluation of the
# proportion of the patch that represents well-oxygenated regions (i.e., 
# where oxygen levels exceed a threshold). Texture features are also extracted 
# from the patches using the Grey Level Co-occurrence Matrix (GLCM), which 
# calculates properties like contrast, homogeneity, energy, and correlation. 
# If GLCM extraction fails, the code handles this by filling the feature set 
# with zeros.
#
# Feature Extraction Script - Updated with Gabor, LBP, and Wavelet Features
#
# Each extracted patch is then labeled as either hypoxic (1) or normoxic (0) 
# based on its average intensity. If the mean intensity of the patch is below
# a set hypoxia threshold, it is classified as hypoxic; otherwise, it is 
# classified as normoxic. The extracted features and labels are stored in the X
# (features) and y (labels) arrays. These arrays can be used for further 
# analysis or machine learning models. The code concludes by printing out a 
# summary of the feature extraction process, including the shape of the feature
# matrix and the distribution of labels across the dataset.
#
# Author:      Dr. Pamela Franco
# Time-stamp:  2025-05-08
# E-mail:      pamela.franco@unab.cl /pafranco@uc.cl
# Python 3.11 compatible
###############################################################################
import pandas as pd
import os
import numpy as np
import mat73
import scipy.io
import pywt
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from scipy.ndimage import gaussian_gradient_magnitude
import warnings


# Definitions
dataset_path = r"C:\Users\pfran\Desktop\Vascular Response\Dataset"
PO2_THRESHOLD = 10
PATCH_SIZE = 16

# LBP parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

# Function to extract features from a 3D patch
def extract_features(patch):
    features = []

    # Basic statistics
    features.append(np.mean(patch))
    features.append(np.std(patch))
    features.append(np.min(patch))
    features.append(np.max(patch))

    # Spatial gradient
    grad = gaussian_gradient_magnitude(patch, sigma=1)
    features.append(np.mean(grad))
    features.append(np.std(grad))

    # Well-oxygenated proportion
    features.append(np.mean(patch > 30))

    # GLCM features
    try:
        glcm_feats = []
        for axis in [0, 1, 2]:
            slice_ = patch.take(indices=patch.shape[axis] // 2, axis=axis)
            scaled = np.uint8(255 * (slice_ - np.min(slice_)) / (np.ptp(slice_) + 1e-5))
            glcm = graycomatrix(scaled, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
                glcm_feats.append(graycoprops(glcm, prop)[0, 0])
        features.extend(glcm_feats)
    except:
        warnings.warn("GLCM failed; filling with zeros.")
        features.extend([0] * 12)

    # Gabor features
    try:
        gabor_feats = []
        frequencies = [0.1, 0.2, 0.3]
        thetas = [0, np.pi/4, np.pi/2]
        for axis in [0, 1, 2]:
            slice_ = patch.take(indices=patch.shape[axis] // 2, axis=axis)
            scaled = (slice_ - np.min(slice_)) / (np.ptp(slice_) + 1e-5)
            for freq in frequencies:
                for theta in thetas:
                    filt_real, filt_imag = gabor(scaled, frequency=freq, theta=theta)
                    magnitude = np.sqrt(filt_real**2 + filt_imag**2)
                    gabor_feats.append(np.mean(magnitude))
                    gabor_feats.append(np.std(magnitude))
        features.extend(gabor_feats)
    except:
        warnings.warn("Gabor filtering failed; filling with zeros.")
        features.extend([0] * 54)

    # LBP features
    try:
        lbp_feats = []
        for axis in [0, 1, 2]:
            slice_ = patch.take(indices=patch.shape[axis] // 2, axis=axis)
            scaled = np.uint8(255 * (slice_ - np.min(slice_)) / (np.ptp(slice_) + 1e-5))
            lbp = local_binary_pattern(scaled, P=LBP_POINTS, R=LBP_RADIUS, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), density=True)
            lbp_feats.extend(hist)
        features.extend(lbp_feats)
    except:
        warnings.warn("LBP failed; filling with zeros.")
        features.extend([0] * (3 * (LBP_POINTS + 2)))

    # Wavelet features (Daubechies 1)
    try:
        wavelet_feats = []
        for axis in [0, 1, 2]:
            slice_ = patch.take(indices=patch.shape[axis] // 2, axis=axis)
            coeffs2 = pywt.dwt2(slice_, 'db1')
            cA, (cH, cV, cD) = coeffs2
            for comp in [cA, cH, cV, cD]:
                wavelet_feats.append(np.mean(comp))
                wavelet_feats.append(np.std(comp))
        features.extend(wavelet_feats)
    except:
        warnings.warn("Wavelet transform failed; filling with zeros.")
        features.extend([0] * 24)

    return features

# Feature extraction loop
X = []
y = []

for folder_name in tqdm(os.listdir(dataset_path), desc="Processing folders"):
    subfolder_path = os.path.join(dataset_path, folder_name)
    if not os.path.isdir(subfolder_path):
        continue

    for file_name in os.listdir(subfolder_path):
        if not file_name.endswith('P.mat'):
            continue

        file_path = os.path.join(subfolder_path, file_name)
        try:
            mat_data = mat73.loadmat(file_path)
        except:
            try:
                mat_data = scipy.io.loadmat(file_path)
            except:
                print(f"❌ Failed to load: {file_path}")
                continue

        volume = None
        for key in mat_data:
            if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 3:
                volume = mat_data[key]
                break
        if volume is None:
            continue

        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

        sx, sy, sz = volume.shape
        pad_x = (PATCH_SIZE - sx % PATCH_SIZE) % PATCH_SIZE
        pad_y = (PATCH_SIZE - sy % PATCH_SIZE) % PATCH_SIZE
        pad_z = (PATCH_SIZE - sz % PATCH_SIZE) % PATCH_SIZE
        volume = np.pad(volume,
                        pad_width=((0, pad_x), (0, pad_y), (0, pad_z)),
                        mode='constant',
                        constant_values=0)

        sx_p, sy_p, sz_p = volume.shape
        for i in range(0, sx_p, PATCH_SIZE):
            for j in range(0, sy_p, PATCH_SIZE):
                for k in range(0, sz_p, PATCH_SIZE):
                    patch = volume[i:i+PATCH_SIZE, j:j+PATCH_SIZE, k:k+PATCH_SIZE]
                    label = int(np.mean(patch) < PO2_THRESHOLD)
                    features = extract_features(patch)
                    X.append(features)
                    y.append(label)

X = np.array(X)
y = np.array(y)

print("✔️ Feature extraction completed.")
print(f"Number of patches: {X.shape[0]}, Number of features: {X.shape[1]}")
print(f"Label distribution (0 = Normoxia, 1 = Hypoxia): {np.bincount(y)}")

# DataFrame
df = pd.DataFrame(X)

# Column names
feature_columns = [
    'Mean', 'Std', 'Min', 'Max',
    'Grad_Mean', 'Grad_Std', 'Well_Oxygenated_Proportion',
    'GLCM_Contrast_Axial', 'GLCM_Homogeneity_Axial', 'GLCM_Energy_Axial', 'GLCM_Correlation_Axial',
    'GLCM_Contrast_Sagittal', 'GLCM_Homogeneity_Sagittal', 'GLCM_Energy_Sagittal', 'GLCM_Correlation_Sagittal',
    'GLCM_Contrast_Coronal', 'GLCM_Homogeneity_Coronal', 'GLCM_Energy_Coronal', 'GLCM_Correlation_Coronal'
]

for plane in ['Axial', 'Sagittal', 'Coronal']:
    for freq in [0.1, 0.2, 0.3]:
        for theta in ['0', '45', '90']:
            feature_columns.append(f'Gabor_Mean_{plane}_f{freq}_t{theta}')
            feature_columns.append(f'Gabor_Std_{plane}_f{freq}_t{theta}')

for plane in ['Axial', 'Sagittal', 'Coronal']:
    for i in range(LBP_POINTS + 2):
        feature_columns.append(f'LBP_HistBin_{i}_{plane}')

for plane in ['Axial', 'Sagittal', 'Coronal']:
    for comp in ['cA', 'cH', 'cV', 'cD']:
        feature_columns.append(f'Wavelet_Mean_{plane}_{comp}')
        feature_columns.append(f'Wavelet_Std_{plane}_{comp}')

df.columns = feature_columns
df['Label'] = y

# Save output
csv_name = 'FeatureExtraction_Data_AllTexture.csv'
df.to_csv(os.path.join(dataset_path, csv_name), index=False)
print(f"✔️ Data saved to: {csv_name}")
