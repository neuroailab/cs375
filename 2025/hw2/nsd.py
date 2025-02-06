#########################################
# 1. IMPORT LIBRARIES & SET GLOBAL VARS #
#########################################

import os
from os.path import exists
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import xarray as xr

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import fcluster

from sklearn.linear_model import RidgeCV  # using RidgeCV with a fixed alpha
from sklearn.metrics import r2_score as r2_score_sklearn

import gdown

# Threshold used for selecting reliable voxels.
NCSNR_THRESHOLD = 0.2


#########################################
# 2. HELPER FUNCTIONS                   #
#########################################

def r2_over_nc(y, y_pred, ncsnr):
    """
    Compute the R^2 score normalized by the noise ceiling (NC) as in Finzi et al (2022).
    The noise ceiling is computed as:
         NC = ncsnr^2 / (ncsnr^2 + 1/num_trials)
    If ncsnr is None, return the standard R^2.
    """
    ### TODO: Replace the code below with your implementation.
    # Instructions:
    # 1. If ncsnr is None, compute and return the standard R^2 score using
    #    r2_score_sklearn (with multioutput="raw_values").
    # 2. Otherwise, assume there are 3 target trials (i.e. set num_trials = 3.0).
    # 3. Compute the noise ceiling (NC) using the formula:
    #       NC = (ncsnr ** 2) / ( (ncsnr ** 2) + (1.0 / num_trials) )
    # 4. Compute the standard R^2 score (using r2_score_sklearn) and then
    #    return the normalized R^2 score by dividing the R^2 score by NC.
    pass


def get_metadata_concat_hemi(Y):
    """
    Concatenate left- and right-hemisphere metadata for voxels labeled 'nsdgeneral'
    and return the corresponding ncsnr values and metadata DataFrame.
    """
    ncsnr_full = np.concatenate((
        Y['voxel_metadata']['lh']['lh.ncsnr'],
        Y['voxel_metadata']['rh']['rh.ncsnr']
    ))
    
    nsdgeneral_idx = np.concatenate((
        Y['voxel_metadata']['lh']['lh.nsdgeneral.label'],
        Y['voxel_metadata']['rh']['rh.nsdgeneral.label']
    ))
    nsdgeneral_mask = np.logical_and(nsdgeneral_idx == 'nsdgeneral', ncsnr_full > 0)
    ncsnr_nsdgeneral = ncsnr_full[nsdgeneral_mask]
    
    metadata_lh = pd.DataFrame(Y['voxel_metadata']['lh'])
    metadata_rh = pd.DataFrame(Y['voxel_metadata']['rh'])
    nsdgeneral_metadata_df = pd.concat([metadata_lh, metadata_rh])[nsdgeneral_mask]
    
    return ncsnr_nsdgeneral, nsdgeneral_metadata_df


def get_data_dict(Y, brain_data_rep_averaged, ncsnr_nsdgeneral, nsdgeneral_metadata_df, verbose=True):
    """
    For each brain area (both streams and visual ROIs), select voxels with reliable responses
    (ncsnr above threshold) and return a dictionary with responses and ncsnr values.
    """
    data_dict = {}

    # Process streams-based areas.
    for area in ['ventral', 'parietal', 'lateral']:
        data_dict[area] = {}
        lh_area_mask = nsdgeneral_metadata_df['lh.streams.label'].astype(str).str.contains(area, na=False)
        rh_area_mask = nsdgeneral_metadata_df['rh.streams.label'].astype(str).str.contains(area, na=False)
        area_mask = np.logical_or(lh_area_mask, rh_area_mask)
        area_mask = np.logical_and(area_mask, ncsnr_nsdgeneral > NCSNR_THRESHOLD)
        
        if verbose:
            print(f"Size of area {area}: {np.sum(area_mask)}")
        
        area_data = brain_data_rep_averaged[:, area_mask]
        data_dict[area]["responses"] = area_data.copy()
        data_dict[area]["ncsnr"] = ncsnr_nsdgeneral[area_mask].copy()
        
        if verbose:
            print(f"Shape of area {area} responses: {data_dict[area]['responses'].shape}")

    # Process visual ROIs.
    for area in ['V1', 'V2', 'V3', 'V4']:
        data_dict[area] = {}
        lh_area_mask = nsdgeneral_metadata_df['lh.prf-visualrois.label'].astype(str).str.contains(area, na=False)
        rh_area_mask = nsdgeneral_metadata_df['rh.prf-visualrois.label'].astype(str).str.contains(area, na=False)
        area_mask = np.logical_or(lh_area_mask, rh_area_mask)
        area_mask = np.logical_and(area_mask, ncsnr_nsdgeneral > NCSNR_THRESHOLD)
        
        if verbose:
            print(f"Size of area {area}: {np.sum(area_mask)}")
        
        area_data = brain_data_rep_averaged[:, area_mask]
        data_dict[area]["responses"] = area_data.copy()
        data_dict[area]["ncsnr"] = ncsnr_nsdgeneral[area_mask].copy()
        
        if verbose:
            print(f"Shape of area {area} responses: {data_dict[area]['responses'].shape}")

    return data_dict


#########################################
# 3. DOWNLOAD & LOAD NSD DATA           #
#########################################

# Create a data directory if it does not exist.
datadir = os.path.join(os.getcwd(), 'data')
os.makedirs(datadir, exist_ok=True)

# Define subject and corresponding file_id.
subj = 'subj01'  # choose subject: available subjects are subj01, subj02, subj05, subj07
overwrite = False

if subj == 'subj01':
    file_id = '13cRiwhjurCdr4G2omRZSOMO_tmatjdQr'
elif subj == 'subj02':
    file_id = '1MO9reLoV4fqu6Weh4gmE78KJVtxg72ID'
elif subj == 'subj05':
    file_id = '11dPt3Llj6eAEDJnaRy8Ch5CxfeKijX_t'
elif subj == 'subj07':
    file_id = '1HX-6t4c6js6J_vP4Xo0h1fbK2WINpwem'
    
url = f'https://drive.google.com/uc?id={file_id}&export=download'
output = os.path.join(datadir, f'{subj}_nativesurface_nsdgeneral.pkl')

if not exists(output) or overwrite:
    gdown.download(url, output, quiet=False)

# Load NSD data.
Y = np.load(output, allow_pickle=True)
print("Keys in Y:", Y.keys())

# Print shapes of image bricks for each partition.
for partition in ['train', 'val', 'test']:
    print(f"Shape of image brick ({partition}):", Y['image_data'][partition].shape)


#########################################
# 4. PLOT EXAMPLE NSD IMAGE             #
#########################################

idx = 10  # example index for an image
plt.imshow(Y['image_data']['test'][idx])
plt.axis('off')
plt.savefig('nsd_image.png', bbox_inches='tight', dpi=300)
plt.close()


#########################################
# 5. PREPARE FMRI DATA                  #
#########################################

# Concatenate full brain ncsnr and nsdgeneral labels.
ncsnr_full = np.concatenate((
    Y['voxel_metadata']['lh']['lh.ncsnr'].values,
    Y['voxel_metadata']['rh']['rh.ncsnr'].values
))
nsdgeneral_idx = np.concatenate((
    Y['voxel_metadata']['lh']['lh.nsdgeneral.label'].values,
    Y['voxel_metadata']['rh']['rh.nsdgeneral.label'].values
))
print(ncsnr_full.shape, round(np.mean(ncsnr_full), 4), round(np.std(ncsnr_full), 4))
print(np.unique(nsdgeneral_idx))
print(np.count_nonzero(nsdgeneral_idx == 'nsdgeneral'))

# Select only nsdgeneral voxels with positive ncsnr.
nsdgeneral_mask = np.logical_and(nsdgeneral_idx == 'nsdgeneral', ncsnr_full > 0)
ncsnr_nsdgeneral = ncsnr_full[nsdgeneral_mask]
print(ncsnr_nsdgeneral.shape, round(np.mean(ncsnr_nsdgeneral), 4), round(np.std(ncsnr_nsdgeneral), 4))

# Combine metadata for nsdgeneral voxels.
nsdgeneral_metadata = pd.concat((
    Y['voxel_metadata']['lh'],
    Y['voxel_metadata']['rh']
))[nsdgeneral_mask]
ncsnr_nsdgeneral, nsdgeneral_metadata_df = get_metadata_concat_hemi(Y)

# Concatenate train and validation brain data and average over repetitions.
train_brain_data_cat = np.concatenate((
    Y['brain_data']['train']['lh'],
    Y['brain_data']['train']['rh']
), axis=2)
val_brain_data_cat = np.concatenate((
    Y['brain_data']['val']['lh'],
    Y['brain_data']['val']['rh']
), axis=2)
train_brain_data_cat = np.concatenate((train_brain_data_cat, val_brain_data_cat), axis=0)
train_brain_data_cat = np.mean(train_brain_data_cat, axis=1)

# Average test brain data over repetitions.
test_brain_data_cat = np.concatenate((
    Y['brain_data']['test']['lh'],
    Y['brain_data']['test']['rh']
), axis=2)
test_brain_data_cat = np.mean(test_brain_data_cat, axis=1)

# Get fMRI data dictionaries for train and test sets.
train_fmri_data = get_data_dict(Y, train_brain_data_cat, ncsnr_nsdgeneral, nsdgeneral_metadata_df)
test_fmri_data = get_data_dict(Y, test_brain_data_cat, ncsnr_nsdgeneral, nsdgeneral_metadata_df)

# Use both train and validation images for training.
train_image_data = np.concatenate((Y['image_data']['train'], Y['image_data']['val']), axis=0)
test_image_data = Y['image_data']['test']

# Define a torchvision transform: resize, center crop, convert to tensor, and normalize.
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#########################################
# 6. DEFINE MODIFIED ALEXNET MODEL      #
#########################################

### TODO: Implement a modified version of the AlexNet model.
# Instructions:
# 1. Define a class called AlexNet that inherits from nn.Module.
# 2. Implement AlexNet as in HW1
# 3. In the forward method:
#    - Pass the input through the convolutional feature extractor.
#    - After each MaxPool2d layer in the features, flatten the output and store it in a dictionary with a key indicating the layer number (e.g., "conv_pool_after_layer_X").
#    - After processing the convolutional layers, apply the adaptive average pooling and flatten the result.
#    - Pass the flattened tensor through the classifier.
#    - Capture the activations from the first fully-connected layer ("fc1") and the second fully-connected layer ("fc2") and store them in the dictionary.
#    - Return the dictionary containing all the captured features.
#
# Note: Do not capture the final output of the network; only capture the intermediate features as specified.
class AlexNet(nn.Module):
    pass


#########################################
# 7. SET UP MODELS (RANDOM & PRETRAINED)#
#########################################

# Set device.
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Model with random initialization.
model_random = AlexNet().to(device)
model_random.eval()

# Model loaded from an ImageNet checkpoint.
model_loaded = AlexNet().to(device)
### TODO: Replace the placeholder with the actual path to the ImageNet checkpoint.
checkpoint = torch.load("path/to/imagenet_checkpoint.pth",
                         map_location=device)
model_loaded.load_state_dict(checkpoint['model_state_dict'])
model_loaded.eval()

# Model loaded from a barcode checkpoint.
model_barcode = AlexNet(num_classes=32).to(device)
### TODO: Replace the placeholder with the actual path to the barcode checkpoint.
checkpoint = torch.load("path/to/barcode_checkpoint.pth", map_location=device)
model_barcode.load_state_dict(checkpoint['model_state_dict'])
model_barcode.eval()


########################################
# 8. PROCESS IMAGES & EXTRACT FEATURES  #
########################################

def get_model_activations(model, image_data, batch_size=32):
    """
    Process images through the given model in batches and return a dictionary
    containing activations for each recorded feature.
    """
    model.eval()
    all_features = {}
    n = len(image_data)
    
    for i in range(0, n, batch_size):
        # Convert each numpy image to a PIL image and apply preprocessing.
        batch_imgs = image_data[i:i+batch_size]
        batch_tensors = torch.stack([preprocess(Image.fromarray(img)) for img in batch_imgs]).to(device)
        with torch.no_grad():
            out = model(batch_tensors)
        
        if i == 0:
            # Initialize storage for each feature key.
            for key in out.keys():
                all_features[key] = []
        for key, feat in out.items():
            all_features[key].append(feat.cpu().numpy())
    
    # Concatenate results for each key.
    activations = {key: np.concatenate(val, axis=0) for key, val in all_features.items()}
    return activations

# Specify the layers we want to evaluate.
desired_layers = ["conv_pool_after_layer2", "conv_pool_after_layer_5",
                  "conv_pool_after_layer_12", "fc1", "fc2"]

# Define the models to be evaluated.
models = {
    'random': model_random,
    'imagenet': model_loaded,
    'barcode': model_barcode
}


##########################################
# 9. REGRESSION & EVALUATION             #
##########################################

# Dictionary to store productivity scores.
model_results = {}

# Get a list of brain areas from the fmri data dictionary.
brain_areas = list(train_fmri_data.keys())

# Define the desired order for brain areas on the x-axis.
desired_areas_order = ["V1", "V2", "V3", "V4", "ventral", "parietal", "lateral"]

# For each model, compute activations, fit Ridge regression, and compute normalized R2 scores.
for model_name, model_instance in models.items():
    print(f"\nProcessing model: {model_name}")
    features_train = get_model_activations(model_instance, train_image_data, batch_size=32)
    features_test  = get_model_activations(model_instance, test_image_data, batch_size=32)
    
    # Initialize dictionary to hold scores (rows: layers, columns: brain areas).
    scores = {layer: {} for layer in desired_layers}
    
    for layer in desired_layers:
        if layer not in features_train:
            print(f"Warning: Layer {layer} not found in model outputs. Skipping...")
            continue
        
        X_train = features_train[layer]  # (n_train, feature_dim)
        X_test  = features_test[layer]   # (n_test, feature_dim)
        print(f"Layer {layer}: train features {X_train.shape}, test features {X_test.shape}")
        
        for area in brain_areas:

            ### TODO: Implement Ridge regression and compute normalized R2 scores for this brain area.
            # Instructions:
            # 1. Extract the fMRI responses for training (y_train) and testing (y_test) for the current brain area
            #    from train_fmri_data and test_fmri_data respectively.
            # 2. Retrieve the noise ceiling values (ncsnr) for the test set.
            # 3. Create a RidgeCV model using sklearn.linear_model.RidgeCV with a list of alphas.
            #    Suggested alphas: [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000, 1e5, 1e6, 1e7, 1e7].
            # 4. Fit the Ridge regression model on X_train and y_train.
            # 5. Use the fitted model to predict responses (y_pred) on X_test.
            # 6. Print the optimal alpha selected and the R2 score on the test set.
            # 7. Compute the normalized R2 scores using the provided r2_over_nc function.
            # 8. Compute the average normalized R2 score across all voxels and store it in the scores dictionary.
            pass
    
    # Convert scores to a DataFrame (rows: layers, columns: brain areas).
    df_scores = pd.DataFrame(scores).T

    # Reorder the columns to the desired order.
    df_scores = df_scores[desired_areas_order]
    model_results[model_name] = df_scores
    
    # Plot a heatmap of the scores.
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_scores, annot=True, cmap="viridis", fmt=".3f")
    plt.title(f"Model Productivity (r2 over noise ceiling) - {model_name} model")
    plt.xlabel("Brain Area")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(f"heatmap_{model_name}.png", dpi=300)
    plt.close()
