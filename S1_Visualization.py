"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

from pathlib import Path
import matplotlib.pyplot as plt
import os
from VolumeCutBrowser import VolumeCutBrowser, VolumeCutDirection
from NiftyIO import read_nifty

#### Data Folders (change to your path)
DATA_FOLDER = Path(__file__).parent / "data"
CASE_FOLDER = "CT"
INTENSITY_VOLUME_NAME = "LIDC-IDRI-0001.nii.gz"
NODULE_MASK = "LIDC-IDRI-0001_R_1.nii.gz"
INTENSITY_VOLUME_PATH = DATA_FOLDER / CASE_FOLDER / "image" / INTENSITY_VOLUME_NAME
NODULE_MASK_PATH = DATA_FOLDER / CASE_FOLDER / "nodule_mask" / NODULE_MASK


# Load Intensity Volume and Nodule Mask
nii_vol, nii_metadata = read_nifty(INTENSITY_VOLUME_PATH)
nii_mask, nii_mask_metadata = read_nifty(NODULE_MASK_PATH)
assert nii_vol.shape == nii_mask.shape, "Volume and mask must have the same shape"

# VOLUME METADATA
print("Voxel Resolution (mm): ", nii_metadata.spacing)
print("Volume origin (mm): ", nii_metadata.origin)
print("Axes direction: ", nii_metadata.direction)

# Interactive Volume Visualization
# Short Axis View
VolumeCutBrowser(nii_vol, VolumeCutDirection.ShortAxis)
# Coronal View
VolumeCutBrowser(nii_vol, cut_dir=VolumeCutDirection.Coronal)
# Sagittal View
VolumeCutBrowser(nii_vol, cut_dir=VolumeCutDirection.Sagittal)
# #BUG next line:  with contour_stack=nii_mask, no mask is shown
VolumeCutBrowser(nii_vol, VolumeCutDirection.ShortAxis, contour_stack=nii_vol)
