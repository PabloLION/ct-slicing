__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
from config import DATA_FOLDER
from vis_lib.NiftyIO import read_nifty
from scipy.ndimage import filters
from skimage import morphology as Morpho
from skimage.filters import threshold_otsu

from vis_lib.VolumeCutBrowser import CutDirection, VolumeCutBrowser  # local


CASE_FOLDER = "CT"
INTENSITY_VOLUME_NAME = "LIDC-IDRI-0001.nii.gz"
NODULE_MASK = "LIDC-IDRI-0001_R_1.nii.gz"
INTENSITY_VOLUME_PATH = DATA_FOLDER / CASE_FOLDER / "image" / INTENSITY_VOLUME_NAME
NODULE_MASK_PATH = DATA_FOLDER / CASE_FOLDER / "nodule_mask" / NODULE_MASK

# Load Intensity Volume and Nodule Mask
nii_roi, nii_metadata = read_nifty(INTENSITY_VOLUME_PATH)


######## VISUALIZE VOLUMES

### Interactive Volume Visualization
# Short Axis Cuts
VolumeCutBrowser(nii_roi, CutDirection.ShortAxis)


######## SEGMENTATION PIPELINE

### 1. PRE-PROCESSING
# 1.1 Gaussian Filtering
sig = 1
niiROIGauss = filters.gaussian_filter(nii_roi, sigma=sig)
# 1.2 MedFilter
sze = 3
niiROIMed = filters.median_filter(nii_roi, sze)

###

### 2. BINARIZATION
Th = threshold_otsu(nii_roi)
niiROISeg = nii_roi > Th
# ROI Histogram
fig, ax = plt.subplots(1, 1)
ax.hist(nii_roi.flatten(), bins=50, edgecolor="k")
# Visualize Lesion Segmentation
VolumeCutBrowser(nii_roi, CutDirection.ShortAxis, contour_stack=niiROISeg)
VolumeCutBrowser(nii_roi, CutDirection.ShortAxis, contour_stack=niiROIGauss)
VolumeCutBrowser(nii_roi, CutDirection.ShortAxis, contour_stack=niiROIMed)


### 3.POST-PROCESSING

# 3.1  Opening
szeOp = 3
se = Morpho.cube(szeOp)
niiROISegOpen = Morpho.binary_opening(niiROISeg, se)

# 3.2  Closing
szeCl = 3
se = Morpho.cube(szeCl)
niiROISegClose = Morpho.binary_closing(niiROISeg, se)
