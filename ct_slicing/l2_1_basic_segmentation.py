__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Volume Visualization and Lesion Segmentation / Segmentation
# Data: from Unit "Full Dataset"

import matplotlib.pyplot as plt
from ct_slicing.get_data import nii_file
from ct_slicing.vis_lib.nifty_io import read_nifty
from scipy.ndimage import gaussian_filter, median_filter
from skimage import morphology
from skimage.filters import threshold_otsu

from ct_slicing.vis_lib.volume_cut_browser import (
    CutDirection,
    VolumeCutBrowser,
)  # local

# choose the case id and nodule id to get the path of the nodule image and mask
INTENSITY_VOLUME_PATH, NODULE_MASK_PATH = nii_file("CT", 1, 1)
# Load Intensity Volume and Nodule Mask
nii_roi, nii_metadata = read_nifty(INTENSITY_VOLUME_PATH)


######## VISUALIZE VOLUMES

### Interactive Volume Visualization
# Short Axis Cuts
VolumeCutBrowser(nii_roi, CutDirection.ShortAxis)


######## SEGMENTATION PIPELINE

### 1. PRE-PROCESSING
# 1.1 Gaussian Filtering
sigma = 1
nii_roi_gauss = gaussian_filter(nii_roi, sigma=sigma)
# 1.2 MedFilter
size = 3
nii_roi_median = median_filter(nii_roi, size)

### 2. BINARIZATION / THRESHOLDING / PROCESSING

otsu_mask = threshold_otsu(nii_roi)
# #TODO: here this Otsu method is reading the
# histogram of all the images but the images do not have uniformed gray scale
# of their background. So the threshold is not accurate.
nii_roi_otsu = nii_roi > otsu_mask

# ROI Histogram
fig, ax = plt.subplots(1, 1)
ax.hist(nii_roi.flatten(), bins=50, edgecolor="k")

# Visualize Lesion Segmentation
# #TODO: seems the first two are wrong.
VolumeCutBrowser(nii_roi, CutDirection.ShortAxis, contour_stack=nii_roi_otsu)
VolumeCutBrowser(nii_roi, CutDirection.ShortAxis, contour_stack=nii_roi_gauss)
VolumeCutBrowser(nii_roi, CutDirection.ShortAxis, contour_stack=nii_roi_median)


### 3.POST-PROCESSING

# 3.1  Opening
size_open = 3
se = morphology.cube(size_open)
nii_roi_seg_open = morphology.binary_opening(nii_roi_otsu, se)

# 3.2  Closing
size_close = 3
se = morphology.cube(size_close)
nii_roi_seg_close = morphology.binary_closing(nii_roi_otsu, se)
