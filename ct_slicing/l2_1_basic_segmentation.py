__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Segmentation
# Data: from Unit "Full Dataset"

import matplotlib.pyplot as plt
from ct_slicing.config.data_path import DATA_FOLDER
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
sigma = 1
nii_roi_gauss = filters.gaussian_filter(nii_roi, sigma=sigma)
# 1.2 MedFilter
size = 3
nii_roi_median = filters.median_filter(nii_roi, size)

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
se = Morpho.cube(size_open)
nii_roi_seg_open = Morpho.binary_opening(nii_roi_otsu, se)

# 3.2  Closing
size_close = 3
se = Morpho.cube(size_close)
nii_roi_seg_close = Morpho.binary_closing(nii_roi_otsu, se)
