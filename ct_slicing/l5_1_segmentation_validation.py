"""
This is a sample source code for basic validation of a segmentation method

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

# Unit: Segmentation Validation / Basic Validation (Segmentation)

import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from skimage.measure import find_contours as contour
from ct_slicing.config.data_path import DATA_FOLDER

from vis_lib.NiftyIO import read_nifty
from vis_lib.VolumeCutBrowser import VolumeCutBrowser
from ct_slicing.vis_lib.segmentation_quality_scores import (
    RelVolDiff,
    VOE,
    DICE,
    DistScores,
)


######## 1. LOAD DATA

#### Data Folders
IMG_PATH = DATA_FOLDER / "CT" / "image" / "LIDC-IDRI-0001.nii.gz"
MASK_PATH = DATA_FOLDER / "CT" / "nodule_mask" / "LIDC-IDRI-0001_R_1.nii.gz"

#### Load Intensity Volume
niiROI, _ = read_nifty(IMG_PATH)
niiROIGT, _ = read_nifty(MASK_PATH)


### 2. VOLUME SEGMENTATION
Th = threshold_otsu(niiROI)
niiROISeg = niiROI > Th

### 3. VALIDATION SCORES
# Axial Cut
# k = int(niiROI.shape[2] / 2) # Wrong: SAGT is all 0
# Cut at the middle of the volume. Change k to get other cuts


sums = np.sum(niiROIGT, axis=(0, 1))
k = np.argmax(sums)  # I added this line other wize SAGT would be all 0


SA = niiROI[:, :, k]
SAGT = niiROIGT[:, :, k]

assert np.all(SAGT == 0) == False, f"SAGT is all 0? {np.all(SAGT == 0)}"


SASeg = niiROISeg[:, :, k]

# 3.1 Visualize GT contours over SA
fig = plt.figure()
plt.imshow(SA, cmap="gray")
plt.contour(SAGT, [0.5], colors="r")

# 3.2 Volumetric Measures
SegVOE = VOE(niiROISeg, niiROIGT)
SegDICE = DICE(niiROISeg, niiROIGT)
SegRelDiff = RelVolDiff(niiROISeg, niiROIGT)

SegVOE_SA = VOE(SASeg, SAGT)
SegDICE_SA = DICE(SASeg, SAGT)
SegRelDiff_SA = RelVolDiff(niiROISeg, niiROIGT)

# 3.3 Distance Measures
# 3.3.1 Distance Map to Otsu Segmentation SA cut
DistSegInt = bwdist(SASeg)  # Distance Map inside Segmentation
DistSegExt = bwdist(1 - SASeg)  # Distance Map outside Segmentation
DistSeg = np.maximum(DistSegInt, DistSegExt)  # Distance Map at all points

# 3.3.2 Distance from GT to Otsu Segmentation
# GT Mask boundary points
BorderGT = contour(SAGT, 0.5)
i = BorderGT[0][:, 0].astype(int)
j = BorderGT[0][:, 1].astype(int)

# Show histogram
fig = plt.figure()
plt.hist(DistSeg[i, j], bins=50, edgecolor="k")
plt.show()

# 3.3.3 Distance Scores
AvgDist, MxDist = DistScores(SASeg, SAGT)
