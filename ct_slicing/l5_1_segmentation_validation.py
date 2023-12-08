__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""
# Unit: Segmentation Validation / Basic Validation (Segmentation)

"""
## Problem

Marked problem (*.) with #PR* in code comments like `#PR1` means (1.).
Problem 2 has and 4 has no code comments.

1. Why sometimes it's all 0
2. Otsu threshold is applied to the whole volume, but we are only interested 
in one slice. This might cause the threshold to be inaccurate.
3. In the code `SegRelDiff_SA = RelVolDiff(niiROISeg, niiROIGT)` was 
    copied and repeated, but not changed. Looks like a typo.
4. What is distance map?
5. Why is there another contour that we didn't use? What if they are connected
    in 3D but not in 2D?

## Terminology and Abbreviation:
- GT: Ground Truth
- VOE: Volumetric Overlap Error
- DICE: Dice Similarity Coefficient
- RelVolDiff: Relative Volume Difference
- DistSegInt: Distance Map inside Segmentation
- DistSegExt: Distance Map outside Segmentation
- DistSeg: Distance Map at all points
- BorderGT: GT Mask boundary points
- AvgDist: Average Distance

## Renaming

| Renamed to             | Original Name | Meaning                                       |
| ---------------------- | ------------- | --------------------------------------------- |
| nii_otsu               | niiROISeg     | Segmentation result with otsu threshold       |
| sagittal_cut           | SA            | Sagittal cut                                  |
| sagittal_truth         | SAGT          | Ground truth                                  |
| slice_index            | k             | index of the slice we are focusing on         |
| sagittal_otsu          | SAseg         | Segmentation result with otsu on sagittal cut |
| whole_voe              | SegVOE        | VOE of the whole volume                       |
| whole_dice             | SegDICE       | DICE of the whole volume                      |
| whole_rel_diff         | SegRelDiff    | RelVolDiff of the whole volume                |
| sagittal_voe           | SegVOE_SA     | VOE of the sagittal cut                       |
| sagittal_dice          | SegDICE_SA    | DICE of the sagittal cut                      |
| sagittal_rel_diff      | SegRelDiff_SA | RelVolDiff of the sagittal cut                |
| interior_sagittal_dist | DistSegInt    | Distance Map inside Segmentation (?)          |
| exterior_sagittal_dist | DistSegExt    | Distance Map outside Segmentation (?)         |
| otsu_sagittal_dist     | DistSeg       | Distance Map at all points (?)                |
| borders_truth          | BorderGT      | Ground Truth Mask boundary points             |
| border_ys              | i             | y values of border_truth                      |
| border_xs              | j             | x values of border_truth                      |
| average_dist           | AvgDist       | AvgDist                                       |
| max_dist               | MxDist        | MaxDist                                       |
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import filters as filt
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from skimage import morphology as Morpho
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from vis_lib.NiftyIO import read_nifty
from vis_lib.VolumeCutBrowser import VolumeCutBrowser

from ct_slicing.config.data_path import DATA_FOLDER
from ct_slicing.vis_lib.segmentation_quality_scores import (
    DICE,
    VOE,
    DistScores,
    RelVolDiff,
)

# 1. Load Intensity Volume
IMG_PATH = DATA_FOLDER / "CT" / "image" / "LIDC-IDRI-0001.nii.gz"
MASK_PATH = DATA_FOLDER / "CT" / "nodule_mask" / "LIDC-IDRI-0001_R_1.nii.gz"

nii_roi, _ = read_nifty(IMG_PATH)  # ROI(region of interest) from nii file
nii_truth, _ = read_nifty(MASK_PATH)  # ground truth of ROI from nii file


# 2. VOLUME SEGMENTATION
otsu_threshold = threshold_otsu(nii_roi)
nii_otsu = nii_roi > otsu_threshold  # segmentation result with otsu threshold

# 3. VALIDATION SCORES
# preparation: Axial Cut, here with sagittal direction
# to fucus on only one slice of the volume, find a proper slice_index (was `k`)

# find the slice with the most non-zero pixels in the ground truth
slice_index = np.argmax(np.sum(nii_truth, axis=(0, 1)))
# wrong: with middle cut, sagittal_truth is all 0: k = int(nii_roi.shape[2] / 2)
# TODO: #PR1 check why it is wrong, why all 0?

# get the only slice under our focus
sagittal_cut = nii_roi[:, :, slice_index]  # the raw image
sagittal_truth = nii_truth[:, :, slice_index]  # the ground truth
sagittal_otsu = nii_otsu[:, :, slice_index]  # the segmentation result
assert (  # make sure the truth is not all 0
    np.all(sagittal_truth == 0) == False
), f"sagittal_truth is all 0? {np.all(sagittal_truth == 0)}"


# 3.1 Visualize GT contours over sagittal_cuts
plt.figure()
plt.imshow(sagittal_cut, cmap="gray")
plt.contour(sagittal_truth, [0.5], colors="r")
# will do plt.show() in 3.3.2

# 3.2 Volumetric Measures
# measures for the whole volume
whole_voe = VOE(nii_otsu, nii_truth)
volume_dice = DICE(nii_otsu, nii_truth)
volume_rel_diff = RelVolDiff(nii_otsu, nii_truth)

# measures for the sagittal_cuts cut
sagittal_voe = VOE(sagittal_otsu, sagittal_truth)
sagittal_dice = DICE(sagittal_otsu, sagittal_truth)
sagittal_rel_diff = RelVolDiff(sagittal_otsu, sagittal_truth)  # #PR3: typo?

# 3.3 Distance Measures
# 3.3.1 Distance Map to Otsu Segmentation sagittal_cuts cut
interior_sagittal_dist = bwdist(sagittal_otsu)  # Distance Map inside Segmentation
exterior_sagittal_dist = bwdist(1 - sagittal_otsu)  # Distance Map outside Segmentation
otsu_sagittal_dist = np.maximum(
    interior_sagittal_dist, exterior_sagittal_dist
)  # Distance Map at all points

# 3.3.2 Distance from GT to Otsu Segmentation
# GT Mask boundary points
borders_truth = find_contours(sagittal_truth, 0.5)  # find a contour of the truth
assert len(borders_truth) == 2, f"len(borders_truth) = {len(borders_truth)}"
# #TODO: #PR5: why 2 contours? seems there's another contour that we didn't use.
border_truth, border_another, *_ = borders_truth
plt.plot(border_truth[:, 1], border_truth[:, 0], linestyle="dotted", color="y")
plt.plot(border_another[:, 1], border_another[:, 0], linewidth=2, color="b")
plt.show()  # showing the plot from 3.1
border_ys = border_truth[:, 0].astype(int)
border_xs = border_truth[:, 1].astype(int)

# Show histogram
plt.figure()
plt.hist(otsu_sagittal_dist[border_ys, border_xs], bins=50, edgecolor="k")
plt.show()

# 3.3.3 Distance Scores
average_dist, max_dist = DistScores(sagittal_otsu, sagittal_truth)
print(f"Average Distance: {average_dist}")
print(f"Max Distance: {max_dist}")
