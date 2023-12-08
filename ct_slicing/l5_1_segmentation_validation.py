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

| Renamed to          | Original Name | Meaning                                    |
| ------------------- | ------------- | ------------------------------------------ |
| nii_otsu            | niiROISeg     | Segmentation result with otsu threshold    |
| slice_cut           | SA            | slice cut                                  |
| slice_truth         | SAGT          | Ground truth                               |
| slice_index         | k             | index of the slice we are focusing on      |
| slice_otsu          | SAseg         | Segmentation result with otsu on slice cut |
| whole_voe           | SegVOE        | VOE of the whole volume                    |
| whole_dice          | SegDICE       | DICE of the whole volume                   |
| whole_rel_diff      | SegRelDiff    | RelVolDiff of the whole volume             |
| slice_voe           | SegVOE_SA     | VOE of the slice cut                       |
| slice_dice          | SegDICE_SA    | DICE of the slice cut                      |
| slice_rel_diff      | SegRelDiff_SA | RelVolDiff of the slice cut                |
| interior_slice_dist | DistSegInt    | Distance Map inside Segmentation (?)       |
| exterior_slice_dist | DistSegExt    | Distance Map outside Segmentation (?)      |
| otsu_slice_dist     | DistSeg       | Distance Map at all points (?)             |
| borders_truth       | BorderGT      | Ground Truth Mask boundary points          |
| border_ys           | i             | y values of border_truth                   |
| border_xs           | j             | x values of border_truth                   |
| average_dist        | AvgDist       | AvgDist                                    |
| max_dist            | MxDist        | MaxDist                                    |
"""

from typing import cast
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import filters, distance_transform_edt as backward_dist
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from vis_lib.NiftyIO import read_nifty

from ct_slicing.config.data_path import DATA_FOLDER
from ct_slicing.vis_lib.segmentation_quality_scores import (
    DICE,
    VOE,
    DistScores,
    RelVolDiff,
)

""" 1. Load Intensity Volume """
IMG_PATH = DATA_FOLDER / "CT" / "image" / "LIDC-IDRI-0001.nii.gz"
MASK_PATH = DATA_FOLDER / "CT" / "nodule_mask" / "LIDC-IDRI-0001_R_1.nii.gz"

nii_roi, _ = read_nifty(IMG_PATH)  # ROI(region of interest) from nii file
whole_truth, _ = read_nifty(MASK_PATH)  # ground truth of ROI for the whole volume

""" 2. VOLUME SEGMENTATION """
otsu_threshold = threshold_otsu(nii_roi)
whole_otsu = nii_roi > otsu_threshold  # segmentation result with otsu threshold

""" 3. VALIDATION SCORES """
# To fucus on only one slice of the volume, find a proper slice_index (was `k`)
# the slice names are with `slice_` prefix instead of `SA` or `coronal`, to
# be more concise and consistent if we want to use other directions.
# Here we are using the Axial Cut, AKA Short Axis direction.

# find the slice with the most non-zero pixels in the ground truth
slice_index = np.argmax(np.sum(whole_truth, axis=(0, 1)))
# wrong: with middle cut, slice_truth is all 0: k = int(nii_roi.shape[2] / 2)
# TODO: #PR1 check why it is wrong, why all 0?

# get the only slice under our focus
slice_cut = nii_roi[:, :, slice_index]  # the raw image
slice_truth = whole_truth[:, :, slice_index]  # the ground truth
slice_otsu = whole_otsu[:, :, slice_index]  # the segmentation result
assert (  # make sure the truth is not all 0
    np.all(slice_truth == 0) == False
), f"slice_truth is all 0? {np.all(slice_truth == 0)}"


# 3.1 Visualize GT contours over slice_cuts
plt.figure()
plt.imshow(slice_cut, cmap="gray")
plt.contour(slice_truth, [0.5], colors="r")
# will do plt.show() in 3.3.2

# 3.2 Volumetric Measures
# measures for the whole volume
whole_voe = VOE(whole_otsu, whole_truth)
volume_dice = DICE(whole_otsu, whole_truth)
volume_rel_diff = RelVolDiff(whole_otsu, whole_truth)

# measures for the slice_cuts cut
slice_voe = VOE(slice_otsu, slice_truth)
slice_dice = DICE(slice_otsu, slice_truth)
slice_rel_diff = RelVolDiff(slice_otsu, slice_truth)  # #PR3: typo?

# 3.3 Distance Measures
# 3.3.1 Distance Map to Otsu Segmentation slice_cuts cut
# Distance Map inside and outside Segmentation for dist map at all points
interior_slice_dist = cast(np.ndarray, backward_dist(slice_otsu))
exterior_slice_dist = cast(np.ndarray, backward_dist(1 - slice_otsu))
otsu_slice_dist = np.maximum(interior_slice_dist, exterior_slice_dist)

# 3.3.2 Distance from Ground Truth to Otsu Segmentation
# Ground Truth Mask boundary points
borders_truth = find_contours(slice_truth, 0.5)  # find a contour of the truth
assert len(borders_truth) == 2, f"len(borders_truth) = {len(borders_truth)}"
# #TODO: #PR5: why 2 contours? seems there's another contour that we don't need
border_truth, border_another, *_ = borders_truth
plt.plot(border_truth[:, 1], border_truth[:, 0], linestyle="dotted", color="y")
plt.plot(border_another[:, 1], border_another[:, 0], linewidth=2, color="b")
plt.show()  # showing the plot from 3.1
border_ys = border_truth[:, 0].astype(int)
border_xs = border_truth[:, 1].astype(int)

# Show histogram
plt.figure()
plt.hist(otsu_slice_dist[border_ys, border_xs], bins=50, edgecolor="k")
plt.show()

# 3.3.3 Distance Scores
average_dist, max_dist = DistScores(slice_otsu, slice_truth)
print(f"Average Distance: {average_dist}")
print(f"Max Distance: {max_dist}")
