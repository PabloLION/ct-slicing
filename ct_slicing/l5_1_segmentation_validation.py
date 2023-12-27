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
    - Distance map is the Euclidean distance from a point to the nearest 0 or 
    zero-valued element in the array.
5. Why is there another contour that we didn't use? What if they are connected
    in 3D but not in 2D?
6. Colors in different slices does not have a same distribution. For example,
    some slices has a lighter background than others. We should normalize the
    colors in the whole volume, instead of in each slice, to make all the
    background colors have the same intensity.

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

if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

from typing import Callable, cast
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import distance_transform_edt as backward_dist
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from ct_slicing.config.dev_config import DEFAULT_PLOT_BLOCK
from ct_slicing.data_util.nii_file_access import nii_file
from ct_slicing.vis_lib.nifty_io import read_nifty

from ct_slicing.vis_lib.segmentation_quality_scores import (
    dice_index,
    volume_overlap_error,
    distance_scores,
    relative_volume_difference,
)

# choose the case id and nodule id to get the path of the nodule image and mask
IMG_PATH, MASK_PATH = nii_file("CT", 1, 1)

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
# will do plt.show in 3.3.2

# 3.2 Volumetric Measures
# measures for the whole volume
whole_voe = volume_overlap_error(whole_otsu, whole_truth)
volume_dice = dice_index(whole_otsu, whole_truth)
volume_rel_diff = relative_volume_difference(whole_otsu, whole_truth)

# measures for the slice_cuts cut
slice_voe = volume_overlap_error(slice_otsu, slice_truth)
slice_dice = dice_index(slice_otsu, slice_truth)
slice_rel_diff = relative_volume_difference(slice_otsu, slice_truth)  # #PR3: typo?

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
plt.show(block=DEFAULT_PLOT_BLOCK)  # showing the plot from 3.1
border_ys = border_truth[:, 0].astype(int)
border_xs = border_truth[:, 1].astype(int)

# Show histogram
plt.show(block=False)  # flush the plot
plt.figure()
plt.hist(otsu_slice_dist[border_ys, border_xs], bins=50, edgecolor="k")
plt.show(block=DEFAULT_PLOT_BLOCK)

# 3.3.3 Distance Scores
average_dist, max_dist = distance_scores(slice_otsu, slice_truth)
print(f"Average Distance: {average_dist}")
print(f"Max Distance: {max_dist}")


# 4 Exercises
# 4.2 Exercise 2
# 4.2.2 Exercise 2b
# Applying bit operation & on the two masks gives the intersection of them.
def intersection(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Return the intersection of two masks, with bit operation &."""
    return mask1 & mask2


# To visualize the intersection, plot mask 1 in transparent red and mask 2 in
# transparent green, then show a blue contour of the intersection.
def vis_compare(
    truth_mask: np.ndarray, prediction_mask: np.ndarray, comparator: Callable
) -> None:
    """Visualize the intersection of two masks."""

    plt.figure()
    plt.imshow(truth_mask, cmap="Reds")
    plt.title("Truth Mask")
    plt.show(block=DEFAULT_PLOT_BLOCK)
    plt.imshow(prediction_mask, cmap="Greens")
    plt.title("Prediction Mask")
    plt.show(block=DEFAULT_PLOT_BLOCK)
    plt.title("Truth in Red, Prediction in Green, Comparison in Blue Contour")
    comparison_mask = comparator(truth_mask, prediction_mask)
    plt.imshow(truth_mask, cmap="Reds", alpha=0.5)
    plt.imshow(prediction_mask, cmap="Greens", alpha=0.5)
    plt.contour(comparison_mask, [0.5], colors="b")
    plt.show(block=DEFAULT_PLOT_BLOCK)


vis_compare(slice_truth, slice_otsu, intersection)


# 4.2.3 Exercise 2c
# Applying bit operation ^ on the two masks gives the difference of them.
def difference(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Return the difference of two masks, with bit operation ^."""
    return mask1 ^ mask2


vis_compare(slice_truth, slice_otsu, difference)


# 4.3 Exercise 3
plt.show(block=False)  # flush the plot

# 4.3.1 show the intensity image and the binary Otsu segmentation side by side
plt.subplot(1, 2, 1)
plt.imshow(slice_cut, cmap="gray")
plt.title("Intensity Image")

plt.subplot(1, 2, 2)
plt.imshow(slice_otsu, cmap="gray")
plt.title("Otsu Segmentation")
plt.show(block=DEFAULT_PLOT_BLOCK)

# 4.3.2  Distance map to red contours (boundary of segmentation), the brighter
# the distance image is, the further a pixel is from red curves.
plt.imshow(otsu_slice_dist, cmap="gray")
plt.contour(slice_otsu, [0.5], colors="r")
plt.title("Distance Map with Red Contours")
plt.show(block=DEFAULT_PLOT_BLOCK)
"""
## Exercises


### Exercise 1

Exercise 1. Ground Truth. Visualize the contours of the ground truth mask over the
original ROI volume in the middle SA cut (code in Script_SegmentationValidation.py)
and using VolumeCutBrowser. Modify the code in Script_SegmentationValidation.py
in order to visualize the contours of the ground truth mask over the original ROI volume
in the middle Sagittal and Coronal cuts.

    We visualized another slice `slice_index` instead of the middle one, 
    because the middle one's ground truth is all 0.
    The visualization is with the red solid line, from code in in 3.1, and the 
    plot is shown in 3.3.2, together with a yellow dotted line for the Otsu
    segmentation counter and a blue solid line for another negligible contour 
    in problem 5.


### Exercise 2

Exercise 2. Visual Comparison.
a) Edit the code in Script_SegmentationValidation.py to visualize Otsu segmented
contours in the same image that visualizes ground truth contours in the middle
SA cut. Repeat for different SA cuts and also for sagittal and coronal cuts.

The visualization from code in in 3.1, and 3.3.2, shows both contours in the
same image:
    - The red solid line for the ground truth
    - The yellow dotted line for the Otsu segmentation
    - The blue solid line for another negligible contour in problem 5.

b) Compute the intersection of the ground truth and Otsu segmentation and
visualize it over the original ROI in the SA cuts of Exercise 2a).
    The intersection is visualized in 4.3.2, with the blue contour. 
    In the visualization, we can see the Otsu segmentation is a super set of
    the ground truth, because the intersection is the same as the ground truth.

c) Compute the difference between ground truth and Otsu segmentation and
visualize it over the original ROI volume the original ROI in the SA cuts of
Exercise 2a).
    With the code in 4.2.3, we can see the difference between the ground truth
    and the Otsu segmentation is the same as the the complement set of the 
    ground truth regarding Otsu segmentation, because the Otsu segmentation is
    a super set of the ground truth.
    The difference is visualized in 4.2.3, with the blue contour, and filled
    with "green or red but not both" color.
    
Exercise 4. Distance Measures.
Compute the distance map to the ground truth mask at the middle SA cut. Visualize
the distance map and plot the mask contours on it in red as shown in figure2 for Otsu
segmentation of ROI shown in figure1. Place the cursor on the image to understand
the meaning of a distance map.
The distance map evaluated at Otsu segmentation boundary points gives a list of
values, one for each point belonging to the segmentation border curve. Compute Otsu
boundary points using skimage function find_contours. Plot the histogram of the
distance map to the ground truth evaluated at Otsu boundary points. Compare the
histogram to the values returned by the function DistScores
    The binary Otsu segmentation and the intensity image are shown side by side
    is visualized in 4.3.1.
    The distance map is visualized in 4.3.2, with the red contour.

Exercise 5. Computation of Segmentation Scores.
a) Implement a script to segment all cases and save the segmentations in .nii files
with names identifying the case. Consider several options for structuring the
saved data: one file for each segmented ROI?; one file for each case storing all
its segmented ROIs?; one single file for all segmentations?
    We can use python's pickle (`pickle.dump()`) to save the segmented ROIs in
    whatever structure we want, and use python's pickle (`pickle.load()`) to
    load them back like in l5_2_k_fold_sample_code.py.
    
    To save a file compatible with l5_2_k_fold_sample_code.py, we can use
    data_dict = {
        'slice_features': slice_features, # pretend `slice_features` exists
        'slice_meta': slice_meta, # pretend `slice_meta` exists
    }
    with open(SLICE_FEATURES_PATH, 'wb') as file:
        pickle.dump(data_dict, file)

    However, on the safety aspect, pickle is not doing well. We can also use
    numpy's `np.save()` to save the segmented ROIs in a .npz file, and use
    `np.load()` to load them back like in l5_2_k_fold_sample_code.py.

b) Implement a script to load all segmented cases together with their ground truth
masks, compute their validation scores and save the scores.
    We can use python's pickle to save the segmented ROIs in one file, and use
    python's pickle to load them back.
    
    The code is in l5_2_k_fold_sample_code.py, with the following line:
    `np.load(SLICE_FEATURES_PATH, allow_pickle=True)`
"""
