# -*- coding: utf-8 -*-
__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"
__year__ = "2023"
__doc__ = """Quality Measures of an automatic segmentation computed from
a mask of the object (ground truth) 
Two types of measures are implemented:
    1. Volumetric (dice, voe, relvoldiff) compute differences and 
    similarities between the two volumes. They are similar to precision and
    recall.
    2. Distance-base (AvDist, MxDist) compare volume surfaces 
    in terms of distance between segmentation and ground truth.
    Average distances, AvDist, is equivalent to Euclidean distance between
    volumes, while Maximum distance, MxDist, is the infinite norm and detects
    puntual deviations between surfaces

References: 
    1. T. Heimann et al, Comparison and Evaluation of Methods for
Liver Segmentation From CT Datasets, IEEE Trans Med Imag, 28(8),2009

Computer Vision Center
Universitat Autonoma de Barcelona
Created on Sat Dec 15 12:09:57 2018
"""

# markdown explanation
"""
# Renaming Table
Old Name     | New Name
------------ | -------------
DICE         | dice_index
VOE          | volume_overlap_error
RelVolDiff   | relative_volume_difference
DistScores   | distance_scores
Seg          | segmentation
GT           | ground_truth
DistSegInt   | dist_seg_interior
DistSegExt   | dist_seg_exterior
DistSeg      | dist_seg
DistGTInt    | dist_truth_interior
DistGTExt    | dist_truth_exterior
DistGT       | dist_truth
BorderSeg    | border_seg
BorderGT     | border_truth
DistAll      | combined_distances
DistAvg      | - (used in return statement)
DistMx       | - (used in return statement)
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


def typed_distance_transform(segmentation: np.ndarray) -> np.ndarray:
    """
    Computes the distance transform of a binary image.

    Parameters:
    segmentation: Binary ndarray of segmentation

    Returns:
    Distance transform of the segmentation.
    """
    dist_map = distance_transform_edt(segmentation)
    assert isinstance(dist_map, np.ndarray)
    return dist_map


def dice_index(segmentation: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Computes Dice index between segmentation and ground truth mask.

    Parameters:
    segmentation: Binary ndarray of segmentation
    ground_truth: Binary ndarray of true object

    Returns:
    Dice index.
    """
    intersection = np.sum(segmentation[np.nonzero(ground_truth)])
    return 2.0 * intersection / (np.sum(segmentation) + np.sum(ground_truth))


def volume_overlap_error(segmentation: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Computes Volume Overlap Error (VOE) between segmentation and ground truth mask.

    Parameters:
    segmentation: Binary ndarray of segmentation
    ground_truth: Binary ndarray of true object

    Returns:
    Volume Overlap Error.
    """
    intersection = np.sum(segmentation * ground_truth)
    return 1 - 2 * intersection / (np.sum(segmentation) + np.sum(ground_truth))


def relative_volume_difference(
    segmentation: np.ndarray, ground_truth: np.ndarray
) -> float:
    """
    Computes Relative Volume Difference between segmentation and ground truth mask.

    Parameters:
    segmentation: Binary ndarray of segmentation
    ground_truth: Binary ndarray of true object

    Returns:
    Relative Volume Difference.
    """
    return (np.sum(segmentation) - np.sum(ground_truth)) / np.sum(segmentation)


def distance_scores(
    segmentation: np.ndarray, ground_truth: np.ndarray
) -> tuple[float, float]:
    """
    Computes Average and Maximum distances between segmentation and ground truth masks.

    Parameters:
    segmentation: Binary ndarray of segmentation
    ground_truth: Binary ndarray of true object

    Returns:
    A tuple containing the Average and Maximum distance.
    """
    dist_seg_interior = typed_distance_transform(segmentation)
    dist_seg_exterior = typed_distance_transform(1 - segmentation)
    dist_seg = np.maximum(dist_seg_interior, dist_seg_exterior)

    dist_truth_interior = typed_distance_transform(ground_truth)
    dist_truth_exterior = typed_distance_transform(1 - ground_truth)
    dist_truth = np.maximum(dist_truth_interior, dist_truth_exterior)

    border_seg = dist_seg_interior == 1
    border_truth = dist_truth_interior == 1

    combined_distances = np.concatenate(
        (dist_seg[border_truth], dist_truth[border_seg]), axis=0
    )

    return np.mean(combined_distances), np.max(combined_distances)
