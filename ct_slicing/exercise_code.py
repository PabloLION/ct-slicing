from typing import cast
from radiomics import featureextractor as feature_extractor, setVerbosity
import radiomics
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

from ct_slicing.config.data_path import REPO_ROOT
from ct_slicing.data_util.data_access import nii_file
from ct_slicing.ct_logger import logger

# to fix wrong implementation of radiomics.setVerbosity(60)
import logging

logging.getLogger("radiomics").setLevel(logging.CRITICAL)  # run radiomics quietly
logging.getLogger("pykwalify").setLevel(logging.CRITICAL)  # pykwalify from radiomics


# Load your mask (assuming it's a binary mask)m
img_path, mask_path = nii_file("CT", 1, 1)
img = sitk.ReadImage(img_path)
mask = sitk.ReadImage(mask_path)


# l1.1 Visualization
# Ex 1.1.2.a Bounding Boxes
radiomics_params = str(
    REPO_ROOT / "ct_slicing" / "config" / "FeaturesExtraction_Params.yaml"
)  # using another params will cause error
# #TODO: ref params
# Calculate bounding boxes
bbox = radiomics.imageoperations.checkMask(img, mask, label=1)[0]
logger.info(f"Ex1.1.2.a Bounding Boxes:\n{bbox}")
bbox = cast(np.ndarray, bbox)

# 1.1.2.b affine transformation


# Get the affine matrix from the image's metadata
affine_matrix = np.array(img.GetDirection()).reshape(3, 3)
# The affine_matrix now contains the 3x3 rotation matrix from the image's metadata

# Define the bounding box in world coordinates (physical coordinates)
# Replace these values with the actual bounding box coordinates from your metadata
bounding_box_world = [bbox[:3], bbox[3:]]

# Invert the affine matrix to get the transformation from voxel to world coordinates
affine_matrix_inv = np.linalg.inv(affine_matrix)

# Convert the bounding box coordinates to voxel coordinates
bounding_box_voxel = []
for corner in bounding_box_world:
    corner_homogeneous = np.array(corner + (1.0,))
    corner_voxel_homogeneous = np.dot(affine_matrix_inv, corner_homogeneous)
    corner_voxel = corner_voxel_homogeneous[:3]
    # Extract the first three values (x, y, z)
    bounding_box_voxel.append(tuple(corner_voxel))

logger.info(f"Ex1.1.2.b Bounding Box in Voxel Coordinates:\n{bounding_box_voxel}")

# Ex 1.1.3

# Apply thresholding to segment the lesion
arbitrary_slice = img[:, :, 90]
lower_threshold, upper_threshold = -20, 200  # using the value from Ex 1.1.a
segmented_lesion = sitk.BinaryThreshold(
    arbitrary_slice, lowerThreshold=lower_threshold, upperThreshold=upper_threshold
)
# Convert the segmented lesion to a NumPy array for visualization
segmented_lesion_np = sitk.GetArrayFromImage(segmented_lesion)
arbitrary_slice_np = sitk.GetArrayFromImage(arbitrary_slice)

# Visualize the segmented lesion
plt.subplot(1, 3, 1)
plt.imshow(arbitrary_slice_np, cmap="Blues")
plt.title("Original Image in blue")

plt.subplot(1, 3, 2)
plt.imshow(segmented_lesion_np, cmap="Oranges", alpha=0.8)
plt.imshow(arbitrary_slice_np, cmap="Blues", alpha=0.8)
plt.title("Both in blue and orange")

plt.subplot(1, 3, 3)
plt.imshow(segmented_lesion_np, cmap="Oranges")
plt.title("Segmented Lesion in orange")
plt.show()


extractor = feature_extractor.RadiomicsFeatureExtractor(radiomics_params)
extractor.loadImage(img, mask)
result = extractor.execute(img, mask, label=1)
