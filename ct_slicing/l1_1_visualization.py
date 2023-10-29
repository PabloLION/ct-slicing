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

from ct_slicing.log import logger
from vis_lib.VolumeCutBrowser import VolumeCutBrowser, CutDirection  # local
from vis_lib.NiftyIO import read_nifty  # local
from ct_slicing.config.data_path import DATA_FOLDER  # local

#### Data Folders (change to your path)
CASE_FOLDER = "CT"
INTENSITY_VOLUME_NAME = "LIDC-IDRI-0001.nii.gz"
NODULE_MASK = "LIDC-IDRI-0001_R_1.nii.gz"
INTENSITY_VOLUME_PATH = DATA_FOLDER / CASE_FOLDER / "image" / INTENSITY_VOLUME_NAME
NODULE_MASK_PATH = DATA_FOLDER / CASE_FOLDER / "nodule_mask" / NODULE_MASK

logger.debug(f"DATA_FOLDER: {DATA_FOLDER}")
logger.debug(f"INTENSITY_VOLUME_PATH: {INTENSITY_VOLUME_PATH}")

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
VolumeCutBrowser(nii_vol, CutDirection.ShortAxis)
# Coronal View
VolumeCutBrowser(nii_vol, cut_dir=CutDirection.Coronal)
# Sagittal View
VolumeCutBrowser(nii_vol, cut_dir=CutDirection.Sagittal)
# The lesion mask can be added as a contour
VolumeCutBrowser(nii_vol, CutDirection.ShortAxis, contour_stack=nii_mask)
