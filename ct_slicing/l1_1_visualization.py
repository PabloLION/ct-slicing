__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Data Exploration / Volume Visualization
# Data: from "LUNA Dataset / Full Dataset"

from ct_slicing.ct_logger import logger
from ct_slicing.vis_lib.volume_cut_browser import (
    VolumeCutBrowser,
    CutDirection,
)  # local
from ct_slicing.vis_lib.nifty_io import read_nifty  # local
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
assert (
    nii_vol.shape == nii_mask.shape
), f"Volume shape {nii_vol.shape} != Mask shape {nii_mask.shape}"

# VOLUME METADATA
logger.info(f"Voxel Resolution (mm): {nii_metadata.spacing}")
logger.info(f"Volume origin (mm): {nii_metadata.origin}")
logger.info(f"Axes direction: {nii_metadata.direction}")

# Interactive Volume Visualization
# Short Axis View
VolumeCutBrowser(nii_vol, CutDirection.ShortAxis)
# Coronal View
VolumeCutBrowser(nii_vol, cut_dir=CutDirection.Coronal)
# Sagittal View
VolumeCutBrowser(nii_vol, cut_dir=CutDirection.Sagittal)
# The lesion mask can be added as a contour
VolumeCutBrowser(nii_vol, CutDirection.ShortAxis, contour_stack=nii_mask)
