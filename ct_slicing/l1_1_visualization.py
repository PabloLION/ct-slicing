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
from ct_slicing.get_data import nii_file
from ct_slicing.vis_lib.volume_cut_browser import (
    VolumeCutBrowser,
    CutDirection,
)
from ct_slicing.vis_lib.nifty_io import read_nifty

# choose the case id and nodule id to get the path of the nodule image and mask
INTENSITY_VOLUME_PATH, NODULE_MASK_PATH = nii_file("CT", 1, 1)

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
