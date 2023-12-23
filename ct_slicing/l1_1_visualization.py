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
from ct_slicing.data_util.nii_file_access import nii_file
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

"""
Exercises:

Exercise 1. Visualization of 3D Volumes. Download the materials to a local folder,
launch Spyder 5 from Desktop and open S1_Visualization.py. Set the variables
SessionPyFolder, SessionDataFolder to your local path. Otherwise, use Slicer3D
available at https://www.slicer.org.
a) Use the interactive visualization routines to browse though the volumes. What
are intensity values for the lungs, soft tissue, bones and pulmonary lesion?
    lungs: shown as dark gray in the image, with value range (-800, -1050)
    soft tissue: light gray in the image, with value range (-100, 500)
    bone: white in the image, with value range (1000, 3000)
    pulmonary lesion: also light gray in the image, but with a narrower range (-10, 100)
        Later in Exercise 3c, we see that a better range is (-20, 200)
    
b) What SA cuts show a lesion? How many lesions does LIDC-IDRI-0003 have?
    According to the Metadata file, LIDC-IDRI-0003 has 3 lesion. 
    The SA cuts with lesion contour 86 to 94 shows the lesion. for LIDC-IDRI-0003.
    #TODO: show multiple the lesion contours for LIDC-IDRI-0003
    
c) Install the Slicer3D software available at https://www.slicer.org and visualize
lesions of the LIDC-IDRI-0003 case.
    None

Exercise 2. Generate ROIs
a) Use the mask to generate the bounding boxes. You can use Pyradiomics’
Imageoperations’ checkMask(). Install it using: pip install pyradiomics==3.0.1.
    Ex1.1.2.a Bounding Boxes: [298 338 341 390  86  94]
    Code in `exercise_code.py`

b) Use the affine matrix to convert the bounding box coordinates of the metadata
file to voxel coordinates. More information about it can be found at:
https://nipy.org/nibabel/coordinate_systems.html
    Ex1.1.2.b Bounding Box in Voxel Coordinates:
    [(299.0, 339.0, 342.0), (391.0, 87.0, 95.0)]
    Code in `exercise_code.py`


Exercise 3. Lesion Segmentation. Load and visualize one of the lesion ROIs.
a) Set a value for thresholding, segment the lesion and visualize the segmentation
    Using the result from Ex1.a, the threshold value is -20 to 200.
    The result is shown in the figure in `exercise_code.py`

b) Use the threshold set in a) to segment the remaining lesions. Discuss the quality
of the results
    The quality is not good. Although it includes the lesion, it also includes
    other tissues.

c) Define your own ROIs from the whole volume and segment the lesion using the
threshold defined in a). Discuss the quality of the results
    Use the result from Ex 2.b Bound Box to define the ROI.
"""
