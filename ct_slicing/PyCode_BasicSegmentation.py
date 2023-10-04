"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
import numpy as np

# Pyhton standard Visualization Library
import matplotlib.pyplot as plt

# Pyhton standard IOs Library
import os

# Basic Processing
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt

### IMPORT SESSION FUNCTIONS
#### Session Code Folder
SessionPyFolder = r""
os.chdir(SessionPyFolder)  # Change Dir 2 load session functions
# .nii Read Data
from NiftyIO import readNifty

# Volume Visualization
from VolumeCutBrowser import VolumeCutBrowser

######## LOAD DATA

#### Data Folders
SessionDataFolder = "/Users/pau/Downloads/OneDrive_1_12-4-2023"
os.chdir(SessionDataFolder)


CaseFolder = "CT"
NiiFile = "LIDC-IDRI-0001.nii.gz"


#### Load Intensity Volume
NiiFile = os.path.join(SessionDataFolder, CaseFolder, "image", NiiFile)
niiROI, niimetada = readNifty(NiiFile)


######## VISUALIZE VOLUMES

### Interactive Volume Visualization
# Short Axis Cuts
VolumeCutBrowser(niiROI)


######## SEGMENTATION PIPELINE

### 1. PRE-PROCESSING
# 1.1 Gaussian Filtering
sig = 1
niiROIGauss = filt.gaussian_filter(niiROI, sigma=sig)
# 1.2 MedFilter
sze = 3
niiROIMed = filt.median_filter(niiROI, sze)
###

### 2. BINARIZATION
Th = threshold_otsu(niiROI)
niiROISeg = niiROI > Th
# ROI Histogram
fig, ax = plt.subplots(1, 1)
ax.hist(niiROI.flatten(), bins=50, edgecolor="k")
# Visualize Lesion Segmentation
VolumeCutBrowser(niiROI, IMSeg=niiROISeg)


### 3.POST-PROCESSING

# 3.1  Opening
szeOp = 3
se = Morpho.cube(szeOp)
niiROISegOpen = Morpho.binary_opening(niiROISeg, se)

# 3.2  Closing
szeCl = 3
se = Morpho.cube(szeCl)
niiROISegClose = Morpho.binary_closing(niiROISeg, se)
