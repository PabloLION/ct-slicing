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


import numpy as np
import matplotlib.pyplot as plt
import os

SessionPyFolder = os.path.dirname(__file__)
SessionDataFolder = os.path.join(SessionPyFolder, "data")


### IMPORT SESSION FUNCTIONS
#### Session Code Folder (change to your path)
os.chdir(SessionPyFolder)  # Change Dir 2 load session functions
# .nii Read Data
from NiftyIO import read_nifty

# Volume Visualization
from VolumeCutBrowser import VolumeCutBrowser


######## LOAD DATA

#### Data Folders (change to your path)
os.chdir(SessionDataFolder)


CaseFolder = "CT"
Nii_File = "LIDC-IDRI-0001.nii.gz"


#### Load Intensity Volume
Nii_File = os.path.join(SessionDataFolder, CaseFolder, "image", Nii_File)
nii_vol, nii_metadata = read_nifty(Nii_File)
#### Load Nodule Mask
Nii_File = os.path.join(SessionDataFolder, CaseFolder, "nodule_mask", Nii_File)
nii_mask, nii_metadata = read_nifty(Nii_File)

######## VOLUME METADATA
print("Voxel Resolution (mm): ", nii_metadata.spacing)
print("Volume origin (mm): ", nii_metadata.origin)
print("Axes direction: ", nii_metadata.direction)
######## VISUALIZE VOLUMES

### Interactive Volume Visualization
# Short Axis View
VolumeCutBrowser(nii_vol)
VolumeCutBrowser(nii_vol, IMSSeg=nii_mask)
# Coronal View
VolumeCutBrowser(nii_vol, cut="Cor")
# Sagital View
VolumeCutBrowser(nii_vol, cut="Sag")


### Short Axis (SA) Image
# Define SA cut
k = int(nii_vol.shape[2] / 2)  # Cut at the middle of the volume
SA = nii_vol[:, :, k]
# Image
fig1 = plt.figure()
plt.imshow(SA, cmap="gray")
plt.close(fig1)  # close figure fig1

# Cut Level Sets
levels = [400]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect="equal")
ax1.imshow(SA, cmap="gray")
plt.contour(SA, levels, colors="r", linewidths=2)
plt.close("all")  # close all plt figures
