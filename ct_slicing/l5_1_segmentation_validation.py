"""
This is a sample source code for basic validation of a segmentation method

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

# Validation 
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from skimage.measure import find_contours as contour

### IMPORT SESSION FUNCTIONS
#### PyCode Folder
PyFolder=''
os.chdir(PyFolder) #Change Dir 2 load session functions
# .nii Read Data
from NiftyIO import readNifty
# Volume Visualization
from VolumeCutBrowser import VolumeCutBrowser
# Segmentation Quality Scores
from l5_3_k_fold_sample_code import RelVolDiff,VOE,DICE,DistScores


######## 1. LOAD DATA

#### Data Folders
SessionDataFolder='/Users/pau/Downloads/OneDrive_1_12-4-2023'#r'D:\Teaching\Master\DataSci4Health\2023_ML4PM\Week 08 - Introduction\Dataset'
os.chdir(SessionDataFolder)


CaseFolder='CT'
NiiFile='LIDC-IDRI-0001.nii.gz'


#### Load Intensity Volume
NiiFile=os.path.join(SessionDataFolder,CaseFolder,'image',NiiFile)
niiROI,_=readNifty(NiiFile)

# Ground truth  mask of the lesion (niiROIGT)
NiiFile=os.path.join(SessionDataFolder,CaseFolder,'nodule_mask',NiiFile)
niiROIGT,_=readNifty(NiiFile)


### 2. VOLUME SEGMENTATION
Th = threshold_otsu(niiROI)
niiROISeg=niiROI>Th

### 3. VALIDATION SCORES
# Axial Cut
k=int(niiROI.shape[2]/2) # Cut at the middle of the volume. Change k to get other cuts
SA=niiROI[:,:,k]
SAGT=niiROIGT[:,:,k]
SASeg=niiROISeg[:,:,k]

# 3.1 Visualize GT contours over SA
fig=plt.figure()
plt.imshow(SA,cmap='gray')
plt.contour(SAGT,[0.5],colors='r')

# 3.2 Volumetric Measures
SegVOE=VOE(niiROISeg,niiROIGT)
SegDICE=DICE(niiROISeg,niiROIGT)
SegRelDiff=RelVolDiff(niiROISeg,niiROIGT)

SegVOE_SA=VOE(SASeg,SAGT)
SegDICE_SA=DICE(SASeg,SAGT)
SegRelDiff_SA=RelVolDiff(niiROISeg,niiROIGT)

# 3.3 Distance Measures 
# 3.3.1 Distance Map to Otsu Segmentation SA cut
DistSegInt=bwdist(SASeg) #Distance Map inside Segmentation
DistSegExt=bwdist(1-SASeg) #Distance Map outside Segmentation
DistSeg=np.maximum(DistSegInt,DistSegExt) #Distance Map at all points

# 3.3.2 Distance from GT to Otsu Segmentation
# GT Mask boundary points
BorderGT=contour(SAGT,0.5)
i=BorderGT[0][:,0].astype(int)
j=BorderGT[0][:,1].astype(int)

# Show histogram
fig=plt.figure()
plt.hist(DistSeg[i,j],bins=50,edgecolor='k')

# 3.3.3 Distance Scores
AvgDist,MxDist=DistScores(SASeg,SAGT)