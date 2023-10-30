"""
This is the pipeline for the introduction to local image descriptors. 

Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""
### REFERENCES
# www.scipy-lectures.org/advanced/image_processing/#edge-detection

### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
import numpy as np
# Mayavi Visualization Functions
from mayavi import mlab
# Pyhton standard Visualization Library
import matplotlib.pyplot as plt
# Pyhton standard IOs Library
import os
import sys
# Basic Processing
from skimage.filters import threshold_otsu
from scipy.ndimage import filters as filt
from scipy import ndimage as ndi

# Kmeans Clustering
from sklearn.cluster import KMeans

### IMPORT SESSION FUNCTIONS 
#### Session Code Folder
CodeMainDir=r''
sys.path.append(CodeMainDir)
# .nii Read Data
from NiftyIO import readNifty
# Volume Visualization
from VolumeViewer import VolumeCutBrowser
# Image Basic Descriptors
from PyCode_LocalDescriptors.GaborFilters import GaborFilterBank2D,GaborFilterBank3D
from PyCode_LocalDescriptors.BrowseGaborFiltBank import BrowseGaborFiltBank


######## PARAMETERS
#### Data Folders
SessionDataFolder=r''
CaseFolder='Case0016'
ROIFolder='LesionROIs'
MaskROINiiFile='LIDC-IDRI-0016_GT1_4_Mask.nii.gz'
ROINiiFile='LIDC-IDRI-0016_GT1_4.nii.gz'
NiiFile='LIDC-IDRI-0016_GT1.nii.gz'

#### Processing Parameters

image_name='filled_square' # filled_square, square, SA, SA_Mask, SAROI
sig=10 #sigma of gaussian filter
Medsze=3 #size of median filter
filter_image='gaussian' # none, gaussian, median
gabor_params='default' # default, non_default

######## LOAD DATA

#### Load ROI Volumes

NiiPath=os.path.join(SessionDataFolder,CaseFolder,ROIFolder,ROINiiFile)
NiiPath=NiiPath.replace('\\','/')
niiROI,_=readNifty(NiiPath)

NiiPath=os.path.join(SessionDataFolder,CaseFolder,ROIFolder,MaskROINiiFile)
NiiPath=NiiPath.replace('\\','/')
niiMask,_=readNifty(NiiPath)

NiiPath=os.path.join(SessionDataFolder,CaseFolder,ROIFolder,NiiFile)
NiiPath=NiiPath.replace('\\','/')
niivol,_=readNifty(NiiPath)


### Define Use Case Image  

if image_name == 'filled_square':
    # Synthetic Filled Square
    im = np.zeros((256, 256))
    im[64:-64, 64:-64] = 1
    Angle=15
    im = ndi.rotate(im, Angle, mode='constant') #2D rotation
    
elif image_name == 'square':
    # Synthetic Square
    im = np.zeros((256, 256))
    im[64:-64, 64:-64] = 1
    Angle=15
    im = ndi.rotate(im, Angle, mode='constant') #2D rotation
    
    im=ndi.gaussian_gradient_magnitude(im,sigma=2)
   
 
elif image_name == 'SA':
    k=int(niivol.shape[2]/2) # Cut at the middle of the volume 
    im=niivol[:,:,k]

elif image_name == 'SAROI':
    k=int(niiROI.shape[2]/2) # Cut at the middle of the volume 
    im=niiROI[:,:,k]
    
elif image_name == 'SA_Mask':
    k=int(niiMask.shape[2]/2) # Cut at the middle of the volume 
    im=niiMask[:,:,k]
else:
    raise Exception('Incorrect image name.')

# Image Filtering
if filter_image == 'gaussian':
    img= filt.gaussian_filter(im, sigma=sig)
elif filter_image == 'median':
    img= filt.median_filter(im, Medsze)
else:
    img=im
    
######## IMAGE DESCRIPTORS

### 1. GRADIENT
## 1.1 2D Images
# EX1. Compute gradient of 'filled_square', 'SAROI'
#      for different filters and analyze results
   
# Derivative along x-axis
sx = filt.sobel(img, axis=1, mode='constant')
# Derivative along y-axis
sy = filt.sobel(img, axis=0, mode='constant')
# Image Gradient (Sobel Edge Detector)
EdgeSob=np.sqrt(sx**2 + sy**2)

# Show Results
fig1 = plt.figure()
ax = fig1.add_subplot(141) 
ax.imshow(img,cmap='gray')
ax = fig1.add_subplot(142) 
ax.imshow(sx,cmap='gray')
ax.set_title('vertical edges')
ax = fig1.add_subplot(143) 
ax.imshow(sy,cmap='gray')
ax.set_title('horizontal edges')
ax = fig1.add_subplot(144) 
ax.imshow(EdgeSob,cmap='gray')
ax.set_title('Gradient Magnitude (EdgeDetector)')

## 1.2 3D Volumes
# EX2. Compute the gradient of a 3D volume 
# Hint: Add the derivatives in the z-axis

### 2. RIDGES, VALLEYS
## 2.1 2D Images
# EX3. Analyze results for ridges/valleys of 'square', 
#      the opposite 1-'square' and 'filled_square'  

# Second Derivative along x-axis
sx = filt.sobel(img, axis=1, mode='constant')
sxx = filt.sobel(sx, axis=1, mode='constant')
# Second Derivative along y-axis
sy = filt.sobel(img, axis=0, mode='constant')
syy = filt.sobel(sy, axis=0, mode='constant')
# Laplacian (Ridge/Valley Detector)
Lap=sxx + syy

# Show Results
fig1 = plt.figure()
ax = fig1.add_subplot(141) 
ax.imshow(img,cmap='gray')
ax = fig1.add_subplot(142) 
ax.imshow(sxx,cmap='gray')
ax.set_title('vertical ridges/valleys')
ax = fig1.add_subplot(143) 
ax.imshow(syy,cmap='gray')
ax.set_title('horizontal ridges/valleys')
ax = fig1.add_subplot(144) 
ax.imshow(Lap,cmap='gray')
ax.set_title('Laplacian')


fig1 = plt.figure()
ax = fig1.add_subplot(131) 
ax.imshow(img,cmap='gray')
ax = fig1.add_subplot(132) 
ax.imshow(abs(Lap)*(Lap>0),cmap='gray')
ax.set_title('valleys (Positive Laplacian)')
ax = fig1.add_subplot(133) 
ax.imshow(abs(Lap)*(Lap<0),cmap='gray')
ax.set_title('ridges (Negative Laplacian)')


### 3. GABOR FILTERS
## 3.1 2D Images
# EX4. Compare the response to each bank of Gabor filters
#      to the edge detector and Laplacian (ridge/valley detector)
#      of EX1 and EX3

# Filter Bank
if gabor_params=='default':
 GaborBank2D_1,GaborBank2D_2,params=GaborFilterBank2D() 
elif gabor_params=='non_default':
 sigGab=[2,4]
 freqGab=[0.25,0.5]
 GaborBank2D_1,GaborBank2D_2,params=GaborFilterBank2D(sigma=sigGab,frequency=freqGab)
 
# Show Filters
BrowseGaborFiltBank(GaborBank2D_1,params)
BrowseGaborFiltBank(GaborBank2D_2,params)

Gab2Show=1
fig1=mlab.figure()
mlab.surf(GaborBank2D_1[Gab2Show],warp_scale='auto')

# Apply Filters
NFilt=np.shape(GaborBank2D_1)
NFilt=NFilt[0]

Ressze=np.concatenate((im.shape,np.array([NFilt])))
imGab1= np.empty(Ressze)
imGab2= np.empty(Ressze)
for k in range(NFilt):
    imGab1[:,:,k]=ndi.convolve(im, GaborBank2D_1[k], mode='wrap')
    imGab2[:,:,k]=ndi.convolve(im, GaborBank2D_2[k], mode='wrap')
    
VolumeCutBrowser(imGab1)  
BrowseGaborFiltBank(GaborBank2D_1,params)

VolumeCutBrowser(imGab2)  
BrowseGaborFiltBank(GaborBank2D_2,params)



### 4. FEATURE SPACES
## EX5: Compare Otsu binarization that only takes into account
#  image intensity to using the values of intensity together with
#  some of the local descriptors: (intensity,Laplacian)

### 4.0 Data
# SA Cut
k=int(niiROI.shape[2]/2) # Cut at the middle of the volume 
im=niiROI[:,:,k]
im=(im-im.min())/(im.max()-im.min())
imMask=niiMask[:,:,k]

# SA cut Laplacian
sx = filt.sobel(im, axis=1, mode='constant')
sxx = filt.sobel(sx, axis=1, mode='constant')
sy = filt.sobel(im, axis=0, mode='constant')
syy = filt.sobel(sy, axis=0, mode='constant')
Lap=sxx + syy

### 4.1 Intensity thresholding
Th=threshold_otsu(im)
imSeg=im>Th

# Show Segmentation
plt.figure()
plt.imshow(im,cmap='gray')
plt.contour(im,[Th],colors='red')

# Show Intensity histogram and Otsu Threshold
plt.figure()
plt.hist(im.flatten(),edgecolor='k',bins=5)
plt.hist(im[np.nonzero(imMask)],bins=5,edgecolor='k',alpha=0.5,facecolor='r',label='Lesion Values')
plt.plot([Th,Th],[0,4000],'k',lw=2,label='Otsu Threshold')
plt.legend()

### 4.2 Feature Space Partition
# Pixel Representation in a 2D space. In the plot
# each pixel it is assigned a x-coordinate given by its intensity
# and a y-coordinate given by its Laplacian
#              Pixel(i,j) ---> (im(i,j),Lap(i,j))
#
# EX6: Try to divide the plane with a line splitting (discriminating)
#      the red (lesion) and blue points (background)



plt.figure()
plt.plot(im.flatten(),Lap.flatten(),'.')
plt.plot(im[np.nonzero(imMask)],Lap[np.nonzero(imMask)],'r.',label='Lesion Values')
plt.xlabel('Intensity')
plt.ylabel('Laplacian')
plt.title('Pixel Distribution in the Space of Values given by (Intensity,Laplacian).' )
plt.legend()

### 4.3 Kmeans Clustering
# EX7: Run k-means with and without normalization of the feature space

Lap=(Lap-Lap.min())/(Lap.max()-Lap.min())
X=np.array((im.flatten(),Lap.flatten()))
kmeans = KMeans(n_clusters=2)
kmeans=kmeans.fit(X.transpose())
labels = kmeans.predict(X.transpose())

plt.figure()
plt.plot(im.flatten(),Lap.flatten(),'.')
plt.plot(im[np.nonzero(imMask)],Lap[np.nonzero(imMask)],'r.',label='Lesion Values')
plt.xlabel('Intensity')
plt.ylabel('Laplacian')
plt.title('Pixel Distribution in the Space of Values given by (Intensity,Laplacian).' )
x=X[0,np.nonzero(labels==1)]
y=X[1,np.nonzero(labels==1)]
plt.plot(x.flatten(),y.flatten(),'k.',label='k-means Clustering')
plt.legend()