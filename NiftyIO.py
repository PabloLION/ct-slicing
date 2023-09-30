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
# Library for reading .nii Files
import SimpleITK as sitk
# Python Library 2 manage volumetric data
import numpy as np

#######################################################
# .nii Metadata (Origin, Resolution, Orientation)
class Metadata():
    def __init__(self, origen=None, spacing=None, direction=None):
        self.origen = origen
        self.spacing = spacing
        self.direction = direction
        
########################################################
# FUNCTION: readNifty(filePath)
#        
# INPUT: 
# 1> filePath is the full path to the file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
# 2> CoordinateOrder: Order of dimensions in array: 
#                     'xyz' (Default) sets z as volume third dimension
#                     'zyx' swaps x and z to set z as first dimension
# OUTPUT: 
# 1> volume_xyz: np.ndarray containing .nii volume
# 2> metadata: .nii metadata         
def readNifty(filePath,CoordinateOrder='xyz'):
    """
 # INPUT: 
 # 1> filePath is the full path to the file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
 # 2> CoordinateOrder: Order of dimensions in array: 
 #                     'xyz' (Default) sets z as volume third dimension
 #                     'zyx' swaps x and z to set z as first dimension
 #
 # OUTPUT: 
 # 1> volume_xyz: np.ndarray containing .nii volume
 # 2> metadata: .nii metadata 
 #
 # EXAMPLE:
 # 1. Skip metadata output argument
 # import os
 # from PyCode_Session1.NiftyIO import readNifty
 # filePath=os.path.join("Data_Session1","LIDC-IDRI-0001_GT1.nii.gz")
 # vol,_=readNifty(filePath)
    """
    image = sitk.ReadImage(filePath)
    print("Reading Nifty format from {}".format(filePath))
    print("Image size: {}".format(image.GetSize()))

    metadata = Metadata(image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # Converting from SimpleITK image to Numpy array. But also is changed the coordinate systems
    # from the image which use (x,y,z) to the array using (z,y,x).
    volume_zyx = sitk.GetArrayFromImage(image)
    if CoordinateOrder=='xyz':
        volume_xyz = np.transpose(volume_zyx, (2, 1, 0))  # to back to the initial xyz coordinate system.
    else:
        volume_xyz=volume_zyx

    print("Volume shape: {}".format(volume_xyz.shape))
    print("Minimum value: {}".format(np.min(volume_xyz)))
    print("Maximum value: {}".format(np.max(volume_xyz)))

    return volume_xyz, metadata     # return two items.

########################################################
# FUNCTION: saveNifty(volume, metadata, filename)
#        
# INPUT: 
# 1> volume: np.ndarray containing .nii volume
# 2> metadata: .nii metadata (optional).
#    If ommitted default (identity) values are used
# 3> filename is the full path to the output file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
# 4> CoordinateOrder: Order of dimensions in array: 
#                     'xyz' (Default) sets z as volume third dimension
#                     'zyx' swaps x and z to set z as first dimension
# OUTPUT:
#    
def saveNifty(volume, metadata, filename,CoordinateOrder='xyz'):
    """
    # FUNCTION: saveNifty(volume, metadata, filename,CoordinateOrder)
#        
# INPUT: 
# 1> volume: np.ndarray containing .nii volume
# 2> metadata: .nii metadata (optional).
#    If ommitted default (identity) values are used
# 3> filename is the full path to the output file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
# 4> CoordinateOrder: Order of dimensions in array: 
#                     'xyz' (Default) sets z as volume third dimension
#                     'zyx' swaps x and z to set z as first dimension
# OUTPUT:
# 
    """
    # Converting from Numpy array to SimpleITK image.
    if CoordinateOrder=='xyz':
        volume = np.transpose(volume, (2, 1, 0)) # from (x,y,z) to (z,y,x)
    image = sitk.GetImageFromArray(volume)  # It is supposed that GetImageFromArray receive an array with (z,y,x)

    if metadata is not None:
        # Setting some properties to the new image
        image.SetOrigin(metadata.origen)
        image.SetSpacing(metadata.spacing)
        image.SetDirection(metadata.direction)

    sitk.WriteImage(image, filename)



