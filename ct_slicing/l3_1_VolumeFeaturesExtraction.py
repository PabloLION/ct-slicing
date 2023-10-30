"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""


import os

import SimpleITK as sitk
from radiomics import featureextractor, setVerbosity

setVerbosity(60)


#### Parameters to be configured
db_path = "/Users/yixin/syh/ct-slicing/data/CT"
imageDirectory = "image"
maskDirectory = "nodule_mask"
imageName = os.path.join(db_path, imageDirectory, "LIDC-IDRI-0001.nii.gz")
maskName = os.path.join(db_path, maskDirectory, "LIDC-IDRI-0001_R_1.nii.gz")
####


# Reading image and mask
imageITK = sitk.ReadImage(imageName)
maskITK = sitk.ReadImage(maskName)

# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.
params = "config/FeaturesExtraction_Params.yaml"

# Initializing the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# Calculating features
featureVector = extractor.execute(imageITK, maskITK)

# Showing the features and its calculated values
for featureName in featureVector.keys():
    print("Computed {}: {}".format(featureName, featureVector[featureName]))
