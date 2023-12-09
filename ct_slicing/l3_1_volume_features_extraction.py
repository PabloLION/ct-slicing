__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

import SimpleITK as sitk
from radiomics import featureextractor, setVerbosity

from ct_slicing.config.data_path import DATA_FOLDER, REPO_ROOT

setVerbosity(60)


#### Parameters to be configured

IMG_FOLDER = DATA_FOLDER / "CT" / "image"
MASK_FOLDER = DATA_FOLDER / "CT" / "nodule_mask"
IMG = IMG_FOLDER / "LIDC-IDRI-0001.nii.gz"
MASK = MASK_FOLDER / "LIDC-IDRI-0001_R_1.nii.gz"

radiomics_params = str(
    REPO_ROOT / "ct_slicing" / "pr_config" / "FeaturesExtraction_Params.yaml"
)

####


# Reading image and mask
imageITK = sitk.ReadImage(IMG)
maskITK = sitk.ReadImage(MASK)

# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.

# Initializing the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(radiomics_params)

# Calculating features
featureVector = extractor.execute(imageITK, maskITK)

# Showing the features and its calculated values
for featureName in featureVector.keys():
    print("Computed {}: {}".format(featureName, featureVector[featureName]))
