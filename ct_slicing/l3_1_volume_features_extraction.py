__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

import SimpleITK as itk
from radiomics import featureextractor as feature_extractor, setVerbosity
from ct_slicing.ct_logger import logger

from ct_slicing.config.data_path import REPO_ROOT
from ct_slicing.get_data import nii_file

setVerbosity(60)


# choose the case id and nodule id to get the path of the nodule image and mask
IMG, MASK = nii_file("CT", 1, 1)

radiomics_params = str(
    REPO_ROOT / "ct_slicing" / "config" / "FeaturesExtraction_Params.yaml"
)
# Reading image and mask
imageITK = itk.ReadImage(IMG)
maskITK = itk.ReadImage(MASK)

# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.

# Initialize the feature extractor
extractor = feature_extractor.RadiomicsFeatureExtractor(radiomics_params)
# Calculate features
featureVector = extractor.execute(imageITK, maskITK)
# Show the features and its calculated values
for featureName in featureVector.keys():
    logger.info(f"Computed {featureName}: {featureVector[featureName]}")
