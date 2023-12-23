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
from ct_slicing.data_util.nii_file_access import nii_file

setVerbosity(60)


# choose the case id and nodule id to get the path of the nodule image and mask
IMG_PATH, MASK_PATH = nii_file("CT", 1, 1)

# #TODO: refactor
radiomics_params = str(
    REPO_ROOT / "ct_slicing" / "config" / "FeaturesExtraction_Params.yaml"
)
# Reading image and mask
img = itk.ReadImage(IMG_PATH)
mask = itk.ReadImage(MASK_PATH)

# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.

# Initialize the feature extractor
extractor = feature_extractor.RadiomicsFeatureExtractor(radiomics_params)
# Calculate features
feature_vector = extractor.execute(img, mask)
# Show the features and its calculated values
for feature_name in feature_vector.keys():
    logger.info(f"Computed {feature_name}: {feature_vector[feature_name]}")

"""
## Exercises

### Exercise 1. Feature extraction from a volume image and mask.
a) Execute the code VolumeFeaturesExtraction.py and analyse its output. What are the
features extracted? What is the meaning of each value associated to each feature?
    Very verbose output, but the features extracted are:
    Computed diagnostics_Image-original_Spacing: (0.703125, 0.703125, 2.5)l3_1_volume_features_extraction.py:41
    Computed diagnostics_Image-original_Size: (512, 512, 133)
    Computed diagnostics_Image-original_Mean: -826.9439289121699
    Computed diagnostics_Image-original_Minimum: -2048.0
    Computed diagnostics_Image-original_Maximum: 3071.0
    Computed diagnostics_Mask-original_Hash: c59f4d5ddb4a11baa8dbfac94be85891c98a8d9c
    Computed diagnostics_Mask-original_Spacing: (0.703125, 0.703125, 2.5)
    Computed diagnostics_Mask-original_Size: (512, 512, 133)
    Computed diagnostics_Mask-original_BoundingBox: (298, 341, 86, 41, 50, 9)
    Computed diagnostics_Mask-original_VoxelNum: 6547
    Computed diagnostics_Mask-original_VolumeNum: 1
    Computed diagnostics_Mask-original_CenterOfMassIndex: (316.1580876737437, 367.08232778371774, 89.63846036352528)
    Computed diagnostics_Mask-original_CenterOfMass: (56.29865539560103, 86.40476477468434, -115.9038490911868)
    Computed diagnostics_Image-interpolated_Spacing: (1.0, 1.0, 1.0)
    Computed diagnostics_Image-interpolated_Size: (41, 47, 35)
    Computed diagnostics_Image-interpolated_Mean: -557.1375639409889
    Computed diagnostics_Image-interpolated_Minimum: -988.0
    Computed diagnostics_Image-interpolated_Maximum: 1006.0
    Computed diagnostics_Mask-interpolated_Spacing: (1.0, 1.0, 1.0)
    Computed diagnostics_Mask-interpolated_Size: (41, 47, 35)
    Computed diagnostics_Mask-interpolated_BoundingBox: (6, 6, 7, 28, 35, 22)
    Computed diagnostics_Mask-interpolated_VoxelNum: 8174
    Computed diagnostics_Mask-interpolated_VolumeNum: 1
    Computed diagnostics_Mask-interpolated_CenterOfMassIndex: (18.16540249571813, 23.924516760459994, 16.65218987032053)
    Computed diagnostics_Mask-interpolated_CenterOfMass: (56.313839995718126, 86.3729573122178, -116.09781012967947)
    Computed diagnostics_Mask-interpolated_Mean: -172.20846586738438
    Computed diagnostics_Mask-interpolated_Minimum: -888.0
    Computed diagnostics_Mask-interpolated_Maximum: 256.0
    Computed original_shape_VoxelVolume: 8174.0
    Computed original_shape_SurfaceArea: 2970.851587589866
    Computed original_shape_SurfaceVolumeRatio: 0.3655907772327911   

b) Identify each part/section of the output. Why could be useful to have this information.
    The output is divided into 3 parts:
        1. diagnostics: information about the image and mask
        2. original_shape: information about the shape of the lesion
        3. original_firstorder: information about the intensity of the lesion
    This information is useful to understand the lesion and to compare it with other lesions.

c) Make the corresponding modification to extract other type of features.
    Maybe use another parameter file?

Exercise 2. See l3_2_slice_features_extraction.py
"""
