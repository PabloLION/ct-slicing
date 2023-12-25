__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Feature Extraction / PreTrained Networks


import logging
from typing import Iterable, Literal
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor

from ct_slicing.config.data_path import (
    DEFAULT_EXPORT_XLSX_PATH,
    RADIOMICS_DEFAULT_PARAMS_PATH,
)
from ct_slicing.data_util.metadata_access import (
    load_metadata,
    load_metadata_excel_to_data_frame,
)
from ct_slicing.data_util.nii_file_access import (
    case_id_to_patient_id,
    get_nii_path_iter,
    get_section_case_id_mask_id_iter,
    load_nodule_id_pickle,
    nii_file,
)
from ct_slicing.image_process import process_image
from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty, NiiMetadata
from ct_slicing.ct_logger import logger

# instead of MIN_PIXELS, we might should use a percentage of the mask size?
DEFAULT_MASK_MIN_PIXELS = 15  # was 200, too large to see `diagnosis==0` samples

# to fix wrong implementation of radiomics.setVerbosity(60)
logging.getLogger("radiomics").setLevel(logging.CRITICAL)  # run radiomics quietly
logging.getLogger("pykwalify").setLevel(logging.CRITICAL)  # pykwalify from radiomics
# #TODO: fix Shape features are only available 3D input (for 2D input, use shape2D). Found 2D input

logger.setLevel(logging.INFO)
extractor = RadiomicsFeatureExtractor(str(RADIOMICS_DEFAULT_PARAMS_PATH))


def format_feature_dict(
    feature_vector: dict,
    case_id: int,
    nodule_id: int,
    slice_index: int,
) -> dict[str, float]:
    """
    filter and sort the feature vector
    Was called getFeatures
    """

    # In Python 3.7 and later, the built-in dict type maintains insertion order
    # by default. So OrderedDict(items) is no longer needed.
    feature_entry = {
        "patient_id": case_id_to_patient_id(case_id),
        "nodule_id": nodule_id,
        "slice_index": slice_index,
        "diagnosis": load_metadata(case_id, nodule_id).diagnosis_value,
    }

    for feature_name, feature_value in sorted(feature_vector.items()):
        if not (
            ("firstorder" in feature_name)
            or ("glszm" in feature_name)
            or ("glcm" in feature_name)
            or ("glrlm" in feature_name)
            or ("gldm" in feature_name)
            or ("shape" in feature_name)
        ):
            continue
        if feature_name not in feature_entry:
            feature_entry[feature_name] = feature_value
        else:
            logger.error(f"Overwriting {feature_name=} of {case_id=}, {nodule_id=}")
    return feature_entry


def extract_features_of_one_nodule(
    section: Literal["CT", "VOI"],
    case_id: int,
    nodule_id: int,
    mask_pixel_min_threshold: int = DEFAULT_MASK_MIN_PIXELS,
) -> list[dict[str, float]]:
    """
    extract features from a single patient and return a DataFrame
    """
    img_path, mask_path = nii_file(section, case_id, nodule_id)
    image, img_meta = read_nifty(img_path, coordinate_order=CoordinateOrder.xyz)
    mask, mask_meta = read_nifty(mask_path, coordinate_order=CoordinateOrder.xyz)

    image = process_image(image)  # pre-processing
    records = []

    for slice_index in range(image.shape[2]):
        # Get the axial cut
        mask_slice = mask[:, :, slice_index]
        if mask_slice.sum() < mask_pixel_min_threshold:
            logger.debug(
                f"Skipping {slice_index=} of {case_id=} {nodule_id=} because it has less than {mask_pixel_min_threshold=} pixels"
            )
            continue
        img_slice = image[:, :, slice_index]

        img_slice_sitk = sitk.GetImageFromArray(img_slice)
        mask_slice_sitk = sitk.GetImageFromArray(mask_slice)
        img_slice_sitk.SetSpacing(img_meta.spacing[:2])  # [x,y,_z]
        mask_slice_sitk.SetSpacing(mask_meta.spacing[:2])  # [x,y,_z]

        # Extract features and append to records
        feature_vector = extractor.execute(
            img_slice_sitk, mask_slice_sitk, voxelBased=False
        )
        records.append(
            format_feature_dict(feature_vector, case_id, nodule_id, slice_index)
        )

    logger.info(
        f"Skipping patient {case_id} nodule {nodule_id}: no slices with more than {mask_pixel_min_threshold} pixels"
        if len == 0
        else f"Extracted feature from {len(records):2} slices of {case_id=} {nodule_id=}"
    )

    return records


def extract_features_of_all_nodules_to_excel(
    case_nodule_id_to_extract: Iterable[tuple[Literal["CT", "VOI"], int, int]]
):
    """Was the script body"""

    records = []
    for section, case_id, nodule_id in case_nodule_id_to_extract:
        records.extend(extract_features_of_one_nodule(section, case_id, nodule_id))
    df = pd.DataFrame.from_records(records)
    df.to_excel(DEFAULT_EXPORT_XLSX_PATH, index=False)


if __name__ == "__main__":
    # #TODO long-term: extract more features from VOIs dataset
    # in original code, only ("CT", 3, 2) case was processed
    ct_iter, voi_iter = get_section_case_id_mask_id_iter()
    extract_features_of_all_nodules_to_excel(ct_iter)

"""
Exercise 3. Features extraction for all images and masks in the database.
a) Observing the python code featuresExtraction.py, which files are used as input and
output? What features are extracted? Where is the diagnosis included? Notice that this
code applies pre-processing, can you explain what it does?
    Input: 
        py-radiomics params: ct_slicing/config/Params.yaml
        data: 
            full image: REPO_ROOT/data/CT/image/LIDC-IDRI-0003_1.nii.gz
            nodule mask: REPO_ROOT/data/CT/nodule_mask/LIDC-IDRI-0003_R_2.nii.gz
        metadata: REPO_ROOT/data/MetadatabyNoduleMaxVoting.xlsx
    Output:
        REPO_ROOT/output/features.xlsx
        
    What features are extracted? Very verbose output, but the features extracted are:
        slice_index
        diagnosis
        original_firstorder_10Percentile
        original_firstorder_90Percentile
        original_firstorder_Energy
        original_firstorder_Entropy
        original_firstorder_InterquartileRange
        original_firstorder_Kurtosis
        original_firstorder_Maximum
        original_firstorder_Mean
        original_firstorder_MeanAbsoluteDeviation
        original_firstorder_Median
        original_firstorder_Minimum
        original_firstorder_Range
        original_firstorder_RobustMeanAbsoluteDeviation
        original_firstorder_RootMeanSquared
        original_firstorder_Skewness
        original_firstorder_TotalEnergy
        original_firstorder_Uniformity
        original_firstorder_Variance
        original_glcm_Autocorrelation
        original_glcm_ClusterProminence
        original_glcm_ClusterShade
        original_glcm_ClusterTendency
        original_glcm_Contrast
        original_glcm_Correlation
        original_glcm_DifferenceAverage
        original_glcm_DifferenceEntropy
        original_glcm_DifferenceVariance
        original_glcm_Id
        original_glcm_Idm
        original_glcm_Idmn
        original_glcm_Idn
        original_glcm_Imc1
        original_glcm_Imc2
        original_glcm_InverseVariance
        original_glcm_JointAverage
        original_glcm_JointEnergy
        original_glcm_JointEntropy
        original_glcm_MCC
        original_glcm_MaximumProbability
        original_glcm_SumAverage
        original_glcm_SumEntropy
        original_glcm_SumSquares
        original_gldm_DependenceEntropy
        original_gldm_DependenceNonUniformity
        original_gldm_DependenceNonUniformityNormalized
        original_gldm_DependenceVariance
        original_gldm_GrayLevelNonUniformity
        original_gldm_GrayLevelVariance
        original_gldm_HighGrayLevelEmphasis
        original_gldm_LargeDependenceEmphasis
        original_gldm_LargeDependenceHighGrayLevelEmphasis
        original_gldm_LargeDependenceLowGrayLevelEmphasis
        original_gldm_LowGrayLevelEmphasis
        original_gldm_SmallDependenceEmphasis
        original_gldm_SmallDependenceHighGrayLevelEmphasis
        original_gldm_SmallDependenceLowGrayLevelEmphasis
        original_glrlm_GrayLevelNonUniformity
        original_glrlm_GrayLevelNonUniformityNormalized
        original_glrlm_GrayLevelVariance
        original_glrlm_HighGrayLevelRunEmphasis
        original_glrlm_LongRunEmphasis
        original_glrlm_LongRunHighGrayLevelEmphasis
        original_glrlm_LongRunLowGrayLevelEmphasis
        original_glrlm_LowGrayLevelRunEmphasis
        original_glrlm_RunEntropy
        original_glrlm_RunLengthNonUniformity
        original_glrlm_RunLengthNonUniformityNormalized
        original_glrlm_RunPercentage
        original_glrlm_RunVariance
        original_glrlm_ShortRunEmphasis
        original_glrlm_ShortRunHighGrayLevelEmphasis
        original_glrlm_ShortRunLowGrayLevelEmphasis
        original_glszm_GrayLevelNonUniformity
        original_glszm_GrayLevelNonUniformityNormalized
        original_glszm_GrayLevelVariance
        original_glszm_HighGrayLevelZoneEmphasis
        original_glszm_LargeAreaEmphasis
        original_glszm_LargeAreaHighGrayLevelEmphasis
        original_glszm_LargeAreaLowGrayLevelEmphasis
        original_glszm_LowGrayLevelZoneEmphasis
        original_glszm_SizeZoneNonUniformity
        original_glszm_SizeZoneNonUniformityNormalized
        original_glszm_SmallAreaEmphasis
        original_glszm_SmallAreaHighGrayLevelEmphasis
        original_glszm_SmallAreaLowGrayLevelEmphasis
        original_glszm_ZoneEntropy
        original_glszm_ZonePercentage
        original_glszm_ZoneVariance
    Where is the diagnosis included?
        in the metadata and in the output excel file
    Notice that this code applies pre-processing, can you explain what it does?
        There are three steps:
        1. shift_values: add 1024 to all values
        2. set_range: scale the range to 0-4000
        3. set_gray_level: quantize the gray intensity to 24 discrete levels

b) Make the necessary modification to only extract all the GLCM features.
    In the the code, change condition `... or ("glcm" in feature_name) ...`
    to `"glcm" in feature_name` to only extract GLCM features.

c) Modify the featureExtraction.py file to extract features from all nodules in the database.
    Done for CT dataset. See `extract_feature` function.
    But we can easily improve it to extract features from VOIs dataset as well,
    since we now have the `ct_iter` and `voi_iter`.
    I've not done it because it's not required by the exercise, and we might
    need more data later on.
"""
