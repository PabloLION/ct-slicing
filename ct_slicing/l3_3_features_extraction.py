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
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from ct_slicing.config.data_path import (
    DEFAULT_EXPORT_XLSX_PATH,
    RADIOMICS_DEFAULT_PARAMS_PATH,
)
from ct_slicing.data_util.metadata_access import load_metadata_excel_to_data_frame
from ct_slicing.data_util.nii_file_access import (
    case_id_to_patient_id,
    nii_file,
)
from ct_slicing.image_process import process_image
from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty, NiiMetadata
from ct_slicing.ct_logger import logger

DEFAULT_MASK_MIN_PIXELS = 15  # was 200, too large to see `diagnosis==0` samples

# to fix wrong implementation of radiomics.setVerbosity(60)
logging.getLogger("radiomics").setLevel(logging.CRITICAL)  # run radiomics quietly
logging.getLogger("pykwalify").setLevel(logging.CRITICAL)  # pykwalify from radiomics
# #TODO: fix Shape features are only available 3D input (for 2D input, use shape2D). Found 2D input

logger.setLevel(logging.INFO)


# #TODO: refactor move, or maybe use pandas.DataFrame.to_excel
def write_excel(df: pd.DataFrame, path: Path):
    # (over)write DataFrame to excel file

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(path, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1", index=False)
    writer.close()


def get_features(
    feature_vector: dict,
    i: int,
    patient_id: str,
    patient_nodule_index: int,
    diagnosis: int,
) -> dict[str, float]:
    """Was called getFeatures"""

    new_row = {}
    # Showing the features and its calculated values
    for feature_name in feature_vector.keys():
        # print("Computed {}: {}".format(featureName, featureVector[featureName]))
        if (
            ("firstorder" in feature_name)
            or ("glszm" in feature_name)
            or ("glcm" in feature_name)
            or ("glrlm" in feature_name)
            or ("gldm" in feature_name)
            or ("shape" in feature_name)
        ):
            new_row.update({feature_name: feature_vector[feature_name]})
    items = sorted(new_row.items())  # Ordering the new_row dictionary
    # Adding some columns
    items.insert(0, ("diagnosis", diagnosis))
    items.insert(0, ("slice_number", i))
    items.insert(0, ("nodule_id", patient_nodule_index))
    items.insert(0, ("patient_id", patient_id))
    # In Python 3.7 and later, the built-in dict type maintains insertion order
    # by default. So OrderedDict(items) is no longer needed.
    return dict(items)


def get_record(
    patient_id: str,
    nodule_index: int,
    diagnosis: int,
    image: np.ndarray,
    mask: np.ndarray,
    img_meta: NiiMetadata,
    mask_meta: NiiMetadata,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    mask_pixel_min_threshold: int,
):
    record = []

    for slice_index in range(image.shape[2]):  # X, Y, Z
        # Get the axial cut
        mask_slice = mask[:, :, slice_index]
        if mask_slice.sum() < mask_pixel_min_threshold:
            logger.debug(
                f"Skipping {slice_index=} of {patient_id=} {nodule_index=} because it has less than {mask_pixel_min_threshold=} pixels"
            )
            continue
        img_slice = image[:, :, slice_index]

        # Get back to the format sitk
        img_slice_sitk = sitk.GetImageFromArray(img_slice)
        mask_slice_sitk = sitk.GetImageFromArray(mask_slice)

        # Recover the pixel dimension in X and Y
        (x1, y1, _z1) = img_meta.spacing
        (x2, y2, _z2) = mask_meta.spacing
        img_slice_sitk.SetSpacing((float(x1), float(y1)))
        mask_slice_sitk.SetSpacing((float(x2), float(y2)))

        # Extract features
        feature_vector = extractor.execute(
            img_slice_sitk, mask_slice_sitk, voxelBased=False
        )
        feat_dict = get_features(
            feature_vector, slice_index, patient_id, nodule_index, diagnosis
        )
        record.append(feat_dict)

    if len(record) == 0:
        logger.info(
            f"Skipping patient {patient_id} nodule {nodule_index} because it has no slices with more than {mask_pixel_min_threshold} pixels"
        )

    return record


def extract_features_of_one_record(
    section: Literal["CT", "VOI"],
    case_id: int,
    nodule_id: int,
) -> list[dict[str, float]]:
    """
    extract features from a single patient and return a DataFrame
    """

    img_path, mask_path = nii_file(section, case_id, nodule_id)
    patient_id = case_id_to_patient_id(case_id)

    # Reading image and mask
    image, img_meta = read_nifty(img_path, coordinate_order=CoordinateOrder.xyz)
    mask, mask_meta = read_nifty(mask_path, coordinate_order=CoordinateOrder.xyz)

    df_metadata = load_metadata_excel_to_data_frame()

    nodule_identity = df_metadata[
        (df_metadata.patient_id == patient_id) & (df_metadata.nodule_id == nodule_id)
    ]
    assert len(nodule_identity) == 1, f"Error: Found {len(nodule_identity)} rows"
    diagnosis: int = nodule_identity.Diagnosis_value.values[0]

    # pre-processing
    image = process_image(image)

    extractor = featureextractor.RadiomicsFeatureExtractor(
        str(RADIOMICS_DEFAULT_PARAMS_PATH)
    )
    mask_min_pixels = DEFAULT_MASK_MIN_PIXELS
    # Extract features slice by slice.
    record = get_record(
        patient_id,
        nodule_id,
        diagnosis,
        image,
        mask,
        img_meta,
        mask_meta,
        extractor,
        mask_min_pixels,
    )
    logger.info(
        f"Extracted features from {len(record):2} slices in {patient_id=} {nodule_id=} {diagnosis=}"
    )
    return record


def extract_features_of_all_records(
    case_nodule_id_to_extract: list[tuple[Literal["CT", "VOI"], int, int]]
):
    """Was the script body"""

    records = []
    for section, case_id, nodule_id in case_nodule_id_to_extract:
        records.extend(extract_features_of_one_record(section, case_id, nodule_id))
    df = pd.DataFrame.from_records(records)
    write_excel(df, DEFAULT_EXPORT_XLSX_PATH)


if __name__ == "__main__":
    records = []

    # #TODO long-term: extract more features from VOIs dataset
    case_nodule_id_to_extract = [
        # (section, case_id, nodule_id)
        ("CT", 1, 1),  # diagnosis:1
        ("CT", 3, 2),  # diagnosis:1 # in original code, only this case was processed
        ("CT", 3, 3),  # diagnosis:1
        ("CT", 3, 4),  # diagnosis:1
        ("CT", 5, 1),  # diagnosis:0
        ("CT", 5, 2),  # diagnosis:0
    ]

    extract_features_of_all_records(case_nodule_id_to_extract)

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
        slice_number
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
