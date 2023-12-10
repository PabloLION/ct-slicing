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
from radiomics import featureextractor, setVerbosity

from ct_slicing.config.data_path import DATA_FOLDER, OUTPUT_FOLDER, REPO_ROOT
from ct_slicing.data_util.data_access import nii_file
from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty, NiiMetadata
from ct_slicing.ct_logger import logger

RADIOMICS_PARAMS_STR = str(REPO_ROOT / "ct_slicing" / "config" / "Params.yaml")
DEFAULT_MASK_MIN_PIXELS = 15  # was 200, too large to see `diagnosis==0` samples

# to fix wrong implementation of radiomics.setVerbosity(60)
logging.getLogger("radiomics").setLevel(logging.CRITICAL)  # run radiomics quietly
logging.getLogger("pykwalify").setLevel(logging.CRITICAL)  # pykwalify from radiomics
# #TODO: fix Shape features are only available 3D input (for 2D input, use shape2D). Found 2D input

logger.setLevel(logging.INFO)


def get_img_mask_pair_paths(
    data_set: Literal["CT", "VOIs"], patient_id: str, nodule_index: int
):
    section = "VOI" if data_set == "VOIs" else data_set
    case_id = int(patient_id.replace("LIDC-IDRI-", ""))
    return nii_file(section, case_id, nodule_index)


DEFAULT_EXPORT_XLSX_PATH = OUTPUT_FOLDER / "features.xlsx"
META_DATA_PATH = DATA_FOLDER / "MetadatabyNoduleMaxVoting.xlsx"


def shift_values(image: np.ndarray, value: int) -> np.ndarray:
    """Was called ShiftValues"""

    image = image + value
    logger.debug(f"Range after Shift: {image.min()} - {image.max()}")
    return image


def set_range(image: np.ndarray, in_min: int, in_max: int) -> np.ndarray:
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (in_max - in_min) + in_min

    image[image < 0] = 0
    image[image > image.max()] = image.max()
    logger.debug(f"Range after SetRange: {image.min():.2f} - {image.max():.2f}")
    return image


def set_gray_level(image: np.ndarray, levels: int) -> np.ndarray:
    """Was called SetGrayLevel

    Args:
        image (np.ndarray): an image with values between 0 and 1
        levels (int): the number of gray levels to use
    """
    image = (image * levels).astype(np.uint8)  # get into integer values
    logger.debug(
        f"Range after SetGrayLevel: {image.min():.2f} - {image.max():.2f} levels={levels}"
    )
    return image


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
    # In Python 3.7 and later, the built-in dict type maintains insertion order by default.
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


def extract_feature_record(
    data_set: Literal["CT", "VOIs"],
    patient_id: str,
    patient_nodule_index: int,
) -> list[dict[str, float]]:
    """
    extract features from a single patient and return a DataFrame
    """

    img_path, mask_path = get_img_mask_pair_paths(
        data_set, patient_id, patient_nodule_index
    )

    # Reading image and mask
    image, img_meta = read_nifty(img_path, coordinate_order=CoordinateOrder.xyz)
    mask, mask_meta = read_nifty(mask_path, coordinate_order=CoordinateOrder.xyz)

    df_metadata = pd.read_excel(
        META_DATA_PATH,
        sheet_name="ML4PM_MetadatabyNoduleMaxVoting",
        engine="openpyxl",
    )

    nodule_identity = df_metadata[
        (df_metadata.patient_id == patient_id)
        & (df_metadata.nodule_id == patient_nodule_index)
    ]
    assert len(nodule_identity) == 1, f"Error: Found {len(nodule_identity)} rows"
    diagnosis: int = nodule_identity.Diagnosis_value.values[0]

    # pre-processing
    image = shift_values(image, value=1024)
    image = set_range(image, in_min=0, in_max=4000)
    image = set_gray_level(image, levels=24)

    extractor = featureextractor.RadiomicsFeatureExtractor(RADIOMICS_PARAMS_STR)
    mask_min_pixels = DEFAULT_MASK_MIN_PIXELS
    # Extract features slice by slice.
    record = get_record(
        patient_id,
        patient_nodule_index,
        diagnosis,
        image,
        mask,
        img_meta,
        mask_meta,
        extractor,
        mask_min_pixels,
    )
    logger.info(
        f"Extracted {len(record):2} slices from {patient_id=} {patient_nodule_index=}"
    )
    return record


def extract_feature(patient_nodule_diagnosis: list[tuple[str, int, int]]):
    """Was the script body"""

    records = []
    for patient_id, patient_nodule_index, _diagnosis in patient_nodule_diagnosis:
        record = extract_feature_record(
            data_set="CT",
            patient_id=patient_id,
            patient_nodule_index=patient_nodule_index,
        )
        records.extend(record)
    df = pd.DataFrame.from_records(records)
    write_excel(df, DEFAULT_EXPORT_XLSX_PATH)


if __name__ == "__main__":
    records = []

    # TODO: extract this part to data loading module
    # FIX: extract features from VOIs dataset
    patient_nodule_diagnosis = [
        # (patient_id, patient_nodule_index, diagnosis)
        ("LIDC-IDRI-0001", 1, 1),
        ("LIDC-IDRI-0003", 2, 1),  # originally only this case was processed
        ("LIDC-IDRI-0003", 3, 1),
        ("LIDC-IDRI-0003", 4, 1),
        ("LIDC-IDRI-0005", 1, 0),
        ("LIDC-IDRI-0005", 2, 0),
    ]

    extract_feature(patient_nodule_diagnosis)
