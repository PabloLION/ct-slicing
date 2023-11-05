__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Feature Extraction / PyRadiomics
# Unit: Feature Extraction / PreTrained Networks


from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor, setVerbosity

from ct_slicing.config.data_path import DATA_FOLDER, OUTPUT_FOLDER, REPO_ROOT
from ct_slicing.vis_lib.NiftyIO import CoordinateOrder, read_nifty, NiiMetadata


setVerbosity(60)
# DATA_SET = "CT"
# PATIENT_ID = "LIDC-IDRI-0003"
# PATIENT_NODULE_INDEX = 2
# These original values are not valid for l3_5_featuresExtractionSVM.py
# because the "diagnosis" column all "malign"

## Testing with VOIs
# diagnosis = 0 case
# DATA_SET = "VOIs"
# PATIENT_ID = "LIDC-IDRI-0004"
# PATIENT_NODULE_INDEX = 1

# diagnosis = 1 case
DATA_SET = "CT"
PATIENT_ID = "LIDC-IDRI-0003"
PATIENT_NODULE_INDEX = 2

# TODO: extract this part to data loading module

DEFAULT_EXPORT_XLSX_PATH = OUTPUT_FOLDER / "features.xlsx"
META_DATA_PATH = DATA_FOLDER / "MetadatabyNoduleMaxVoting.xlsx"


IMG_FOLDER = DATA_FOLDER / DATA_SET / "image"
MASK_FOLDER = DATA_FOLDER / DATA_SET / "nodule_mask"
if DATA_SET == "CT":
    IMG = IMG_FOLDER / f"{PATIENT_ID}.nii.gz"
elif DATA_SET == "VOIs":
    IMG = IMG_FOLDER / f"{PATIENT_ID}_R_{PATIENT_NODULE_INDEX}.nii.gz"
else:
    raise ValueError(f'DATA_SET can only be "CT" or "VOIs", not {DATA_SET}')

MASK = MASK_FOLDER / f"{PATIENT_ID}_R_{PATIENT_NODULE_INDEX}.nii.gz"

radiomics_params = str(REPO_ROOT / "ct_slicing" / "pr_config" / "Params.yaml")


def shift_values(image: np.ndarray, value: int) -> np.ndarray:
    """Was called ShiftValues"""

    image = image + value
    print("Range after Shift: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image


def set_range(image: np.ndarray, in_min: int, in_max: int) -> np.ndarray:
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (in_max - in_min) + in_min

    image[image < 0] = 0
    image[image > image.max()] = image.max()
    print("Range after SetRange: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image


def set_gray_level(image: np.ndarray, levels: int) -> np.ndarray:
    """Was called SetGrayLevel

    Args:
        image (np.ndarray): an image with values between 0 and 1
        levels (int): the number of gray levels to use
    """
    image = (image * levels).astype(np.uint8)  # get into integer values
    print("Range after SetGrayLevel: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image


def append_to_excel(df: pd.DataFrame, path: Path):
    # append DataFrame to excel file

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    # append instead of overwrite to get more "diagnosis" classes for later
    # classification. (l3_5_features_extraction_svm.py)
    writer = pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="new")
    df.to_excel(writer, sheet_name="Sheet1", index=False)
    writer.close()


def get_features(
    feature_vector: dict, i: int, patient_id: str, nodule_id: int, diagnosis: int
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
    items.insert(0, ("nodule_id", nodule_id))
    items.insert(0, ("patient_id", patient_id))
    # In Python 3.7 and later, the built-in dict type maintains insertion order by default.
    return dict(items)


def get_record(
    patient_id: str,
    nodule_id: int,
    diagnosis: int,
    image: np.ndarray,
    mask: np.ndarray,
    img_meta: NiiMetadata,
    mask_meta: NiiMetadata,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    mask_min_pixels: int = 200,
):
    record = []

    for i in range(image.shape[2]):  # X, Y, Z
        # Get the axial cut
        mask_slice = mask[:, :, i]
        if mask_min_pixels >= mask_slice.sum():
            print(  # TODO: log
                f"Skipping slice {i} because it has less than {mask_min_pixels} pixels"
            )
            continue
        img_slice = image[:, :, i]

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
        feat_dict = get_features(feature_vector, i, patient_id, nodule_id, diagnosis)
        record.append(feat_dict)

    return record


def slice_mode(
    patient_id: str,
    nodule_id: int,
    diagnosis: int,
    image: np.ndarray,
    mask: np.ndarray,
    img_meta: NiiMetadata,
    mask_meta: NiiMetadata,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    mask_min_pixels: int = 200,
):
    record = []

    get_record(
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

    df = pd.DataFrame.from_records(record)
    return df


def extend_records_target(
    patient_id: str,
    nodule_id: int,
    diagnosis: int,
    image: np.ndarray,
    mask: np.ndarray,
    img_meta: NiiMetadata,
    mask_meta: NiiMetadata,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    mask_min_pixels: int = 200,
    records_target: list[dict[str, float]] | None = None,
):
    if records_target is None:
        records_target = []

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
    records_target.extend(record)


def extract_feature(
    img_path: Path, mask_path: Path, export_excel_path: Path = DEFAULT_EXPORT_XLSX_PATH
):
    """
    extract features from a single patient and append to an excel file.
    """

    # Reading image and mask
    image, img_meta = read_nifty(img_path, coordinate_order=CoordinateOrder.xyz)
    mask, mask_meta = read_nifty(mask_path, coordinate_order=CoordinateOrder.xyz)

    df_metadata = pd.read_excel(
        META_DATA_PATH,
        sheet_name="ML4PM_MetadatabyNoduleMaxVoting",
        engine="openpyxl",
    )

    nodule_identity = df_metadata[
        (df_metadata.patient_id == PATIENT_ID)
        & (df_metadata.nodule_id == PATIENT_NODULE_INDEX)
    ]
    assert len(nodule_identity) == 1, f"Error: Found {len(nodule_identity)} rows"
    diagnosis: int = nodule_identity.Diagnosis_value.values[0]

    # pre-processing
    image = shift_values(image, value=1024)
    image = set_range(image, in_min=0, in_max=4000)
    image = set_gray_level(image, levels=24)

    # Extract features slice by slice.
    df = slice_mode(
        PATIENT_ID,
        PATIENT_NODULE_INDEX,
        diagnosis,
        image,
        mask,
        img_meta,
        mask_meta,
        extractor=featureextractor.RadiomicsFeatureExtractor(radiomics_params),
        mask_min_pixels=200,
    )

    append_to_excel(df, export_excel_path)


if __name__ == "__main__":
    records_target = []
    extract_feature(IMG, MASK, DEFAULT_EXPORT_XLSX_PATH)
