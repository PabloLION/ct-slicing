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
from typing import Literal
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor, setVerbosity

from ct_slicing.config.data_path import DATA_FOLDER, OUTPUT_FOLDER, REPO_ROOT
from ct_slicing.vis_lib.NiftyIO import CoordinateOrder, read_nifty, NiiMetadata


setVerbosity(60)  # TODO: how to quietly run? (verbosity level 60 is showing info)
RADIOMICS_PARAMS_STR = str(REPO_ROOT / "ct_slicing" / "pr_config" / "Params.yaml")


# TODO: extract this part to data loading module
def get_img_mask_pair_paths(
    data_set: Literal["CT", "VOIs"], patient_id: str, nodule_index: int
):
    img_folder = DATA_FOLDER / data_set / "image"
    mask_folder = DATA_FOLDER / data_set / "nodule_mask"
    if data_set == "CT":
        img_path = img_folder / f"{patient_id}.nii.gz"
    elif data_set == "VOIs":
        img_path = img_folder / f"{patient_id}_R_{nodule_index}.nii.gz"
    else:
        raise ValueError(f'data_set can only be "CT" or "VOIs", not {data_set}')

    mask_path = mask_folder / f"{patient_id}_R_{nodule_index}.nii.gz"
    return img_path, mask_path


DEFAULT_EXPORT_XLSX_PATH = OUTPUT_FOLDER / "features.xlsx"
META_DATA_PATH = DATA_FOLDER / "MetadatabyNoduleMaxVoting.xlsx"


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
    patient_nodule_index: int,
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
        feat_dict = get_features(
            feature_vector, i, patient_id, patient_nodule_index, diagnosis
        )
        record.append(feat_dict)

    return record


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
    data_set: Literal["CT", "VOIs"],
    patient_id: str,
    patient_nodule_index: int,
) -> pd.DataFrame:
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
    mask_min_pixels = 200
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

    df = pd.DataFrame.from_records(record)
    return df


if __name__ == "__main__":
    records_target = []

    patient_nodule_diagnosis = [
        # (patient_id, patient_nodule_index, diagnosis)
        ("LIDC-IDRI-0001", 1, 1),
        ("LIDC-IDRI-0003", 2, 1),
        ("LIDC-IDRI-0003", 3, 1),
        ("LIDC-IDRI-0003", 4, 1),
        ("LIDC-IDRI-0005", 1, 0),
        ("LIDC-IDRI-0005", 2, 0),
    ]

    data_set, patient_id, patient_nodule_index = "CT", "LIDC-IDRI-0003", 2

    df = extract_feature(data_set, patient_id, patient_nodule_index)
    write_excel(df, DEFAULT_EXPORT_XLSX_PATH)
