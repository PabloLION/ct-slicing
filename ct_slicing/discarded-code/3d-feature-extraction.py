"""
Modified from l3_3_features_extraction.py
Discarded because the extracted features are 3D images instead of scalars.
I wanted to extract some scalar features from the 3D images then use them as
the input of a classifier.
"""

if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import logging
from typing import Iterable, Literal
from matplotlib import pyplot as plt
import numpy
import pandas as pd
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor

from ct_slicing.config.data_path import (
    OUTPUT_FOLDER,
    RADIOMICS_CUSTOM_PARAMS_PATH,
)
from ct_slicing.data_util.metadata_access import (
    load_metadata,
)
from ct_slicing.data_util.nii_file_access import (
    case_id_to_patient_id,
    get_section_case_id_mask_id_iter,
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
extractor = RadiomicsFeatureExtractor(
    str(
        (
            RADIOMICS_CUSTOM_PARAMS_PATH / ".." / "feature-extraction-param-3d.yaml"
        ).resolve()
    )
)


def format_feature_dict(
    feature_vector: dict,
    case_id: int,
    nodule_id: int,
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
        "diagnosis": load_metadata(case_id, nodule_id).diagnosis_value,
    }

    for feature_name, feature_value in sorted(feature_vector.items()):
        # print(f"{feature_name=}")
        if not any(
            k in feature_name
            for k in ("firstorder", "glcm", "gldm", "glrlm", "glszm", "shape")
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
) -> list[dict[str, float]]:
    """
    extract features from a single patient and return a DataFrame
    """
    img_path, mask_path = nii_file(section, case_id, nodule_id)
    img, img_meta = read_nifty(img_path, coordinate_order=CoordinateOrder.zyx)
    mask, mask_meta = read_nifty(mask_path, coordinate_order=CoordinateOrder.zyx)
    img = process_image(img)  # pre-processing

    img_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)
    img_sitk.SetSpacing(img_meta.spacing)
    mask_sitk.SetSpacing(mask_meta.spacing)

    print(f"Extracting features from {case_id=}, {nodule_id=}")

    feature_vector = extractor.execute(img_sitk, mask_sitk, voxelBased=True)

    # show a slice of the image
    slice_to_show = img[:, :, img.shape[0] // 2]
    plt.imshow(slice_to_show)

    for k, v in feature_vector.items():
        if isinstance(v, sitk.Image):
            np_image = sitk.GetArrayFromImage(v)
            print(f"dimension of {k=} is {np_image.shape}")
            print(f"the original input image is {img.shape}")
            print(f"nonzero pixels of {k=} is {np_image.nonzero()}")
            for nzp in np_image.nonzero():
                print(f"{k=} nonzero pixels {nzp=}")
            if np_image.nonzero()[0].size == 0:
                print(f"empty {k=}")
                continue

            plt.imshow(np_image[0, :, :])  # Change indices to view different slices
            plt.title(k)
            plt.show()

    # print(feature_vector)  # This should print the dictionary of features

    return [format_feature_dict(feature_vector, case_id, nodule_id)]


def extract_features_of_all_nodules_to_excel(
    case_nodule_id_to_extract: Iterable[tuple[Literal["CT", "VOI"], int, int]]
):
    records = []
    limit = 3
    for section, case_id, nodule_id in case_nodule_id_to_extract:
        limit -= 1
        records.extend(extract_features_of_one_nodule(section, case_id, nodule_id))
        if limit == 0:
            break

    df = pd.DataFrame.from_records(records)
    df.to_excel(OUTPUT_FOLDER / "3d-features.xlsx", index=False)


if __name__ == "__main__":
    # #TODO long-term: extract more features from VOIs dataset
    # in original code, only ("CT", 3, 2) case was processed
    ct_iter, voi_iter = get_section_case_id_mask_id_iter()
    extract_features_of_all_nodules_to_excel(voi_iter)
