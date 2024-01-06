"""
The result shows that no VOI file is same as the corresponding CT file.
So we cannot remove the CT folder.
"""
if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

from collections import defaultdict
from ct_slicing.config.data_path import CT_FOLDER, VOI_FOLDER
import filecmp
from ct_slicing.data_util.metadata_access import load_all_metadata

from ct_slicing.data_util.nii_file_access import (
    iter_files,
    load_nodule_id_pickle,
)

EXCLUDED_FILENAMES = (".DS_Store",)


def check_voi_contains_ct():
    for file_rel_path in iter_files(CT_FOLDER):
        if file_rel_path.name in EXCLUDED_FILENAMES:
            continue
        corresponding_voi = VOI_FOLDER / file_rel_path
        if not corresponding_voi.exists():
            print(f"VOI file not found for {file_rel_path}")
            continue
        if not filecmp.cmp(CT_FOLDER / file_rel_path, corresponding_voi):
            print(f"VOI file not equal for {file_rel_path}")
            continue
        print(f"VOI file equal for {file_rel_path}")
        print("")


def check_voi_img_eq_mask():
    for file_rel_path in iter_files(VOI_FOLDER / "nodule_mask"):
        if file_rel_path.name in EXCLUDED_FILENAMES:
            continue
        if (VOI_FOLDER / "image" / file_rel_path).exists():
            print(f"VOI image found for {file_rel_path}")
            continue
        print(f"VOI image not found for {file_rel_path}")
    for file_rel_path in iter_files(VOI_FOLDER / "image"):
        if file_rel_path.name in EXCLUDED_FILENAMES:
            continue
        if (VOI_FOLDER / "nodule_mask" / file_rel_path).exists():
            print(f"VOI nodule mask found for {file_rel_path}")
            continue
        print(f"VOI nodule mask not found for {file_rel_path}")


def compare_voi_and_metadata():
    _ct_file_ids, voi_file_ids = load_nodule_id_pickle()
    metadata_ids = set(load_all_metadata().keys())
    print(f"{voi_file_ids-metadata_ids=}, {metadata_ids-voi_file_ids=}")


def malignancy_value_vs_diagnosis():
    all_metadata = load_all_metadata()
    malignancy_values_to_diagnosis_values = defaultdict(set)
    for metadata in all_metadata.values():
        diagnosis = metadata.diagnosis_value
        malignancy = metadata.malignancy_value
        malignancy_values_to_diagnosis_values[malignancy].add(diagnosis)
    print(malignancy_values_to_diagnosis_values)
    # The result {5: {1}, 4: {1}, 2: {0, 1}, 1: {0, 1}, 3: {0, 1}} shows that
    # malignancy value 5 and 4 are always diagnosis value 1, and malignancy
    # value 3, 2 and 1 can have diagnosis value 0 or 1. This is saying a
    # moderately unlikely malignant nodule can be malignant (LIDC-IDRI-1001)
    # and a Indeterminate nodule can be benign (LIDC-IDRI-0005).
    # So just classify the nodule to (benign/malignant) would be enough.


if __name__ == "__main__":
    # check_voi_contains_ct()
    # result: nothing from CT is equal to anything from VOI

    # check_voi_img_eq_mask()
    # result: two VOI images are not found
    # VOI image not found for LIDC-IDRI-1002_R_2.nii.gz
    # VOI image not found for LIDC-IDRI-1002_R_1.nii.gz
    # All images have mask

    # compare_voi_and_metadata()
    # file_ids-metadata_ids=set(), metadata_ids-file_ids={(1002, 1), (1002, 2)}
    # this result shows that all VOI files are in metadata, but two metadata are not in VOI
    # so when iter over VOI files, we can just assume the metadata exists

    malignancy_value_vs_diagnosis()
